"""Shape, leakage, loss, and checkpoint checks using synthetic tensors only."""

import tempfile
import unittest
from pathlib import Path

import torch

from src.models.baselines import GraphGRU, TimeFrequencyFusion
from src.training import fit, load_checkpoint, soft_target_kl


class DeepBaselineTests(unittest.TestCase):
    def test_fusion_backward(self):
        model = TimeFrequencyFusion(eeg_channels=3, width=8, embedding_dim=12)
        batch = {"eeg": torch.randn(2, 3, 128),
                 "spectrogram": torch.randn(2, 3, 16, 16)}
        logits = model(batch)
        self.assertEqual(tuple(logits.shape), (2, 6))
        logits.square().mean().backward()
        self.assertTrue(any(p.grad is not None and bool((p.grad != 0).any())
                            for p in model.parameters()))

    def test_graph_forecast_does_not_read_future_target(self):
        torch.manual_seed(7)
        model = GraphGRU(node_features=4, hidden_dim=8, forecast_steps=2)
        model.eval()
        batch = {"graph_nodes": torch.randn(2, 3, 5, 4),
                 "graph_adj": torch.rand(2, 3, 5, 5),
                 "graph_mask": torch.ones(2, 3, 5, dtype=torch.bool),
                 "future_graph_nodes": torch.randn(2, 2, 5, 4)}
        logits_a, extra_a = model(batch, return_dynamics=True)
        batch["future_graph_nodes"] = torch.full_like(batch["future_graph_nodes"], 999)
        logits_b, extra_b = model(batch, return_dynamics=True)
        self.assertTrue(torch.equal(logits_a, logits_b))
        self.assertTrue(torch.equal(extra_a["predicted_graph_nodes"],
                                    extra_b["predicted_graph_nodes"]))

    def test_soft_kl_and_checkpoint_roundtrip(self):
        torch.manual_seed(11)
        model = GraphGRU(node_features=2, hidden_dim=8, forecast_steps=1)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        target = torch.tensor([[0.5, 0.5, 0, 0, 0, 0]], dtype=torch.float32)
        self.assertAlmostEqual(float(soft_target_kl(torch.log(target.clamp_min(1e-8)),
                                                    target)), 0, places=6)
        batch = {"graph_nodes": torch.randn(1, 2, 3, 2),
                 "graph_adj": torch.eye(3).repeat(1, 2, 1, 1),
                 "graph_mask": torch.ones(1, 2, 3, dtype=torch.bool),
                 "future_graph_nodes": torch.randn(1, 1, 3, 2),
                 "future_graph_mask": torch.ones(1, 1, 3, dtype=torch.bool),
                 "target": target}
        with tempfile.TemporaryDirectory() as tmp:
            history = fit(model, [batch], [batch], optimizer, tmp,
                          {"synthetic_smoke": True}, max_epochs=1,
                          forecast_weight=0.1)
            self.assertEqual(len(history), 1)
            original = {k: v.clone() for k, v in model.state_dict().items()}
            restored = GraphGRU(node_features=2, hidden_dim=8, forecast_steps=1)
            checkpoint = load_checkpoint(Path(tmp) / "best.pt", restored,
                                         restore_rng=False)
            self.assertEqual(checkpoint["epoch"], 0)
            for key, value in restored.state_dict().items():
                self.assertTrue(torch.equal(value, original[key]))
            resumed_optimizer = torch.optim.Adam(restored.parameters(), lr=1e-3)
            continued = fit(restored, [batch], [batch], resumed_optimizer, tmp,
                            {"synthetic_smoke": True}, max_epochs=2,
                            forecast_weight=0.1,
                            resume_from=Path(tmp) / "latest.pt")
            self.assertEqual(continued[0]["epoch"], 1)

    def test_forecast_only_pretraining_does_not_require_labels(self):
        model = GraphGRU(node_features=2, hidden_dim=8, forecast_steps=1)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        batch = {"graph_nodes": torch.randn(1, 2, 3, 2),
                 "graph_adj": torch.eye(3).repeat(1, 2, 1, 1),
                 "graph_mask": torch.ones(1, 2, 3, dtype=torch.bool),
                 "future_graph_nodes": torch.randn(1, 1, 3, 2),
                 "future_graph_mask": torch.ones(1, 1, 3, dtype=torch.bool)}
        with tempfile.TemporaryDirectory() as tmp:
            history = fit(model, [batch], [batch], optimizer, tmp,
                          {"synthetic_smoke": True}, max_epochs=1,
                          classification_weight=0, forecast_weight=1)
            self.assertEqual(history[0]["monitor"], "forecast_mse")
            self.assertIsNone(history[0]["train"]["kl"])
            self.assertTrue((Path(tmp) / "best.pt").exists())


if __name__ == "__main__":
    unittest.main()
