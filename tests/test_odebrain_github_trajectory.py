"""Regression checks for the upstream-inspired trajectory/CNN/ODE path."""

import torch

from src.dynamics.export import load_trained_model
from src.models.odebrain_hms import ODEBrainInspiredHMS


def _model():
    torch.manual_seed(12)
    return ODEBrainInspiredHMS(5, 3, hidden_dim=8, raw_dim=36,
                               architecture="github_trajectory_v2", dropout=0)


def _batch(forecast=False):
    torch.manual_seed(14)
    batch = {
        "graph_nodes": torch.randn(1, 5, 3, 5),
        "graph_adj": torch.rand(1, 5, 3, 3),
        "graph_mask": torch.ones(1, 5, 3),
    }
    if forecast:
        batch.update({"observed_eeg": torch.randn(1, 3, 5 * 200),
                      "future_graph_nodes": torch.randn(1, 3, 3, 5),
                      "future_graph_mask": torch.ones(1, 3, 3),
                      "future_times_s": torch.tensor([1., 3., 5.])})
    else:
        batch["eeg"] = torch.randn(1, 3, 50 * 200)
    return batch


def test_github_path_uses_raw_cnn_and_trajectory_mean_for_classification():
    model = _model().eval()
    batch = _batch()
    assert isinstance(model.raw_encoder.conv1, torch.nn.Conv2d)
    assert isinstance(model.raw_encoder.conv2, torch.nn.Conv2d)
    assert model.raw_encoder.fc.in_features == 32 * 3 * 50
    assert model.raw_encoder.fc.out_features == 36
    assert len([m for m in model.vector_field.net if isinstance(m, torch.nn.Linear)]) == 3
    assert all(torch.count_nonzero(m.bias) == 0 for m in model.vector_field.net
               if isinstance(m, torch.nn.Linear))
    with torch.no_grad():
        logits, dyn = model(batch, return_dynamics=True)
        class_nodes = dyn["trajectory"][:, :12].mean(dim=1)
        expected = model.classifier(class_nodes).mean(dim=1)
        direct = model(batch)
    assert dyn["trajectory"].shape[1] > 12
    assert torch.allclose(dyn["times_s"][:12], torch.linspace(0., 1., 12))
    assert torch.allclose(logits, expected, atol=1e-6)
    assert torch.allclose(direct, expected, atol=1e-6)
    assert dyn["raw_branch_used"]


def test_github_path_forecast_is_past_only_and_backpropagates():
    model = _model().eval()
    batch = _batch(forecast=True)
    batch["eeg"] = torch.randn(1, 3, 50 * 200)
    logits, dyn = model(batch, return_dynamics=True)
    assert dyn["predicted_graph_nodes"].shape == (1, 3, 3, 5)
    assert torch.allclose(dyn["times_s"][:12], torch.linspace(0., 1., 12))
    modified = dict(batch)
    modified["future_graph_nodes"] = torch.randn_like(batch["future_graph_nodes"]) * 100
    modified["eeg"] = torch.randn_like(batch["eeg"]) * 100
    with torch.no_grad():
        logits2, dyn2 = model(modified, return_dynamics=True)
    assert torch.equal(logits, logits2)
    assert torch.equal(dyn["predicted_graph_nodes"], dyn2["predicted_graph_nodes"])
    loss = logits.square().mean() + dyn["predicted_graph_nodes"].square().mean()
    loss.backward()
    assert model.vector_field.net[0].weight.grad.abs().sum() > 0
    assert model.raw_encoder.conv1.weight.grad.abs().sum() > 0
    assert model.node_encoder.weight_ih.grad.abs().sum() > 0


def test_github_path_forecast_without_raw_eeg_uses_zero_branch():
    model = _model().eval()
    batch = _batch(forecast=True)
    del batch["observed_eeg"]
    with torch.no_grad():
        _, dyn = model(batch, return_dynamics=True)
    assert dyn["predicted_graph_nodes"].shape == (1, 3, 3, 5)
    assert not dyn["raw_branch_used"]


def test_github_checkpoint_loads_for_dynamics_export(tmp_path):
    model = ODEBrainInspiredHMS(5, 19, hidden_dim=8, raw_dim=36,
                                architecture="github_trajectory_v2")
    path = tmp_path / "checkpoint.pt"
    torch.save({"model": model.state_dict(), "config": {"model": {"kwargs": {
        "architecture": "github_trajectory_v2"}}}}, path)
    restored, _ = load_trained_model(path)
    assert restored.architecture == "github_trajectory_v2"
    assert restored.raw_encoder.fc.weight.shape == model.raw_encoder.fc.weight.shape
