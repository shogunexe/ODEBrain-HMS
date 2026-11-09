"""High-value shape, gradient, causality and RK4 checks for the HMS adapter."""

import torch

from src.models.odebrain_hms import ODEBrainInspiredHMS, masked_forecast_mse


def _batch(batch_size=2):
    torch.manual_seed(7)
    return {
        "graph_nodes": torch.randn(batch_size, 4, 3, 5),
        "graph_adj": torch.rand(batch_size, 4, 3, 3),
        "graph_mask": torch.ones(batch_size, 4, 3),
        "future_graph_nodes": torch.randn(batch_size, 2, 3, 5),
        "future_graph_mask": torch.ones(batch_size, 2, 3),
        "future_times_s": torch.tensor([1., 3.]),
        "observed_eeg": torch.randn(batch_size, 3, 100),
    }


def test_forward_backward_and_mask():
    model = ODEBrainInspiredHMS(5, 3, hidden_dim=8, raw_dim=4, rk4_step_seconds=.5)
    batch = _batch()
    batch["graph_mask"][0, :, 2] = 0
    batch["future_graph_mask"][0, :, 2] = 0
    logits, dyn = model(batch, return_dynamics=True)
    assert logits.shape == (2, 6)
    assert dyn["trajectory"].shape == (2, 3, 3, 12)
    assert dyn["predicted_graph_nodes"].shape == (2, 2, 3, 5)
    assert torch.all(dyn["predicted_graph_nodes"][0, :, 2] == 0)
    assert torch.allclose(dyn["times_s"], torch.tensor([0., 1., 3.]))
    loss = logits.square().mean() + masked_forecast_mse(
        dyn["predicted_graph_nodes"], batch["future_graph_nodes"], batch["future_graph_mask"])
    loss.backward()
    assert model.vector_field.net[0].weight.grad.abs().sum() > 0
    assert model.node_encoder.weight_ih.grad.abs().sum() > 0
    assert model.forecast_head.weight.grad.abs().sum() > 0


def test_future_targets_and_full_eeg_do_not_enter_forecast():
    model = ODEBrainInspiredHMS(5, 3, hidden_dim=8, raw_dim=4, dropout=0).eval()
    batch = _batch(1)
    batch["eeg"] = torch.randn(1, 3, 200)  # ignored in strict forecast mode
    with torch.no_grad():
        logits, dyn = model(batch, True)
        modified = dict(batch)
        modified["future_graph_nodes"] = 1000 * torch.randn_like(batch["future_graph_nodes"])
        modified["eeg"] = 1000 * torch.randn_like(batch["eeg"])
        logits2, dyn2 = model(modified, True)
    assert torch.equal(logits, logits2)
    assert torch.equal(dyn["predicted_graph_nodes"], dyn2["predicted_graph_nodes"])


def test_rk4_step_refinement():
    torch.manual_seed(3)
    model = ODEBrainInspiredHMS(5, 3, hidden_dim=8, raw_dim=4, dropout=0).eval()
    batch = _batch(1)
    with torch.no_grad():
        model.rk4_step_seconds = .5
        _, coarse = model(batch, True)
        model.rk4_step_seconds = .25
        _, fine = model(batch, True)
    assert torch.allclose(coarse["trajectory"], fine["trajectory"], atol=1e-3, rtol=1e-3)


def test_classification_only_uses_ode_and_allows_full_eeg():
    model = ODEBrainInspiredHMS(5, 3, hidden_dim=8, raw_dim=4, dropout=0)
    batch = _batch(1)
    del batch["future_graph_nodes"]
    del batch["future_graph_mask"]
    del batch["future_times_s"]
    del batch["observed_eeg"]
    batch["eeg"] = torch.randn(1, 3, 200)
    logits = model(batch)
    assert logits.shape == (1, 6)
    logits.square().sum().backward()
    assert model.vector_field.net[0].weight.grad.abs().sum() > 0
    assert model.raw_encoder[0].weight.grad.abs().sum() > 0


def test_masked_future_nan_is_excluded():
    prediction = torch.tensor([[[[1., 2.], [3., 4.]]]])
    target = torch.tensor([[[[0., 0.], [float("nan"), float("nan")]]]])
    mask = torch.tensor([[[1., 0.]]])
    assert torch.allclose(masked_forecast_mse(prediction, target, mask), torch.tensor(2.5))
