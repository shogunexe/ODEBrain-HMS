"""ODEBrain-inspired HMS graph dynamics model.

This is an independent HMS adaptation of the architecture described in
Jia et al. (ICLR 2026), *not* a drop-in port of the official TUSZ code.
Source audit and exact differences: docs/odebrain_adaptation.md.
"""

from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class _GraphConv(nn.Module):
    """Dense weighted node aggregation for a small fixed electrode graph."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.self_linear = nn.Linear(input_dim, output_dim)
        self.neighbor_linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: Tensor, weights: Tensor, node_mask: Tensor) -> Tensor:
        pair_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
        weights = weights.clamp(min=0) * pair_mask
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        neighbors = torch.bmm(weights, x)
        return (self.self_linear(x) + self.neighbor_linear(neighbors)) * node_mask.unsqueeze(-1)


class _VectorField(nn.Module):
    """Autonomous node-wise ODE vector field, as in the official code."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.nfe = 0

    def forward(self, t: Tensor, z: Tensor) -> Tensor:
        self.nfe += 1
        return self.net(z)


class _GithubODEFunc(nn.Module):
    """ODEBRAIN commit 31e251e's ODEFunc, with a diagnostic call counter.

    The upstream vector field is an autonomous three-layer Tanh MLP with
    N(0, 0.1) linear weights and zero biases. It has no gate or decay term.
    """

    def __init__(self, ode_input: int, feat_dim: int):
        super().__init__()
        self.ode_input_size = ode_input
        self.feat_dim = feat_dim
        self.net = nn.Sequential(
            nn.Linear(ode_input, feat_dim), nn.Tanh(),
            nn.Linear(feat_dim, feat_dim), nn.Tanh(),
            nn.Linear(feat_dim, ode_input),
        )
        for module in self.net.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0, std=0.1)
                nn.init.constant_(module.bias, val=0)
        self.nfe = 0

    def forward(self, t: Tensor, y: Tensor) -> Tensor:
        self.nfe += 1
        return self.net(y)


class _GithubRawCNN(nn.Module):
    """Upstream CNNEncoder adapted from 12 to variable whole-second EEG windows.

    The two Conv2d/BatchNorm/ReLU/MaxPool blocks and 4800-to-36 FC match
    upstream. Adaptive pooling restores its fixed 3x50 FC input when HMS
    provides 50 seconds for classification or 5 seconds for forecasting.
    """

    def __init__(self, feature_dim: int = 36, samples_per_second: int = 200):
        super().__init__()
        self.samples_per_second = samples_per_second
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3, 3), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.adapt = nn.AdaptiveAvgPool2d((3, 50))
        self.fc = nn.Linear(32 * 3 * 50, feature_dim)

    def forward(self, raw_eeg: Tensor) -> Tensor:
        if raw_eeg.ndim != 3 or raw_eeg.shape[-1] % self.samples_per_second:
            raise ValueError("raw EEG must be [B,N,whole seconds at 200 Hz]")
        batch, nodes, samples = raw_eeg.shape
        if samples < 4 * self.samples_per_second:
            raise ValueError("raw CNN requires at least four seconds of EEG")
        x = raw_eeg.reshape(batch * nodes, 1, samples // self.samples_per_second,
                            self.samples_per_second)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.adapt(x).reshape(batch * nodes, -1)
        return self.fc(x).reshape(batch, nodes, -1)


class ODEBrainInspiredHMS(nn.Module):
    """Causal graph forecast with a six-class HMS classification head.

    Required batch: ``graph_nodes[B,L,N,F]``, ``graph_adj[B,L,N,N]`` and
    ``graph_mask[B,L,N]``. Optional ``eeg[B,N,T]`` (offline classification),
    ``observed_eeg[B,N,T_past]`` (strict forecasting),
    ``future_graph_nodes[B,H,N,F]`` and ``future_graph_mask[B,H,N]``.
    Future tensors are never read by the encoder or solver. For forecast
    batches, ``eeg`` is intentionally ignored; provide ``observed_eeg``
    explicitly to use the raw branch with verified pre-cutoff samples.

    ``forward(..., False)`` returns logits[B,6]. ``forward(..., True)``
    returns ``(logits, dynamics)``. Predictions live in
    ``dynamics['predicted_graph_nodes']`` [B,H,N,F].
    """

    def __init__(self, node_features: int, num_nodes: int, hidden_dim: int = 64,
                 raw_dim: int = 32, num_classes: int = 6,
                 rk4_step_seconds: float = 0.25, dropout: float = 0.1,
                 architecture: str = "legacy_v1"):
        super().__init__()
        if min(node_features, num_nodes, hidden_dim, raw_dim, num_classes) <= 0:
            raise ValueError("all dimensions must be positive")
        if rk4_step_seconds <= 0:
            raise ValueError("rk4_step_seconds must be positive")
        if architecture not in ("legacy_v1", "github_trajectory_v2"):
            raise ValueError("unsupported ODEBrain HMS architecture")
        if architecture == "github_trajectory_v2" and raw_dim != 36:
            raise ValueError("GitHub CNN path requires raw_dim=36")
        self.architecture = architecture
        self.node_features = node_features
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.rk4_step_seconds = float(rk4_step_seconds)
        self.node_encoder = nn.GRUCell(node_features, hidden_dim)
        self.edge_encoder = nn.GRUCell(1, hidden_dim)
        self.edge_weight = nn.Linear(hidden_dim, 1)
        self.graph_conv1 = _GraphConv(hidden_dim, hidden_dim)
        self.graph_conv2 = _GraphConv(hidden_dim, hidden_dim)
        if architecture == "github_trajectory_v2":
            self.raw_encoder = _GithubRawCNN(feature_dim=raw_dim)
            self.vector_field = _GithubODEFunc(hidden_dim + raw_dim,
                                                hidden_dim + raw_dim)
        else:
            self.raw_encoder = nn.Sequential(
                nn.Conv1d(1, raw_dim, kernel_size=15, stride=4, padding=7),
                nn.GELU(), nn.Conv1d(raw_dim, raw_dim, kernel_size=7, stride=2, padding=3),
                nn.GELU(), nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            )
            self.vector_field = _VectorField(hidden_dim + raw_dim)
        self.forecast_head = nn.Linear(hidden_dim + raw_dim, node_features)
        if architecture == "github_trajectory_v2":
            self.classifier = nn.Sequential(nn.Dropout(dropout), nn.ReLU(),
                                            nn.Linear(hidden_dim + raw_dim, num_classes))
        else:
            self.classifier = nn.Sequential(nn.Dropout(dropout),
                                            nn.Linear(hidden_dim + raw_dim, num_classes))

    def _validate(self, batch: Dict[str, Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        nodes = batch["graph_nodes"]
        adj = batch["graph_adj"]
        mask = batch["graph_mask"]
        if nodes.ndim != 4:
            raise ValueError("graph_nodes must have shape [B,L,N,F]")
        b, l, n, f = nodes.shape
        if l == 0 or n != self.num_nodes or f != self.node_features:
            raise ValueError("graph_nodes dimensions do not match model")
        if tuple(adj.shape) != (b, l, n, n) or tuple(mask.shape) != (b, l, n):
            raise ValueError("graph_adj or graph_mask shape mismatch")
        if not nodes.is_floating_point() or not adj.is_floating_point():
            raise TypeError("graph nodes and adjacency must be floating tensors")
        mask = mask.to(dtype=nodes.dtype)
        if not torch.isfinite(nodes).all() or not torch.isfinite(adj).all():
            raise ValueError("graph inputs contain NaN or Inf")
        return nodes, adj, mask

    def _encode(self, nodes: Tensor, adj: Tensor, mask: Tensor,
                raw_eeg: Optional[Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        b, l, n, f = nodes.shape
        # Reverse temporal encoding follows the official initialization path.
        node_h = nodes.new_zeros(b, n, self.hidden_dim)
        edge_h = nodes.new_zeros(b, n, n, self.hidden_dim)
        for i in range(l - 1, -1, -1):
            valid = mask[:, i]
            next_node = self.node_encoder(nodes[:, i].reshape(-1, f),
                                          node_h.reshape(-1, self.hidden_dim)).reshape(b, n, -1)
            node_h = torch.where(valid.unsqueeze(-1) > 0, next_node, node_h)
            pair = valid.unsqueeze(1) * valid.unsqueeze(2)
            next_edge = self.edge_encoder(adj[:, i].reshape(-1, 1),
                                          edge_h.reshape(-1, self.hidden_dim)).reshape(b, n, n, -1)
            edge_h = torch.where(pair.unsqueeze(-1) > 0, next_edge, edge_h)
        valid_nodes = (mask.sum(dim=1) > 0).to(dtype=nodes.dtype)
        learned_edges = torch.sigmoid(self.edge_weight(edge_h).squeeze(-1))
        # Latest observed adjacency gives the support; learned edge histories
        # modulate rather than replace it.
        weights = adj[:, -1].clamp(min=0) * learned_edges
        g = torch.tanh(self.graph_conv1(node_h, weights, valid_nodes))
        g = torch.tanh(self.graph_conv2(g, weights, valid_nodes))
        if raw_eeg is None:
            raw = nodes.new_zeros(b, n, 36 if self.architecture == "github_trajectory_v2"
                                  else self.raw_encoder[0].out_channels)
        else:
            if raw_eeg.ndim != 3 or raw_eeg.shape[0] != b or raw_eeg.shape[1] != n:
                raise ValueError("raw EEG must have shape [B,N,T]")
            if raw_eeg.shape[2] == 0:
                raise ValueError("raw EEG must contain samples")
            if self.architecture == "github_trajectory_v2":
                raw = self.raw_encoder(raw_eeg)
            else:
                raw = self.raw_encoder(raw_eeg.reshape(b*n, 1, -1)).reshape(b, n, -1)
            raw = raw * valid_nodes.unsqueeze(-1)
        return torch.cat([g, raw], dim=-1), valid_nodes, weights

    def _integrate(self, z0: Tensor, times_s: Tensor) -> Tensor:
        if times_s.ndim != 1 or len(times_s) == 0 or times_s[0].item() != 0:
            raise ValueError("times_s must be one-dimensional and start at zero")
        if bool(torch.any(times_s[1:] <= times_s[:-1])):
            raise ValueError("times_s must be strictly increasing")
        self.vector_field.nfe = 0
        if self.architecture == "github_trajectory_v2":
            from torchdiffeq import odeint_adjoint
            trajectory = odeint_adjoint(self.vector_field, z0.float(), times_s,
                                       method="rk4")
            return trajectory.transpose(0, 1)
        states = [z0]
        z = z0.float()  # Keep integration in FP32 under autocast.
        field = self.vector_field
        for end in times_s[1:]:
            start = times_s[len(states)-1]
            duration = (end - start).item()
            steps = max(1, int(torch.ceil(torch.tensor(duration / self.rk4_step_seconds)).item()))
            dt = duration / steps
            t = start
            for _ in range(steps):
                k1 = field(t, z)
                k2 = field(t + dt/2, z + (dt/2)*k1)
                k3 = field(t + dt/2, z + (dt/2)*k2)
                k4 = field(t + dt, z + dt*k3)
                z = z + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
                t = t + dt
            states.append(z)
        return torch.stack(states, dim=1)  # [B,T,N,D]

    def forward(self, batch: Dict[str, Tensor], return_dynamics: bool = False):
        nodes, adj, mask = self._validate(batch)
        forecasting = "future_graph_nodes" in batch
        raw_eeg = batch.get("observed_eeg") if forecasting else batch.get("eeg")
        if raw_eeg is not None:
            raw_eeg = raw_eeg.to(dtype=nodes.dtype)
        z0, valid_nodes, weights = self._encode(nodes, adj, mask, raw_eeg)
        if forecasting:
            future = batch["future_graph_nodes"]
            if future.ndim != 4 or future.shape[0] != nodes.shape[0] or future.shape[2:] != nodes.shape[2:]:
                raise ValueError("future_graph_nodes must have shape [B,H,N,F]")
            h = future.shape[1]
            future_times = batch.get("future_times_s")
            if future_times is None:
                future_times = torch.arange(1, h+1, device=nodes.device, dtype=torch.float32)
            else:
                future_times = future_times.to(device=nodes.device, dtype=torch.float32)
            if future_times.ndim != 1 or future_times.numel() != h:
                raise ValueError("future_times_s must have shape [H]")
            if bool(torch.any(future_times <= 0)) or bool(torch.any(future_times[1:] <= future_times[:-1])):
                raise ValueError("future_times_s must be positive and strictly increasing")
        else:
            requested = batch.get("dynamics_times_s")
            if requested is None:
                requested = torch.tensor([0., 1., 3., 5.], device=nodes.device)
            else:
                requested = requested.to(device=nodes.device, dtype=torch.float32)
            if requested.ndim != 1 or requested.numel() == 0 or bool(torch.any(requested < 0)):
                raise ValueError("dynamics_times_s must be a nonempty nonnegative 1D tensor")
            future_times = requested[requested > 0]
        class_time = torch.tensor([1.], device=nodes.device)
        zero_time = torch.tensor([0.], device=nodes.device)
        if self.architecture == "github_trajectory_v2":
            # Match upstream's 12 RK4 trajectory samples over [0, 1]. The
            # classifier consumes their time mean, not the final state.
            class_grid = torch.linspace(0., 1., 12, device=nodes.device)
            if return_dynamics:
                max_time = float(future_times.max().item()) if len(future_times) else 1.
                extra = (torch.arange(1. + self.rk4_step_seconds, max_time,
                                      self.rk4_step_seconds, device=nodes.device)
                         if max_time > 1. else zero_time[:0])
                times = torch.unique(torch.cat([class_grid, extra, future_times]),
                                     sorted=True)
            else:
                times = class_grid
        elif return_dynamics:
            times = torch.unique(torch.cat([zero_time, class_time, future_times]), sorted=True)
        else:
            times = torch.cat([zero_time, class_time])
        trajectory = self._integrate(z0, times)
        if self.architecture == "github_trajectory_v2":
            class_indices = torch.searchsorted(times, class_grid)
            class_nodes = trajectory.index_select(1, class_indices).mean(dim=1)
        else:
            class_index = int(torch.searchsorted(times, class_time)[0].item())
            class_nodes = trajectory[:, class_index]
        if self.architecture == "github_trajectory_v2":
            node_logits = self.classifier(class_nodes)
            logits = (node_logits * valid_nodes.unsqueeze(-1)).sum(dim=1) / valid_nodes.sum(dim=1, keepdim=True).clamp(min=1)
        else:
            pooled = (class_nodes * valid_nodes.unsqueeze(-1)).sum(dim=1) / valid_nodes.sum(dim=1, keepdim=True).clamp(min=1)
            logits = self.classifier(pooled)
        if not return_dynamics:
            return logits
        future_indices = torch.searchsorted(times, future_times)
        predicted = self.forecast_head(trajectory.index_select(1, future_indices)) * valid_nodes[:, None, :, None]
        future_mask = batch.get("future_graph_mask")
        if future_mask is not None:
            if tuple(future_mask.shape) != tuple(predicted.shape[:3]):
                raise ValueError("future_graph_mask must have shape [B,H,N]")
            future_mask = future_mask.to(dtype=nodes.dtype)
        else:
            future_mask = valid_nodes[:, None, :].expand_as(predicted[..., 0])
        derivatives = torch.stack([self.vector_field(t, trajectory[:, i])
                                   for i, t in enumerate(times)], dim=1)
        dynamics = {
            "z0": z0,
            "trajectory": trajectory,
            "times_s": times,
            "vector_field": derivatives,
            "predicted_graph_nodes": predicted,
            "future_graph_mask": future_mask,
            "observed_node_mask": valid_nodes,
            "edge_weights": weights,
            "raw_branch_used": raw_eeg is not None,
            "nfe": self.vector_field.nfe,
        }
        return logits, dynamics


def masked_forecast_mse(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """Mean squared error over observed future node attributes only."""
    if prediction.shape != target.shape or mask.shape != target.shape[:-1]:
        raise ValueError("forecast prediction, target or mask shape mismatch")
    valid = mask.unsqueeze(-1) > 0
    if not torch.isfinite(prediction).all() or not torch.isfinite(target[valid.expand_as(target)]).all():
        raise ValueError("valid forecast attributes contain NaN or Inf")
    error = torch.where(valid, prediction - target, torch.zeros_like(prediction))
    count = valid.to(dtype=prediction.dtype).sum() * prediction.shape[-1]
    return error.square().sum() / count.clamp(min=1)
