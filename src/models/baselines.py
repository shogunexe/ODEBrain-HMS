"""HMS waveform, spectrogram, fusion, and discrete graph baselines.

All inputs use batch-first tensors. The graph baseline consumes observed history
only; ``future_graph_nodes`` is a training target and is never read in forward.
"""

from typing import Mapping, Optional, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _required(batch: Mapping[str, Tensor], key: str, ndim: int) -> Tensor:
    if key not in batch:
        raise KeyError(f"batch requires '{key}'")
    value = batch[key]
    if not isinstance(value, Tensor) or value.ndim != ndim:
        raise ValueError(f"batch['{key}'] must be a {ndim}D torch.Tensor")
    return value


def _groups(channels: int) -> int:
    for n in (8, 4, 2):
        if channels % n == 0:
            return n
    return 1


class MultiScaleTemporalBlock(nn.Module):
    def __init__(self, width: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            nn.Conv1d(width, width, kernel_size=k, padding=dilation * (k // 2),
                      dilation=dilation, groups=width)
            for k in (3, 7, 15)
        )
        self.mix = nn.Conv1d(3 * width, width, 1)
        self.norm = nn.GroupNorm(_groups(width), width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        y = torch.cat([conv(x) for conv in self.branches], dim=1)
        return F.gelu(x + self.dropout(self.norm(self.mix(y))))


class WaveformTCN(nn.Module):
    """M1: multiscale temporal CNN over the 50-second EEG waveform."""

    def __init__(self, eeg_channels: int, num_classes: int = 6,
                 width: int = 48, embedding_dim: int = 96,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(eeg_channels, width, 15, stride=4, padding=7),
            nn.GroupNorm(_groups(width), width), nn.GELU(),
        )
        self.blocks = nn.Sequential(*(MultiScaleTemporalBlock(width, d, dropout)
                                      for d in (1, 2, 4, 8)))
        self.embedding = nn.Sequential(nn.Linear(2 * width, embedding_dim), nn.GELU(),
                                       nn.Dropout(dropout))
        self.head = nn.Linear(embedding_dim, num_classes)

    def encode(self, batch: Mapping[str, Tensor]) -> Tensor:
        x = _required(batch, "eeg", 3)
        if "eeg_mask" in batch:
            mask = batch["eeg_mask"]
            if mask.ndim == 2:
                mask = mask.unsqueeze(-1)
            if mask.shape != x.shape and mask.shape != (*x.shape[:2], 1):
                raise ValueError("eeg_mask must have shape [B,C] or [B,C,T]")
            x = x * mask.to(x.dtype)
        x = self.blocks(self.stem(x))
        pooled = torch.cat((x.mean(-1), x.amax(-1)), dim=1)
        return self.embedding(pooled)

    def forward(self, batch: Mapping[str, Tensor], return_dynamics: bool = False):
        embedding = self.encode(batch)
        logits = self.head(embedding)
        return (logits, {"embedding": embedding}) if return_dynamics else logits


class SpectrogramCNN(nn.Module):
    """M2: compact 2D CNN for time-frequency images from the same EEG window."""

    def __init__(self, eeg_channels: int, num_classes: int = 6,
                 width: int = 32, embedding_dim: int = 96,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(eeg_channels, width, 5, stride=2, padding=2),
            nn.GroupNorm(_groups(width), width), nn.GELU(),
            nn.Conv2d(width, 2 * width, 3, stride=2, padding=1),
            nn.GroupNorm(_groups(2 * width), 2 * width), nn.GELU(),
            nn.Conv2d(2 * width, 3 * width, 3, stride=2, padding=1),
            nn.GroupNorm(_groups(3 * width), 3 * width), nn.GELU(),
        )
        self.embedding = nn.Sequential(nn.Linear(3 * width, embedding_dim), nn.GELU(),
                                       nn.Dropout(dropout))
        self.head = nn.Linear(embedding_dim, num_classes)

    def encode(self, batch: Mapping[str, Tensor]) -> Tensor:
        x = _required(batch, "spectrogram", 4)
        x = self.features(x)
        return self.embedding(x.mean(dim=(-2, -1)))

    def forward(self, batch: Mapping[str, Tensor], return_dynamics: bool = False):
        embedding = self.encode(batch)
        logits = self.head(embedding)
        return (logits, {"embedding": embedding}) if return_dynamics else logits


class TimeFrequencyFusion(nn.Module):
    """M3: gated fusion of independent waveform and spectrogram encoders."""

    def __init__(self, eeg_channels: int, num_classes: int = 6,
                 width: int = 32, embedding_dim: int = 96,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.waveform = WaveformTCN(eeg_channels, num_classes, width,
                                    embedding_dim, dropout)
        self.spectrogram = SpectrogramCNN(eeg_channels, num_classes, width,
                                          embedding_dim, dropout)
        self.gate = nn.Linear(2 * embedding_dim, embedding_dim)
        self.head = nn.Sequential(nn.Linear(2 * embedding_dim, embedding_dim),
                                  nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(embedding_dim, num_classes))

    def forward(self, batch: Mapping[str, Tensor], return_dynamics: bool = False):
        waveform = self.waveform.encode(batch)
        spectrogram = self.spectrogram.encode(batch)
        gate = torch.sigmoid(self.gate(torch.cat((waveform, spectrogram), dim=-1)))
        fused = torch.cat((gate * waveform, (1 - gate) * spectrogram), dim=-1)
        logits = self.head(fused)
        extras = {"embedding": fused, "fusion_gate": gate}
        return (logits, extras) if return_dynamics else logits


class GraphGRU(nn.Module):
    """M5: discrete spatial message passing and temporal GRU.

    Input nodes have shape [B,L,N,F], adjacency [B,L,N,N], and optional
    graph_mask [B,L,N]. Forecasts are autoregressive per-node attributes,
    conditioned only on observed input frames. The forecast head is a project
    baseline and does not claim to predict adjacency.
    """

    def __init__(self, node_features: int, num_classes: int = 6,
                 hidden_dim: int = 64, forecast_steps: int = 5,
                 dropout: float = 0.1,
                 forecast_horizons: Optional[Sequence[int]] = None) -> None:
        super().__init__()
        if forecast_steps < 1:
            raise ValueError("forecast_steps must be positive")
        horizons = tuple(range(1, forecast_steps + 1)) if forecast_horizons is None else tuple(forecast_horizons)
        if not horizons or any(not isinstance(h, int) or h < 1 for h in horizons) or tuple(sorted(set(horizons))) != horizons:
            raise ValueError("forecast_horizons must be strictly increasing positive integers")
        self.forecast_horizons = horizons
        self.node_features = node_features
        self.self_proj = nn.Linear(node_features, hidden_dim)
        self.neighbor_proj = nn.Linear(node_features, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.classifier = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                                        nn.Dropout(dropout),
                                        nn.Linear(hidden_dim, num_classes))
        self.forecast = nn.Linear(hidden_dim, node_features)
        self.forecast_input = nn.Linear(node_features, hidden_dim)
        self.forecast_gru = nn.GRUCell(hidden_dim, hidden_dim)

    def forward(self, batch: Mapping[str, Tensor], return_dynamics: bool = False):
        nodes = _required(batch, "graph_nodes", 4)
        adjacency = _required(batch, "graph_adj", 4)
        b, frames, n, features = nodes.shape
        if features != self.node_features:
            raise ValueError(f"expected {self.node_features} graph features, got {features}")
        if adjacency.shape != (b, frames, n, n):
            raise ValueError("graph_adj must have shape [B,L,N,N]")
        mask = batch.get("graph_mask")
        if mask is None:
            mask = torch.ones((b, frames, n), dtype=torch.bool, device=nodes.device)
        if mask.shape != (b, frames, n):
            raise ValueError("graph_mask must have shape [B,L,N]")
        mask = mask.bool()
        if not bool(mask.reshape(b, -1).any(dim=1).all()):
            raise ValueError("each graph sequence needs at least one valid node frame")
        x = nodes * mask.unsqueeze(-1).to(nodes.dtype)
        valid_edges = mask.unsqueeze(-1) & mask.unsqueeze(-2)
        weights = adjacency.masked_fill(~valid_edges, 0).clamp_min(0)
        eye = torch.eye(n, dtype=weights.dtype, device=weights.device)
        weights = weights + eye.view(1, 1, n, n) * mask.unsqueeze(-1)
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1)
        neighborhood = torch.matmul(weights, x)
        encoded = F.gelu(self.norm(self.self_proj(x) + self.neighbor_proj(neighborhood)))
        encoded = encoded * mask.unsqueeze(-1).to(encoded.dtype)
        # Each electrode keeps its own temporal state; invalid frames leave it unchanged.
        hidden = encoded.new_zeros(b * n, encoded.shape[-1])
        states_by_frame = []
        for frame in range(frames):
            candidate = self.gru(encoded[:, frame].reshape(b * n, -1), hidden)
            valid = mask[:, frame].reshape(b * n, 1)
            hidden = torch.where(valid, candidate, hidden)
            states_by_frame.append(hidden.reshape(b, n, -1))
        states = torch.stack(states_by_frame, dim=1)
        # Use the last valid state for each node, then average valid nodes.
        last_idx = torch.arange(frames, device=nodes.device).view(1, frames, 1)
        last_idx = torch.where(mask, last_idx, -1).amax(dim=1)
        valid_nodes = last_idx >= 0
        gather = last_idx.clamp_min(0).view(b, 1, n, 1).expand(-1, 1, -1, states.shape[-1])
        last_state = states.gather(1, gather).squeeze(1)
        pooled = (last_state * valid_nodes.unsqueeze(-1)).sum(1) / valid_nodes.sum(1, keepdim=True).clamp_min(1)
        logits = self.classifier(pooled)
        if not return_dynamics:
            return logits
        hidden = last_state.reshape(b * n, -1)
        predictions = []
        for step in range(1, max(self.forecast_horizons) + 1):
            predicted = self.forecast(hidden)
            if step in self.forecast_horizons:
                predictions.append(predicted.reshape(b, n, features))
            hidden = self.forecast_gru(self.forecast_input(predicted), hidden)
        extras = {
            "embedding": pooled,
            "predicted_graph_nodes": torch.stack(predictions, dim=1),
            "future_times_s": torch.tensor(self.forecast_horizons, dtype=nodes.dtype,
                                             device=nodes.device),
            "observed_graph_states": states,
            "graph_mask": mask,
        }
        if "graph_times" in batch:
            extras["graph_times"] = batch["graph_times"]
        return logits, extras


MODEL_REGISTRY = {
    "tcn": WaveformTCN,
    "spectrogram_cnn": SpectrogramCNN,
    "fusion": TimeFrequencyFusion,
    "graph_gru": GraphGRU,
}


def build_model(name: str, **kwargs) -> nn.Module:
    """Build a baseline from a config name and explicit dimensions."""
    try:
        return MODEL_REGISTRY[name](**kwargs)
    except KeyError as exc:
        raise ValueError(f"unknown baseline '{name}'") from exc
