from __future__ import annotations

"""V7 GPS DEVELOPMENT baselines: matched-head GRU vs actual Mamba.

No model is silently substituted. Both backbones use the same input
projection, target heads, Beta evidential head, 64-frame input window,
loss and training protocol. Parameter counts are reported; equal hidden
width does not imply parameter-matched models.

The exported evidence value U=2/(alpha+beta) is a Beta concentration
proxy, NOT established epistemic uncertainty or calibrated sensor noise.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BetaEvidenceHead(nn.Module):
    def __init__(self, hidden: int = 96, heads: int = 3):
        super().__init__()
        if hidden < 1 or heads < 1:
            raise ValueError("hidden and heads must be positive")
        self.heads = heads
        self.linear = nn.Linear(hidden, 2 * heads)

    def forward(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        raw = self.linear(h).reshape(h.shape[0], self.heads, 2)
        evidence = F.softplus(raw) + 1e-4
        alpha = evidence[..., 0] + 1.0
        beta = evidence[..., 1] + 1.0
        return {
            "alpha": alpha,
            "beta": beta,
            "reliability": alpha / (alpha + beta),
            "uncertainty": 2.0 / (alpha + beta),
        }


class ReliabilityForecastV7(nn.Module):
    """One of two explicit backends: GRU or Mamba, never a silent fallback."""

    def __init__(
        self, *, backend: str, input_dim: int = 21,
        hidden: int = 96, layers: int = 2, horizon: int = 3,
        dropout: float = 0.1, d_state: int = 16,
        d_conv: int = 4, expand: int = 2,
    ):
        super().__init__()
        if backend not in {"gru", "mamba"}:
            raise ValueError("backend must be exactly 'gru' or 'mamba'")
        if input_dim < 1 or hidden < 1 or layers < 1 or horizon < 1:
            raise ValueError("invalid model dimensions")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0,1)")
        self.backend = backend
        self.input_proj = nn.Linear(input_dim, hidden)
        self.dropout = nn.Dropout(dropout)
        if backend == "gru":
            self.gru = nn.GRU(
                hidden, hidden, num_layers=layers, batch_first=True,
                dropout=dropout if layers > 1 else 0.0,
            )
            self.mamba_blocks = None
            self.mamba_norms = None
            self.final_norm = nn.LayerNorm(hidden)
        else:
            try:
                from mamba_ssm import Mamba
            except Exception as exc:
                raise RuntimeError(
                    "backend=mamba requires working mamba_ssm + CUDA kernels; "
                    "install/repair the actual Mamba dependency. "
                    "V7 will NOT fall back to GRU."
                ) from exc
            self.gru = None
            self.mamba_blocks = nn.ModuleList([
                Mamba(
                    d_model=hidden, d_state=d_state,
                    d_conv=d_conv, expand=expand,
                ) for _ in range(layers)
            ])
            self.mamba_norms = nn.ModuleList([
                nn.LayerNorm(hidden) for _ in range(layers)
            ])
            self.final_norm = nn.LayerNorm(hidden)
        self.head = BetaEvidenceHead(hidden=hidden, heads=horizon)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError("input should have shape (batch,window,features)")
        h = self.input_proj(x)
        if self.backend == "gru":
            h, _ = self.gru(h)
        else:
            assert self.mamba_blocks is not None
            assert self.mamba_norms is not None
            for block, norm in zip(self.mamba_blocks, self.mamba_norms):
                h = self.dropout(norm(h + block(h)))
        h = self.final_norm(h[:, -1])
        return self.head(h)


def evidential_loss(
    output: dict[str, torch.Tensor], y: torch.Tensor,
    *, regularizer: float = 1e-3,
) -> torch.Tensor:
    """Equal objective for GRU and Mamba; no special-cased head.

    Beta NLL is a proper continuous score for an interpreted Beta
    predictive density, but V7's evidence proxy is not yet calibrated.
    Ground truth is used ONLY as offline supervised loss.
    """
    if y.ndim != 2 or not 0.0 <= regularizer <= 1.0:
        raise ValueError("invalid target or regularization")
    alpha, beta = output["alpha"], output["beta"]
    r = output["reliability"]
    if alpha.shape != y.shape or beta.shape != y.shape:
        raise ValueError("prediction and target horizon mismatch")
    target = y.clamp(1e-4, 1.0 - 1e-4)
    nll = -torch.distributions.Beta(alpha, beta).log_prob(target)
    if y.shape[1] == 3:
        weights = y.new_tensor([1.0, 0.7, 0.5])
    else:
        weights = y.new_ones((y.shape[1],))
    weights = weights / weights.sum()
    predicted_evidence = alpha + beta - 2.0
    penalty = (torch.abs(y - r).detach() * predicted_evidence).mean()
    return (nll * weights[None, :]).sum(dim=1).mean() + regularizer * penalty


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
