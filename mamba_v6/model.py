
from __future__ import annotations
import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
    MAMBA_IMPORT_ERROR = None
except Exception as exc:
    Mamba = None
    MAMBA_AVAILABLE = False
    MAMBA_IMPORT_ERROR = repr(exc)


class BetaEvidentialHead(nn.Module):
    def __init__(self, hidden_dim: int, horizon: int):
        super().__init__()
        self.horizon = horizon
        self.proj = nn.Linear(hidden_dim, 2 * horizon)
        self.softplus = nn.Softplus()

    def forward(self, x):
        raw = self.proj(x)
        raw = raw.view(x.shape[0], self.horizon, 2)
        evidence = self.softplus(raw) + 1e-4
        alpha = evidence[..., 0] + 1.0
        beta = evidence[..., 1] + 1.0
        reliability = alpha / (alpha + beta)
        uncertainty = 2.0 / (alpha + beta)
        return {
            "alpha": alpha,
            "beta": beta,
            "reliability": reliability,
            "uncertainty": uncertainty,
        }


class MambaReliabilityUncertaintyV6(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 96,
        horizon: int = 3,
        num_layers: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        if not MAMBA_AVAILABLE:
            raise RuntimeError(
                "mamba_ssm is not available. Import error: "
                + str(MAMBA_IMPORT_ERROR)
            )

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList([
            Mamba(
                d_model=hidden_dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.head = BetaEvidentialHead(hidden_dim, horizon)

    def forward(self, x):
        h = self.input_proj(x)
        for layer, norm in zip(self.layers, self.norms):
            residual = h
            h = layer(h)
            h = norm(h + residual)
            h = self.dropout(h)
        return self.head(h[:, -1, :])
