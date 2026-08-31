from __future__ import annotations

import torch
import torch.nn as nn


class SensorSpecificPredictor(nn.Module):
    """
    Independent per-sensor temporal predictor.

    Input:
        x: (B, T, D)
    Output:
        current, future: (B,)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 96,
        num_layers: int = 2,
        dropout: float = 0.10,
    ):
        super().__init__()

        # Prefer Mamba when available; fall back to GRU without changing API.
        self.backend = "gru"
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        try:
            from mamba_ssm import Mamba
            blocks = []
            for _ in range(num_layers):
                blocks.append(
                    nn.ModuleDict(
                        {
                            "norm": nn.LayerNorm(hidden_dim),
                            "mamba": Mamba(
                                d_model=hidden_dim,
                                d_state=16,
                                d_conv=4,
                                expand=2,
                            ),
                        }
                    )
                )
            self.blocks = nn.ModuleList(blocks)
            self.rnn = None
            self.backend = "mamba"
        except Exception:
            self.blocks = None
            self.rnn = nn.GRU(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )

        self.dropout = nn.Dropout(dropout)
        self.current_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        self.future_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        h = self.input_proj(x)

        if self.backend == "mamba":
            for block in self.blocks:
                z = block["norm"](h)
                h = h + block["mamba"](z)
        else:
            h, _ = self.rnn(h)

        z = self.dropout(h[:, -1])
        return {
            "current": self.current_head(z).squeeze(-1),
            "future": self.future_head(z).squeeze(-1),
        }
