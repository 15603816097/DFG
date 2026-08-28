"""
Predictive Reliability Model V3
===============================

Dual-head temporal reliability predictor.

Shared temporal encoder:
    MambaEncoder if available
    otherwise GRU fallback

Heads:
    current_head
        estimates R_current(t)

    future_head
        estimates future-window reliability

Why two heads?
--------------
The current head stabilizes temporal representation learning.
The future head learns degradation trend and early warning behavior.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PredictiveReliabilityModel(
    nn.Module
):
    def __init__(
        self,
        input_dim=30,
        hidden_dim=128,
        num_layers=2,
        dropout=0.10,
    ):
        super().__init__()

        self.backend = "gru"
        self.encoder = None

        try:
            from src.model.mamba_encoder import (
                MambaEncoder
            )

            candidates = [
                {
                    "input_dim":
                        input_dim,
                    "hidden_dim":
                        hidden_dim,
                    "num_layers":
                        num_layers,
                },
                {
                    "input_dim":
                        input_dim,
                    "d_model":
                        hidden_dim,
                    "num_layers":
                        num_layers,
                },
            ]

            for kwargs in candidates:
                try:
                    self.encoder = (
                        MambaEncoder(
                            **kwargs
                        )
                    )

                    self.backend = (
                        "mamba"
                    )

                    break

                except TypeError:
                    pass

        except Exception:
            self.encoder = None

        if self.encoder is None:
            self.encoder = nn.GRU(
                input_dim,
                hidden_dim,
                num_layers=
                    num_layers,
                batch_first=True,
                dropout=(
                    dropout
                    if num_layers > 1
                    else 0.0
                ),
            )

        self.current_head = nn.Sequential(
            nn.Linear(
                hidden_dim,
                64,
            ),
            nn.ReLU(),
            nn.Dropout(
                dropout
            ),
            nn.Linear(
                64,
                1,
            ),
            nn.Sigmoid(),
        )

        self.future_head = nn.Sequential(
            nn.Linear(
                hidden_dim,
                64,
            ),
            nn.ReLU(),
            nn.Dropout(
                dropout
            ),
            nn.Linear(
                64,
                1,
            ),
            nn.Sigmoid(),
        )

    def encode(
        self,
        x,
    ):
        if self.backend == "mamba":
            h = self.encoder(
                x
            )

            if isinstance(
                h,
                (
                    tuple,
                    list,
                ),
            ):
                h = h[0]

            if h.ndim == 3:
                h = h[
                    :,
                    -1,
                    :
                ]

        else:
            h, _ = self.encoder(
                x
            )

            h = h[
                :,
                -1,
                :
            ]

        return h

    def forward(
        self,
        x,
    ):
        h = self.encode(
            x
        )

        current = (
            self.current_head(
                h
            )
            .squeeze(
                -1
            )
        )

        future = (
            self.future_head(
                h
            )
            .squeeze(
                -1
            )
        )

        return {
            "current":
                current,

            "future":
                future,

            "embedding":
                h,
        }
