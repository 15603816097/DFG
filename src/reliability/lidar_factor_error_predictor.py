from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


@dataclass(frozen=True)
class LidarFactorErrorModelConfig:
    """
    LiDAR-specific future factor-error predictor.

    Input:
        causal temporal window of LiDAR factor-quality / motion features.

    Output:
        two positive values for target frame t+H:
            translation factor error [m]
            rotation factor error [deg]

    The network intentionally predicts factor errors directly instead of a
    hand-crafted "reliability" scalar.
    """

    input_dim: int
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.10
    horizon: int = 3
    window: int = 20


class TemporalBlock(nn.Module):
    """
    Lightweight causal temporal backend.

    If mamba_ssm is installed, the outer model can use Mamba.
    Otherwise this GRU block is the stable CPU fallback.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, x):
        y, _ = self.gru(x)
        return y


class LidarFactorErrorPredictor(nn.Module):
    def __init__(
        self,
        config: LidarFactorErrorModelConfig,
    ):
        super().__init__()

        self.config = config
        self.backend_name = "gru"

        backend = None

        try:
            from mamba_ssm import Mamba

            self.input_proj = nn.Linear(
                config.input_dim,
                config.hidden_dim,
            )

            blocks = []

            for _ in range(
                config.num_layers
            ):
                blocks.append(
                    Mamba(
                        d_model=config.hidden_dim,
                        d_state=16,
                        d_conv=4,
                        expand=2,
                    )
                )

            self.mamba_blocks = nn.ModuleList(
                blocks
            )

            self.backend_name = "mamba"

        except Exception:
            self.input_proj = None

            backend = TemporalBlock(
                input_dim=config.input_dim,
                hidden_dim=config.hidden_dim,
                num_layers=config.num_layers,
                dropout=config.dropout,
            )

        self.backend = backend

        self.norm = nn.LayerNorm(
            config.hidden_dim
        )

        self.head = nn.Sequential(
            nn.Linear(
                config.hidden_dim,
                config.hidden_dim,
            ),
            nn.GELU(),
            nn.Dropout(
                config.dropout
            ),
            nn.Linear(
                config.hidden_dim,
                2,
            ),
            nn.Softplus(),
        )

    def forward(self, x):
        if self.backend_name == "mamba":
            y = self.input_proj(x)

            for block in self.mamba_blocks:
                y = y + block(y)

        else:
            y = self.backend(x)

        last = self.norm(
            y[:, -1, :]
        )

        return self.head(
            last
        )


def make_windows(
    features: np.ndarray,
    translation_error: np.ndarray,
    rotation_error_deg: np.ndarray,
    window: int,
    horizon: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Source window:
        [t-window+1, ..., t]

    Target:
        t + horizon

    This guarantees predictive alignment and avoids future leakage.
    """

    features = np.asarray(
        features,
        dtype=np.float64,
    )

    translation_error = np.asarray(
        translation_error,
        dtype=np.float64,
    ).reshape(-1)

    rotation_error_deg = np.asarray(
        rotation_error_deg,
        dtype=np.float64,
    ).reshape(-1)

    n = min(
        len(features),
        len(translation_error),
        len(rotation_error_deg),
    )

    X = []
    Y = []
    target_index = []

    start_t = window - 1
    end_t = n - horizon - 1

    for t in range(
        start_t,
        end_t + 1,
    ):
        target = t + horizon

        X.append(
            features[
                t - window + 1:
                t + 1
            ]
        )

        Y.append(
            [
                translation_error[target],
                rotation_error_deg[target],
            ]
        )

        target_index.append(
            target
        )

    return (
        np.asarray(
            X,
            dtype=np.float32,
        ),
        np.asarray(
            Y,
            dtype=np.float32,
        ),
        np.asarray(
            target_index,
            dtype=np.int64,
        ),
    )


def robust_standardize_fit(
    X: np.ndarray,
) -> Dict[str, np.ndarray]:
    flat = np.asarray(
        X,
        dtype=np.float64,
    ).reshape(
        -1,
        X.shape[-1],
    )

    median = np.median(
        flat,
        axis=0,
    )

    q25 = np.percentile(
        flat,
        25,
        axis=0,
    )

    q75 = np.percentile(
        flat,
        75,
        axis=0,
    )

    scale = q75 - q25

    scale = np.where(
        np.abs(
            scale
        ) < 1e-8,
        1.0,
        scale,
    )

    return {
        "median":
            median.astype(
                np.float32
            ),
        "scale":
            scale.astype(
                np.float32
            ),
    }


def robust_standardize_apply(
    X: np.ndarray,
    stats: Dict[str, np.ndarray],
) -> np.ndarray:
    return (
        (
            np.asarray(
                X,
                dtype=np.float32,
            )
            -
            stats[
                "median"
            ][
                None,
                None,
                :
            ]
        )
        /
        stats[
            "scale"
        ][
            None,
            None,
            :
        ]
    )


def error_to_reliability(
    translation_error_pred: np.ndarray,
    rotation_error_pred_deg: np.ndarray,
    trans_normal: float,
    trans_severe: float,
    rot_normal: float,
    rot_severe: float,
    floor: float = 0.01,
) -> np.ndarray:
    """
    Convert predicted factor errors into the same semantics as the GT oracle.

    Reliability is low if EITHER translation or rotation is predicted bad.
    """

    te = np.asarray(
        translation_error_pred,
        dtype=np.float64,
    )

    re = np.asarray(
        rotation_error_pred_deg,
        dtype=np.float64,
    )

    t_sev = np.clip(
        (
            te
            -
            float(
                trans_normal
            )
        )
        /
        max(
            float(
                trans_severe
                -
                trans_normal
            ),
            0.10,
        ),
        0.0,
        1.0,
    )

    r_sev = np.clip(
        (
            re
            -
            float(
                rot_normal
            )
        )
        /
        max(
            float(
                rot_severe
                -
                rot_normal
            ),
            0.50,
        ),
        0.0,
        1.0,
    )

    severity = np.maximum(
        t_sev,
        r_sev,
    )

    return np.clip(
        1.0 - severity,
        floor,
        1.0,
    )
