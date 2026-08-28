"""
Mamba GPS reliability predictor.

Uses the project's existing:
    MambaEncoder:    B,T,51 -> B,T,128
    ReliabilityHead: B,T,128 -> B,T,4

Only the GPS channel (index 1) is supervised in the first-stage
reliability experiment.

The other three channels are preserved so the architecture remains
compatible with later IMU/LiDAR/Camera reliability supervision.
"""

from __future__ import annotations

import torch
import torch.nn as nn


try:
    from src.model.mamba_encoder import MambaEncoder
    from src.model.reliability_head import ReliabilityHead
except ModuleNotFoundError:
    from src.models.mamba_encoder import MambaEncoder
    from src.models.reliability_head import ReliabilityHead


class MambaReliabilityModel(nn.Module):

    def __init__(
        self,
        input_dim: int = 51,
        hidden_dim: int = 128,
        state_dim: int = 16,
        num_layers: int = 2,
        head_hidden_dim: int = 64,
        sensor_num: int = 4,
        gps_channel: int = 1,
    ):
        super().__init__()

        self.gps_channel = int(
            gps_channel
        )

        self.encoder = MambaEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            state_dim=state_dim,
            num_layers=num_layers,
        )

        self.head = ReliabilityHead(
            input_dim=hidden_dim,
            hidden_dim=head_hidden_dim,
            sensor_num=sensor_num,
        )

    def forward(
        self,
        x,
    ):
        """
        Parameters
        ----------
        x:
            B,T,51

        Returns
        -------
        all_reliability:
            B,T,4

        gps_reliability:
            B,T
        """
        encoded = self.encoder(
            x
        )

        all_reliability = self.head(
            encoded
        )

        gps_reliability = (
            all_reliability[
                :,
                :,
                self.gps_channel
            ]
        )

        return {
            "all_reliability": all_reliability,
            "gps_reliability": gps_reliability,
            "encoded": encoded,
        }


class WeightedReliabilityMSE(nn.Module):
    """
    Continuous reliability regression loss.

    Low-reliability labels receive slightly larger weights so the
    network does not learn the trivial "always reliable" solution.
    """

    def __init__(
        self,
        low_reliability_weight: float = 2.0,
    ):
        super().__init__()

        self.low_reliability_weight = float(
            low_reliability_weight
        )

    def forward(
        self,
        prediction,
        target,
    ):
        target = torch.clamp(
            target,
            0.0,
            1.0,
        )

        weight = (
            1.0
            +
            self.low_reliability_weight
            *
            (
                1.0
                -
                target
            )
        )

        error = (
            prediction
            -
            target
        ) ** 2

        return torch.mean(
            weight
            *
            error
        )
