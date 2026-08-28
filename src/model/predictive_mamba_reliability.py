"""
Predictive Mamba Reliability Model V2
=====================================

Multi-task predictive sensor reliability learning.

Shared temporal encoder:
    MambaEncoder
        B,T,51 -> B,T,128

Two prediction heads:
    1. Continuous GPS reliability regression
        B,T,128 -> B,T

    2. Future GPS state classification
        B,T,128 -> B,T,3

State definition:
    class 0 = Unreliable
              reliability < 0.20

    class 1 = Degrading
              0.20 <= reliability < 0.80

    class 2 = Reliable
              reliability >= 0.80

Why multi-task?
---------------
Pure regression can become over-conservative and predict low
reliability for almost every frame.  The classification head forces
the temporal encoder to distinguish reliable, degrading, and
unreliable regimes while the regression head keeps a continuous
reliability value for dynamic covariance mapping.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


try:
    from src.model.mamba_encoder import MambaEncoder
except ModuleNotFoundError:
    from src.models.mamba_encoder import MambaEncoder


class PredictiveMambaReliabilityModel(
    nn.Module
):
    """
    Multi-task future reliability predictor.

    Input
    -----
    x:
        B,T,51

    Output
    ------
    regression:
        B,T
        continuous reliability in [0,1]

    state_logits:
        B,T,3

    state_probability:
        B,T,3

    fused_reliability:
        B,T

        0.5 * regression
        +
        0.5 * expected reliability from state probabilities
    """

    def __init__(
        self,
        input_dim: int = 51,
        hidden_dim: int = 128,
        state_dim: int = 16,
        num_layers: int = 2,
        head_hidden_dim: int = 64,
        dropout: float = 0.10,
    ):

        super().__init__()


        self.encoder = MambaEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            state_dim=state_dim,
            num_layers=num_layers,
        )


        self.regression_head = nn.Sequential(

            nn.Linear(
                hidden_dim,
                head_hidden_dim,
            ),

            nn.ReLU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                head_hidden_dim,
                1,
            ),

            nn.Sigmoid(),

        )


        self.state_head = nn.Sequential(

            nn.Linear(
                hidden_dim,
                head_hidden_dim,
            ),

            nn.ReLU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                head_hidden_dim,
                3,
            ),

        )


        # Expected reliability assigned to each state.
        #
        # bad        -> 0.05
        # degrading  -> 0.50
        # reliable   -> 0.95

        self.register_buffer(

            "state_reliability_values",

            torch.tensor(
                [
                    0.05,
                    0.50,
                    0.95,
                ],
                dtype=torch.float32,
            ),

        )


    def forward(
        self,
        x,
    ):

        encoded = self.encoder(
            x
        )


        regression = (
            self.regression_head(
                encoded
            )
            .squeeze(
                -1
            )
        )


        state_logits = (
            self.state_head(
                encoded
            )
        )


        state_probability = (
            torch.softmax(
                state_logits,
                dim=-1,
            )
        )


        state_expected_reliability = (
            state_probability
            *
            self.state_reliability_values
            .view(
                1,
                1,
                3,
            )
        ).sum(
            dim=-1
        )


        fused_reliability = (
            0.50
            *
            regression
            +
            0.50
            *
            state_expected_reliability
        )


        return {

            "encoded":
                encoded,


            "regression":
                regression,


            "state_logits":
                state_logits,


            "state_probability":
                state_probability,


            "state_expected_reliability":
                state_expected_reliability,


            "fused_reliability":
                fused_reliability,

        }


def reliability_to_state(
    reliability,
    bad_threshold: float = 0.20,
    reliable_threshold: float = 0.80,
):
    """
    Convert continuous reliability labels to 3 classes.

    0 = unreliable
    1 = degrading
    2 = reliable
    """

    state = torch.ones_like(
        reliability,
        dtype=torch.long,
    )


    state[
        reliability
        <
        bad_threshold
    ] = 0


    state[
        reliability
        >=
        reliable_threshold
    ] = 2


    return state


class MultiTaskPredictiveReliabilityLoss(
    nn.Module
):
    """
    Balanced multi-task objective.

    L =
        classification_weight * CE
        +
        regression_weight * SmoothL1
        +
        safety_weight * mild asymmetric penalty

    The asymmetric term is intentionally mild.  It penalizes only
    severe over-confidence on truly unreliable samples and avoids the
    collapse observed with the previous aggressive safety loss.
    """

    def __init__(
        self,
        classification_weight: float = 1.0,
        regression_weight: float = 0.75,
        safety_weight: float = 0.25,
        class_weights=None,
        bad_threshold: float = 0.20,
        reliable_threshold: float = 0.80,
    ):

        super().__init__()


        self.classification_weight = float(
            classification_weight
        )


        self.regression_weight = float(
            regression_weight
        )


        self.safety_weight = float(
            safety_weight
        )


        self.bad_threshold = float(
            bad_threshold
        )


        self.reliable_threshold = float(
            reliable_threshold
        )


        if class_weights is None:

            class_weights = torch.tensor(
                [
                    1.25,
                    1.50,
                    1.00,
                ],
                dtype=torch.float32,
            )


        else:

            class_weights = torch.as_tensor(
                class_weights,
                dtype=torch.float32,
            )


        self.register_buffer(
            "class_weights",
            class_weights,
        )


    def forward(
        self,
        output,
        target_reliability,
    ):

        target_reliability = torch.clamp(
            target_reliability,
            0.0,
            1.0,
        )


        target_state = reliability_to_state(

            target_reliability,

            bad_threshold=
                self.bad_threshold,

            reliable_threshold=
                self.reliable_threshold,

        )


        logits = output[
            "state_logits"
        ]


        regression = output[
            "regression"
        ]


        classification_loss = (
            F.cross_entropy(

                logits.reshape(
                    -1,
                    3,
                ),

                target_state.reshape(
                    -1
                ),

                weight=
                    self.class_weights,

            )
        )


        regression_loss = (
            F.smooth_l1_loss(

                regression,

                target_reliability,

                beta=0.10,

            )
        )


        # Mild safety penalty:
        #
        # only unreliable ground-truth samples are considered,
        # and only predicted reliability above 0.50 is penalized.

        bad_mask = (
            target_state
            ==
            0
        )


        if torch.any(
            bad_mask
        ):

            severe_overestimate = torch.relu(

                regression[
                    bad_mask
                ]
                -
                0.50

            )


            safety_loss = torch.mean(

                severe_overestimate
                .pow(
                    2
                )

            )


        else:

            safety_loss = regression.new_tensor(
                0.0
            )


        total_loss = (

            self.classification_weight
            *
            classification_loss

            +

            self.regression_weight
            *
            regression_loss

            +

            self.safety_weight
            *
            safety_loss

        )


        return {

            "total":
                total_loss,


            "classification":
                classification_loss,


            "regression":
                regression_loss,


            "safety":
                safety_loss,

        }
