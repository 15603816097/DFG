import numpy as np
import torch

from src.reliability.lidar_factor_error_predictor import (
    LidarFactorErrorModelConfig,
    LidarFactorErrorPredictor,
    error_to_reliability,
    make_windows,
)


def test_window_alignment():
    features = np.arange(
        100,
        dtype=np.float32,
    ).reshape(
        50,
        2,
    )

    te = np.arange(
        50,
        dtype=np.float32,
    )

    re = te * 2.0

    X, Y, index = make_windows(
        features,
        te,
        re,
        window=5,
        horizon=3,
    )

    assert X.shape[
        1
    ] == 5

    assert index[
        0
    ] == 7

    assert Y[
        0,
        0
    ] == te[
        7
    ]

    assert Y[
        0,
        1
    ] == re[
        7
    ]


def test_model_output_shape():
    config = LidarFactorErrorModelConfig(
        input_dim=8,
        hidden_dim=16,
        num_layers=1,
        window=10,
        horizon=3,
    )

    model = LidarFactorErrorPredictor(
        config
    )

    X = torch.randn(
        4,
        10,
        8,
    )

    Y = model(
        X
    )

    assert Y.shape == (
        4,
        2,
    )

    assert torch.all(
        Y >= 0
    )


def test_error_to_reliability_decreases():
    reliability = error_to_reliability(
        np.array(
            [
                0.02,
                0.50,
            ]
        ),
        np.array(
            [
                0.10,
                3.00,
            ]
        ),
        trans_normal=0.05,
        trans_severe=0.50,
        rot_normal=0.20,
        rot_severe=3.00,
    )

    assert reliability[
        0
    ] > reliability[
        1
    ]

    assert np.all(
        (
            reliability
            >=
            0
        )
        &
        (
            reliability
            <=
            1
        )
    )
