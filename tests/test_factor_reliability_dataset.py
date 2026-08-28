import numpy as np

from src.dataset.factor_reliability_dataset import (
    SENSORS,
    FactorReliabilityData,
)


def test_sensor_names():
    assert SENSORS == (
        "gps",
        "imu",
        "lidar",
        "camera",
    )


def test_future_alignment_definition():
    current = np.asarray(
        [
            1.0,
            0.9,
            0.8,
            0.7,
            0.6,
            0.5,
        ],
        dtype=np.float32,
    )

    horizon = 2

    future = np.empty_like(
        current
    )

    future[
        :-horizon
    ] = current[
        horizon:
    ]

    future[
        -horizon:
    ] = current[
        -1
    ]

    assert np.isclose(
        future[
            0
        ],
        current[
            2
        ],
    )

    assert np.isclose(
        future[
            3
        ],
        current[
            5
        ],
    )


def test_dataclass():
    data = FactorReliabilityData(
        features={},
        current_labels={},
        future_labels={},
        horizon=3,
        length=100,
    )

    assert data.horizon == 3

    assert data.length == 100
