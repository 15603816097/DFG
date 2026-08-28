import numpy as np

from src.factor_graph.four_sensor_graph import (
    gps_geodetic_to_local,
)


def test_gps_first_frame_is_origin():
    lla = np.asarray(
        [
            [
                49.0,
                8.0,
                100.0,
            ],
            [
                49.00001,
                8.00001,
                101.0,
            ],
        ],
        dtype=np.float64,
    )

    local = gps_geodetic_to_local(
        lla
    )

    assert local.shape == (
        2,
        3,
    )

    assert np.allclose(
        local[
            0
        ],
        np.zeros(
            3
        ),
    )

    assert local[
        1,
        2
    ] == 1.0


def test_gps_motion_is_metric():
    lla = np.asarray(
        [
            [
                49.0,
                8.0,
                100.0,
            ],
            [
                49.00001,
                8.0,
                100.0,
            ],
        ],
        dtype=np.float64,
    )

    local = gps_geodetic_to_local(
        lla
    )

    distance = np.linalg.norm(
        local[
            1
        ]
        -
        local[
            0
        ]
    )

    assert (
        distance > 0.5
        and
        distance < 2.0
    )
