import numpy as np

from src.reliability.lidar_event_triggered_reliability import (
    LidarEventTriggerConfig,
    build_lidar_event_trigger,
)


def make_identity_transforms(n):
    result = np.repeat(
        np.eye(
            4,
            dtype=np.float64,
        )[
            None,
            :,
            :
        ],
        n,
        axis=0,
    )

    return result


def test_healthy_sequence_does_not_trigger():
    n_frames = 20
    n_pairs = n_frames - 1

    lidar = make_identity_transforms(
        n_pairs
    )

    camera = make_identity_transforms(
        n_pairs
    )

    quality = np.full(
        n_pairs,
        0.95,
        dtype=np.float64,
    )

    camera_valid = np.ones(
        n_pairs,
        dtype=bool,
    )

    prediction = np.full(
        n_frames,
        0.90,
        dtype=np.float64,
    )

    result = build_lidar_event_trigger(
        lidar_between=
            lidar,
        lidar_quality=
            quality,
        camera_between=
            camera,
        camera_valid=
            camera_valid,
        predictive_reliability=
            prediction,
        n_frames=
            n_frames,
    )

    assert int(
        result[
            "trigger_mask"
        ].sum()
    ) == 0


def test_multiple_bad_cues_trigger():
    n_frames = 20
    n_pairs = n_frames - 1

    lidar = make_identity_transforms(
        n_pairs
    )

    camera = make_identity_transforms(
        n_pairs
    )

    lidar[
        10,
        0,
        3
    ] = 1.0

    quality = np.full(
        n_pairs,
        0.95,
        dtype=np.float64,
    )

    quality[
        10
    ] = 0.50

    camera_valid = np.ones(
        n_pairs,
        dtype=bool,
    )

    prediction = np.full(
        n_frames,
        0.90,
        dtype=np.float64,
    )

    prediction[
        11
    ] = 0.10

    result = build_lidar_event_trigger(
        lidar_between=
            lidar,
        lidar_quality=
            quality,
        camera_between=
            camera,
        camera_valid=
            camera_valid,
        predictive_reliability=
            prediction,
        n_frames=
            n_frames,
    )

    assert bool(
        result[
            "trigger_mask"
        ][
            11
        ]
    )


def test_event_reliability_is_bounded():
    n_frames = 12
    n_pairs = n_frames - 1

    lidar = make_identity_transforms(
        n_pairs
    )

    camera = make_identity_transforms(
        n_pairs
    )

    quality = np.full(
        n_pairs,
        0.5,
        dtype=np.float64,
    )

    camera_valid = np.ones(
        n_pairs,
        dtype=bool,
    )

    prediction = np.full(
        n_frames,
        0.1,
        dtype=np.float64,
    )

    result = build_lidar_event_trigger(
        lidar_between=
            lidar,
        lidar_quality=
            quality,
        camera_between=
            camera,
        camera_valid=
            camera_valid,
        predictive_reliability=
            prediction,
        n_frames=
            n_frames,
    )

    reliability = result[
        "event_reliability"
    ]

    assert np.all(
        reliability
        >=
        0.0
    )

    assert np.all(
        reliability
        <=
        1.0
    )
