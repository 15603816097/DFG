"""
Balanced Multi-Sensor Degradation Plan V2
=========================================

Goal
----
Avoid the old problem where train/validation/test contained very different
sensor-degradation distributions because degradation episodes were placed
only at a few absolute frame ranges.

V2 creates degradation episodes independently inside each chronological split:

    Train      0%  - 70%
    Validation 70% - 85%
    Test       85% - 100%

Each split contains, for every sensor:

    normal frames
    mild gradual degradation
    medium gradual degradation
    severe gradual degradation
    short burst degradation

The four sensors still have different schedules and different corruption modes.
The model therefore cannot simply memorize one global absolute timeline.

This module creates only severity + mode plans.
Raw KITTI files are never overwritten.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


SENSORS = (
    "gps",
    "imu",
    "lidar",
    "camera",
)


def _smoothstep(x):
    x = np.clip(
        np.asarray(
            x,
            dtype=np.float64,
        ),
        0.0,
        1.0,
    )

    return (
        x
        *
        x
        *
        (
            3.0
            -
            2.0
            *
            x
        )
    )


def _add_episode(
    curve,
    start,
    rise_length,
    hold_length,
    recovery_length,
    peak,
):
    n = len(
        curve
    )

    start = int(
        max(
            0,
            start,
        )
    )

    rise_end = min(
        start
        +
        int(
            rise_length
        ),
        n,
    )

    hold_end = min(
        rise_end
        +
        int(
            hold_length
        ),
        n,
    )

    recovery_end = min(
        hold_end
        +
        int(
            recovery_length
        ),
        n,
    )

    if rise_end > start:
        x = np.linspace(
            0.0,
            1.0,
            rise_end - start,
            endpoint=False,
        )

        curve[
            start:
            rise_end
        ] = np.maximum(
            curve[
                start:
                rise_end
            ],
            peak
            *
            _smoothstep(
                x
            ),
        )

    if hold_end > rise_end:
        curve[
            rise_end:
            hold_end
        ] = np.maximum(
            curve[
                rise_end:
                hold_end
            ],
            peak,
        )

    if recovery_end > hold_end:
        x = np.linspace(
            0.0,
            1.0,
            recovery_end - hold_end,
            endpoint=False,
        )

        curve[
            hold_end:
            recovery_end
        ] = np.maximum(
            curve[
                hold_end:
                recovery_end
            ],
            peak
            *
            (
                1.0
                -
                _smoothstep(
                    x
                )
            ),
        )


def _add_burst(
    curve,
    start,
    length,
    peak=1.0,
):
    n = len(
        curve
    )

    start = int(
        max(
            0,
            start,
        )
    )

    end = min(
        start
        +
        int(
            length
        ),
        n,
    )

    if end > start:
        curve[
            start:
            end
        ] = np.maximum(
            curve[
                start:
                end
            ],
            float(
                peak
            ),
        )


def _segment_bounds(
    n_frames,
):
    train_end = int(
        n_frames
        *
        0.70
    )

    val_end = int(
        n_frames
        *
        0.85
    )

    return {
        "train":
            (
                0,
                train_end,
            ),

        "val":
            (
                train_end,
                val_end,
            ),

        "test":
            (
                val_end,
                n_frames,
            ),
    }


def _relative_position(
    start,
    end,
    ratio,
):
    length = (
        end
        -
        start
    )

    return int(
        start
        +
        ratio
        *
        length
    )


def _build_sensor_curve_for_split(
    n_frames,
    split_start,
    split_end,
    sensor_index,
    split_index,
):
    """
    Build one sensor curve inside one split.

    All splits get the same severity families, but positions are shifted
    by sensor_index and split_index so the network cannot rely on one
    absolute degradation timing pattern.
    """
    curve = np.zeros(
        n_frames,
        dtype=np.float64,
    )

    split_length = (
        split_end
        -
        split_start
    )

    if split_length <= 40:
        return curve

    sensor_shift = (
        0.025
        *
        sensor_index
    )

    split_shift = (
        0.012
        *
        split_index
    )

    mild_start = (
        0.08
        +
        sensor_shift
        +
        split_shift
    )

    medium_start = (
        0.34
        +
        0.015
        *
        sensor_index
        -
        split_shift
    )

    severe_start = (
        0.61
        -
        0.010
        *
        sensor_index
        +
        split_shift
    )

    burst_start = (
        0.88
        -
        0.015
        *
        sensor_index
    )

    def frames(
        ratio,
    ):
        return _relative_position(
            split_start,
            split_end,
            np.clip(
                ratio,
                0.0,
                0.97,
            ),
        )

    # Episode lengths scale with split size.
    rise = max(
        10,
        int(
            split_length
            *
            0.055
        ),
    )

    hold = max(
        12,
        int(
            split_length
            *
            0.080
        ),
    )

    recovery = max(
        10,
        int(
            split_length
            *
            0.050
        ),
    )

    _add_episode(
        curve,
        frames(
            mild_start
        ),
        rise,
        hold,
        recovery,
        peak=0.35,
    )

    _add_episode(
        curve,
        frames(
            medium_start
        ),
        rise,
        hold,
        recovery,
        peak=0.65,
    )

    _add_episode(
        curve,
        frames(
            severe_start
        ),
        rise,
        hold,
        recovery,
        peak=1.00,
    )

    _add_burst(
        curve,
        frames(
            burst_start
        ),
        max(
            8,
            int(
                split_length
                *
                0.035
            ),
        ),
        peak=1.00,
    )

    # Keep split boundaries clean:
    # no episode is allowed to leak into another split.
    result = np.zeros_like(
        curve
    )

    result[
        split_start:
        split_end
    ] = curve[
        split_start:
        split_end
    ]

    return result


def create_balanced_multisensor_degradation_plan(
    n_frames,
    seed=20260826,
):
    n = int(
        n_frames
    )

    if n <= 0:
        raise ValueError(
            "n_frames must be positive"
        )

    bounds = _segment_bounds(
        n
    )

    severity = {
        sensor:
            np.zeros(
                n,
                dtype=np.float64,
            )
        for sensor in SENSORS
    }

    for split_index, split_name in enumerate(
        (
            "train",
            "val",
            "test",
        )
    ):
        split_start, split_end = (
            bounds[
                split_name
            ]
        )

        for sensor_index, sensor in enumerate(
            SENSORS
        ):
            local = (
                _build_sensor_curve_for_split(
                    n,
                    split_start,
                    split_end,
                    sensor_index,
                    split_index,
                )
            )

            severity[
                sensor
            ] = np.maximum(
                severity[
                    sensor
                ],
                local,
            )

    # Mode encoding remains sensor-specific.
    mode = {}

    # GPS:
    # 0 normal
    # 1 gradual noise/bias
    # 2 outlier burst
    gps_mode = (
        severity[
            "gps"
        ]
        >
        0
    ).astype(
        np.int64
    )

    # Last high-severity portion of every split becomes burst mode.
    for split_name in (
        "train",
        "val",
        "test",
    ):
        a, b = bounds[
            split_name
        ]

        start = _relative_position(
            a,
            b,
            0.88,
        )

        gps_mode[
            start:b
        ][
            severity[
                "gps"
            ][
                start:b
            ]
            >
            0.9
        ] = 2

    mode[
        "gps"
    ] = gps_mode

    # IMU:
    # 0 normal
    # 1 bias + noise
    # 2 spike
    imu_mode = (
        severity[
            "imu"
        ]
        >
        0
    ).astype(
        np.int64
    )

    for split_name in (
        "train",
        "val",
        "test",
    ):
        a, b = bounds[
            split_name
        ]

        start = _relative_position(
            a,
            b,
            0.865,
        )

        imu_mode[
            start:b
        ][
            severity[
                "imu"
            ][
                start:b
            ]
            >
            0.9
        ] = 2

    mode[
        "imu"
    ] = imu_mode

    # LiDAR:
    # 0 normal
    # 1 dropout + range noise
    # 2 partial occlusion
    lidar_mode = (
        severity[
            "lidar"
        ]
        >
        0
    ).astype(
        np.int64
    )

    # Severe regions are partial occlusion.
    lidar_mode[
        severity[
            "lidar"
        ]
        >=
        0.85
    ] = 2

    mode[
        "lidar"
    ] = lidar_mode

    # Camera:
    # 0 normal
    # 1 dark + blur
    # 2 overexposure + noise
    # 3 occlusion + noise
    camera_mode = np.zeros(
        n,
        dtype=np.int64,
    )

    active = (
        severity[
            "camera"
        ]
        >
        0
    )

    camera_mode[
        active
        &
        (
            severity[
                "camera"
            ]
            <
            0.50
        )
    ] = 1

    camera_mode[
        active
        &
        (
            severity[
                "camera"
            ]
            >=
            0.50
        )
        &
        (
            severity[
                "camera"
            ]
            <
            0.85
        )
    ] = 2

    camera_mode[
        severity[
            "camera"
        ]
        >=
        0.85
    ] = 3

    mode[
        "camera"
    ] = camera_mode

    return {
        "seed":
            int(
                seed
            ),

        "n_frames":
            n,

        "split_bounds":
            bounds,

        "severity":
            {
                sensor:
                    severity[
                        sensor
                    ].astype(
                        np.float32
                    )
                for sensor in SENSORS
            },

        "mode":
            {
                sensor:
                    mode[
                        sensor
                    ].astype(
                        np.int64
                    )
                for sensor in SENSORS
            },
    }


def save_balanced_plan(
    plan,
    output_dir,
):
    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    arrays = {}

    for sensor in SENSORS:
        arrays[
            f"{sensor}_severity"
        ] = plan[
            "severity"
        ][
            sensor
        ]

        arrays[
            f"{sensor}_mode"
        ] = plan[
            "mode"
        ][
            sensor
        ]

    np.savez(
        output_dir
        /
        "degradation_plan_v2.npz",
        **arrays,
    )

    metadata = {
        "version":
            "balanced_v2",

        "seed":
            int(
                plan[
                    "seed"
                ]
            ),

        "n_frames":
            int(
                plan[
                    "n_frames"
                ]
            ),

        "split_bounds":
            {
                name:
                    [
                        int(
                            value[0]
                        ),
                        int(
                            value[1]
                        ),
                    ]
                for name, value in plan[
                    "split_bounds"
                ].items()
            },

        "design":
            (
                "Each chronological split contains "
                "normal + mild + medium + severe + burst "
                "degradation for every sensor."
            ),
    }

    with open(
        output_dir
        /
        "degradation_plan_v2.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False,
        )


def load_balanced_plan(
    plan_dir,
):
    plan_dir = Path(
        plan_dir
    )

    data = np.load(
        plan_dir
        /
        "degradation_plan_v2.npz"
    )

    severity = {
        sensor:
            np.asarray(
                data[
                    f"{sensor}_severity"
                ],
                dtype=np.float32,
            )
        for sensor in SENSORS
    }

    mode = {
        sensor:
            np.asarray(
                data[
                    f"{sensor}_mode"
                ],
                dtype=np.int64,
            )
        for sensor in SENSORS
    }

    n = len(
        severity[
            "gps"
        ]
    )

    train_end = int(
        n
        *
        0.70
    )

    val_end = int(
        n
        *
        0.85
    )

    return {
        "seed":
            20260826,

        "n_frames":
            n,

        "split_bounds":
            {
                "train":
                    (
                        0,
                        train_end,
                    ),

                "val":
                    (
                        train_end,
                        val_end,
                    ),

                "test":
                    (
                        val_end,
                        n,
                    ),
            },

        "severity":
            severity,

        "mode":
            mode,
    }
