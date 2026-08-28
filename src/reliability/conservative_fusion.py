from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np


SENSORS = ("gps", "imu", "lidar", "camera")


@dataclass(frozen=True)
class ConservativeFusionConfig:
    """
    Conservative sensor-specific fusion.

    GPS
        Learned future factor reliability is already useful -> prediction dominant.

    IMU
        Prediction is weak -> shrink toward a neutral prior and apply a floor.

    LiDAR
        Online odometry quality is much more trustworthy than the learned branch.
        Prediction only makes a small correction.

    Camera
        Online VO quality is dominant.
        The weak prediction branch is NEVER allowed to increase confidence.
    """
    gps_floor: float = 0.05

    imu_prediction_weight: float = 0.65
    imu_neutral_prior: float = 0.80
    imu_floor: float = 0.30

    lidar_quality_weight: float = 0.85
    lidar_prediction_weight: float = 0.15
    lidar_floor: float = 0.08
    lidar_max_upward_delta: float = 0.05

    camera_quality_weight: float = 0.90
    camera_prediction_weight: float = 0.10
    camera_floor: float = 0.03
    camera_max_upward_delta: float = 0.00


def _clip01(values) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = np.nan_to_num(
        values,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
    return np.clip(values, 0.0, 1.0)


def align_pair_quality_to_frames(
    pair_quality,
    n_frames: int,
) -> np.ndarray:
    """
    Relative odometry quality has N-1 values for factors i -> i+1.

    Predictive reliability files are target-frame aligned with N values.
    Therefore:
        frame 0 quality = 1.0
        frame j quality = pair_quality[j-1], j >= 1
    """
    q = _clip01(pair_quality)

    result = np.ones(
        int(n_frames),
        dtype=np.float64,
    )

    count = min(
        len(q),
        max(
            0,
            int(n_frames) - 1,
        ),
    )

    if count > 0:
        result[
            1:
            count + 1
        ] = q[
            :count
        ]

    if count + 1 < n_frames:
        fill = q[
            count - 1
        ] if count > 0 else 1.0

        result[
            count + 1:
        ] = fill

    return result


def load_factor_quality_npz(
    degraded_measurement_dir: Path,
    sensor: str,
    n_frames: int,
) -> np.ndarray:
    """
    Load quality from the actual degraded relative-factor files created by the
    already-working degradation pipeline.

    Expected:
        degraded_lidar_factor_data.npz
        degraded_camera_factor_data.npz

    with key:
        quality
    """
    degraded_measurement_dir = Path(
        degraded_measurement_dir
    )

    path = (
        degraded_measurement_dir
        /
        f"degraded_{sensor}_factor_data.npz"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing degraded {sensor} factor file: {path}"
        )

    data = np.load(
        path,
        allow_pickle=False,
    )

    if "quality" not in data.files:
        raise KeyError(
            f"{path} has no 'quality' key. "
            f"Available: {data.files}"
        )

    return align_pair_quality_to_frames(
        data[
            "quality"
        ],
        n_frames,
    )


def fuse_sensor_reliability(
    sensor: str,
    predicted,
    quality: Optional[np.ndarray] = None,
    config: ConservativeFusionConfig = ConservativeFusionConfig(),
) -> np.ndarray:
    sensor = str(
        sensor
    ).lower()

    if sensor not in SENSORS:
        raise ValueError(
            f"Unsupported sensor: {sensor}"
        )

    p = _clip01(
        predicted
    )

    if sensor == "gps":
        return np.clip(
            p,
            config.gps_floor,
            1.0,
        )

    if sensor == "imu":
        result = (
            config.imu_prediction_weight
            *
            p
            +
            (
                1.0
                -
                config.imu_prediction_weight
            )
            *
            config.imu_neutral_prior
        )

        return np.clip(
            result,
            config.imu_floor,
            1.0,
        )

    if quality is None:
        raise ValueError(
            f"{sensor} conservative fusion requires "
            f"real online factor quality."
        )

    q = _clip01(
        quality
    )

    if len(
        q
    ) != len(
        p
    ):
        raise ValueError(
            f"{sensor} quality length {len(q)} "
            f"!= prediction length {len(p)}"
        )

    if sensor == "lidar":
        result = (
            config.lidar_quality_weight
            *
            q
            +
            config.lidar_prediction_weight
            *
            p
        )

        # The weak learned LiDAR branch may improve confidence only slightly.
        upper = np.clip(
            q
            +
            config.lidar_max_upward_delta,
            0.0,
            1.0,
        )

        result = np.minimum(
            result,
            upper,
        )

        return np.clip(
            result,
            config.lidar_floor,
            1.0,
        )

    # Camera:
    # weak / near-zero-correlation predictor is allowed to decrease trust,
    # but NEVER make a visual factor stronger than measured VO quality.
    result = (
        config.camera_quality_weight
        *
        q
        +
        config.camera_prediction_weight
        *
        p
    )

    upper = np.clip(
        q
        +
        config.camera_max_upward_delta,
        0.0,
        1.0,
    )

    result = np.minimum(
        result,
        upper,
    )

    return np.clip(
        result,
        config.camera_floor,
        1.0,
    )


def build_conservative_target_aligned_files(
    source_prediction_dir: Path,
    degraded_measurement_dir: Path,
    output_dir: Path,
    config: ConservativeFusionConfig = ConservativeFusionConfig(),
) -> Dict[str, Dict[str, float]]:
    """
    IMPORTANT FIX relative to the previous version:

    Only transform the FOUR files that the factor graph actually reads:

        gps_predictive_prior_target_aligned.txt
        imu_predictive_prior_target_aligned.txt
        lidar_predictive_prior_target_aligned.txt
        camera_predictive_prior_target_aligned.txt

    Do NOT touch:
        *_prediction_source_frame.txt
        *_factor_current_prediction.txt
        *_factor_future_source_aligned.txt

    Source-frame files contain integer frame IDs, NOT reliability values.
    """
    source_prediction_dir = Path(
        source_prediction_dir
    )

    degraded_measurement_dir = Path(
        degraded_measurement_dir
    )

    output_dir = Path(
        output_dir
    )

    if not source_prediction_dir.exists():
        raise FileNotFoundError(
            source_prediction_dir
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions = {}

    for sensor in SENSORS:
        path = (
            source_prediction_dir
            /
            f"{sensor}_predictive_prior_target_aligned.txt"
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Missing V1 target-aligned reliability: {path}"
            )

        predictions[
            sensor
        ] = _clip01(
            np.loadtxt(
                path,
                dtype=np.float64,
            )
        )

    n = min(
        len(
            predictions[
                sensor
            ]
        )
        for sensor in SENSORS
    )

    for sensor in SENSORS:
        predictions[
            sensor
        ] = predictions[
            sensor
        ][
            :n
        ]

    lidar_quality = load_factor_quality_npz(
        degraded_measurement_dir,
        "lidar",
        n,
    )

    camera_quality = load_factor_quality_npz(
        degraded_measurement_dir,
        "camera",
        n,
    )

    qualities = {
        "gps":
            None,
        "imu":
            None,
        "lidar":
            lidar_quality,
        "camera":
            camera_quality,
    }

    summary = {}

    for sensor in SENSORS:
        fused = fuse_sensor_reliability(
            sensor,
            predictions[
                sensor
            ],
            qualities[
                sensor
            ],
            config=config,
        )

        output_path = (
            output_dir
            /
            f"{sensor}_predictive_prior_target_aligned.txt"
        )

        np.savetxt(
            output_path,
            fused,
            fmt="%.10f",
        )

        summary[
            sensor
        ] = {
            "pred_min":
                float(
                    np.min(
                        predictions[
                            sensor
                        ]
                    )
                ),
            "pred_max":
                float(
                    np.max(
                        predictions[
                            sensor
                        ]
                    )
                ),
            "pred_mean":
                float(
                    np.mean(
                        predictions[
                            sensor
                        ]
                    )
                ),
            "final_min":
                float(
                    np.min(
                        fused
                    )
                ),
            "final_max":
                float(
                    np.max(
                        fused
                    )
                ),
            "final_mean":
                float(
                    np.mean(
                        fused
                    )
                ),
        }

        if qualities[
            sensor
        ] is not None:
            summary[
                sensor
            ][
                "quality_min"
            ] = float(
                np.min(
                    qualities[
                        sensor
                    ]
                )
            )

            summary[
                sensor
            ][
                "quality_max"
            ] = float(
                np.max(
                    qualities[
                        sensor
                    ]
                )
            )

            summary[
                sensor
            ][
                "quality_mean"
            ] = float(
                np.mean(
                    qualities[
                        sensor
                    ]
                )
            )

    return summary
