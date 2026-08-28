from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np


@dataclass(frozen=True)
class LidarEventTriggerConfig:
    """
    Event-triggered LiDAR reliability.

    Baseline:
        LiDAR covariance remains FIXED.

    Only when multiple abnormal cues agree do we activate a LiDAR
    covariance inflation event.

    Cues:
        1) learned predictive reliability
        2) ICP/odometry quality
        3) cross-sensor motion inconsistency
        4) temporal motion inconsistency

    The trigger is intentionally conservative because previous experiments
    showed that continuously dynamic LiDAR covariance harmed the final graph.
    """

    # predictive reliability thresholds
    predictive_bad_threshold: float = 0.35
    predictive_severe_threshold: float = 0.18

    # online LiDAR odometry quality thresholds
    quality_bad_threshold: float = 0.78
    quality_severe_threshold: float = 0.68

    # cross-sensor disagreement thresholds
    translation_disagreement_bad: float = 0.18
    translation_disagreement_severe: float = 0.30

    rotation_disagreement_bad_deg: float = 0.35
    rotation_disagreement_severe_deg: float = 0.60

    # temporal inconsistency thresholds
    temporal_translation_z: float = 2.8
    temporal_rotation_z: float = 2.8

    # trigger score
    trigger_score_bad: float = 2.0
    trigger_score_severe: float = 3.5

    # reliability output used by covariance mapping.
    # Healthy frames are effectively "fixed" and will not be mapped.
    event_reliability_bad: float = 0.35
    event_reliability_severe: float = 0.15

    # causal rolling window
    temporal_window: int = 25

    eps: float = 1e-8


def _as_1d(values) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    x = np.nan_to_num(
        x,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
    return x


def _rotation_angle_deg_from_transform(T: np.ndarray) -> float:
    R = np.asarray(T, dtype=np.float64)[:3, :3]
    value = (np.trace(R) - 1.0) * 0.5
    return float(
        np.degrees(
            np.arccos(
                np.clip(
                    value,
                    -1.0,
                    1.0,
                )
            )
        )
    )


def _relative_motion_stats(
    transforms: np.ndarray,
) -> Dict[str, np.ndarray]:
    transforms = np.asarray(
        transforms,
        dtype=np.float64,
    )

    n = len(transforms)

    translation = np.zeros(
        n,
        dtype=np.float64,
    )

    rotation_deg = np.zeros(
        n,
        dtype=np.float64,
    )

    for i in range(n):
        T = transforms[i]

        translation[i] = float(
            np.linalg.norm(
                T[:3, 3]
            )
        )

        rotation_deg[i] = (
            _rotation_angle_deg_from_transform(
                T
            )
        )

    return {
        "translation":
            translation,
        "rotation_deg":
            rotation_deg,
    }


def _causal_robust_zscore(
    values,
    window: int,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Causal robust z-score using rolling median/MAD.

    No future samples are used.
    """
    values = _as_1d(
        values
    )

    n = len(
        values
    )

    result = np.zeros(
        n,
        dtype=np.float64,
    )

    window = max(
        3,
        int(
            window
        ),
    )

    for i in range(n):
        start = max(
            0,
            i - window + 1,
        )

        segment = values[
            start:
            i + 1
        ]

        median = float(
            np.median(
                segment
            )
        )

        mad = float(
            np.median(
                np.abs(
                    segment
                    -
                    median
                )
            )
        )

        robust_sigma = (
            1.4826
            *
            mad
        )

        result[
            i
        ] = (
            abs(
                values[
                    i
                ]
                -
                median
            )
            /
            max(
                robust_sigma,
                eps,
            )
        )

    return result


def _align_pair_values_to_frames(
    pair_values,
    n_frames: int,
    first_value: float = 1.0,
) -> np.ndarray:
    pair_values = _as_1d(
        pair_values
    )

    result = np.full(
        int(
            n_frames
        ),
        float(
            first_value
        ),
        dtype=np.float64,
    )

    count = min(
        len(
            pair_values
        ),
        max(
            0,
            int(
                n_frames
            )
            -
            1
        ),
    )

    if count > 0:
        result[
            1:
            count + 1
        ] = pair_values[
            :count
        ]

    if count + 1 < n_frames:
        result[
            count + 1:
        ] = (
            pair_values[
                count - 1
            ]
            if count > 0
            else
            first_value
        )

    return result


def load_lidar_predictive_reliability(
    predictive_dir: Path,
    n_frames: int,
) -> np.ndarray:
    path = (
        Path(
            predictive_dir
        )
        /
        "lidar_predictive_prior_target_aligned.txt"
    )

    if not path.exists():
        raise FileNotFoundError(
            path
        )

    values = np.loadtxt(
        path,
        dtype=np.float64,
    ).reshape(
        -1
    )

    if len(
        values
    ) < n_frames:
        raise ValueError(
            f"{path}: {len(values)} values, "
            f"expected at least {n_frames}"
        )

    return np.clip(
        values[
            :n_frames
        ],
        0.0,
        1.0,
    )


def build_lidar_event_trigger(
    lidar_between,
    lidar_quality,
    camera_between,
    camera_valid,
    predictive_reliability,
    n_frames: int,
    config: Optional[LidarEventTriggerConfig] = None,
) -> Dict[str, np.ndarray]:
    """
    Build a target-frame-aligned LiDAR event trigger.

    Returns
    -------
    dict with:
        trigger_mask
        severe_mask
        event_score
        event_reliability
        predictive_reliability
        quality_frame
        translation_disagreement
        rotation_disagreement_deg
        temporal_translation_z
        temporal_rotation_z
    """
    cfg = (
        config
        or
        LidarEventTriggerConfig()
    )

    lidar_between = np.asarray(
        lidar_between,
        dtype=np.float64,
    )

    camera_between = np.asarray(
        camera_between,
        dtype=np.float64,
    )

    camera_valid = np.asarray(
        camera_valid,
        dtype=bool,
    ).reshape(
        -1
    )

    predictive_reliability = np.clip(
        _as_1d(
            predictive_reliability
        )[
            :n_frames
        ],
        0.0,
        1.0,
    )

    lidar_quality_frame = (
        _align_pair_values_to_frames(
            lidar_quality,
            n_frames,
            first_value=1.0,
        )
    )

    lidar_stats = _relative_motion_stats(
        lidar_between
    )

    camera_stats = _relative_motion_stats(
        camera_between
    )

    pair_count = min(
        len(
            lidar_between
        ),
        len(
            camera_between
        ),
        len(
            camera_valid
        ),
    )

    trans_disagreement_pair = np.zeros(
        pair_count,
        dtype=np.float64,
    )

    rot_disagreement_pair = np.zeros(
        pair_count,
        dtype=np.float64,
    )

    for i in range(
        pair_count
    ):
        if not bool(
            camera_valid[
                i
            ]
        ):
            continue

        trans_disagreement_pair[
            i
        ] = abs(
            lidar_stats[
                "translation"
            ][
                i
            ]
            -
            camera_stats[
                "translation"
            ][
                i
            ]
        )

        rot_disagreement_pair[
            i
        ] = abs(
            lidar_stats[
                "rotation_deg"
            ][
                i
            ]
            -
            camera_stats[
                "rotation_deg"
            ][
                i
            ]
        )

    trans_disagreement = (
        _align_pair_values_to_frames(
            trans_disagreement_pair,
            n_frames,
            first_value=0.0,
        )
    )

    rot_disagreement = (
        _align_pair_values_to_frames(
            rot_disagreement_pair,
            n_frames,
            first_value=0.0,
        )
    )

    lidar_translation_frame = (
        _align_pair_values_to_frames(
            lidar_stats[
                "translation"
            ],
            n_frames,
            first_value=0.0,
        )
    )

    lidar_rotation_frame = (
        _align_pair_values_to_frames(
            lidar_stats[
                "rotation_deg"
            ],
            n_frames,
            first_value=0.0,
        )
    )

    temporal_translation_z = (
        _causal_robust_zscore(
            lidar_translation_frame,
            window=
                cfg.temporal_window,
            eps=
                cfg.eps,
        )
    )

    temporal_rotation_z = (
        _causal_robust_zscore(
            lidar_rotation_frame,
            window=
                cfg.temporal_window,
            eps=
                cfg.eps,
        )
    )

    score = np.zeros(
        n_frames,
        dtype=np.float64,
    )

    # ----------------------------------------------------------
    # Predictive reliability contribution
    # ----------------------------------------------------------
    score += (
        predictive_reliability
        <
        cfg.predictive_bad_threshold
    ).astype(
        np.float64
    )

    score += (
        predictive_reliability
        <
        cfg.predictive_severe_threshold
    ).astype(
        np.float64
    )

    # ----------------------------------------------------------
    # LiDAR online quality contribution
    # ----------------------------------------------------------
    score += (
        lidar_quality_frame
        <
        cfg.quality_bad_threshold
    ).astype(
        np.float64
    )

    score += (
        lidar_quality_frame
        <
        cfg.quality_severe_threshold
    ).astype(
        np.float64
    )

    # ----------------------------------------------------------
    # Cross-sensor consistency contribution
    # ----------------------------------------------------------
    score += (
        trans_disagreement
        >
        cfg.translation_disagreement_bad
    ).astype(
        np.float64
    )

    score += (
        trans_disagreement
        >
        cfg.translation_disagreement_severe
    ).astype(
        np.float64
    )

    score += (
        rot_disagreement
        >
        cfg.rotation_disagreement_bad_deg
    ).astype(
        np.float64
    )

    score += (
        rot_disagreement
        >
        cfg.rotation_disagreement_severe_deg
    ).astype(
        np.float64
    )

    # ----------------------------------------------------------
    # Temporal consistency contribution
    # ----------------------------------------------------------
    score += (
        temporal_translation_z
        >
        cfg.temporal_translation_z
    ).astype(
        np.float64
    )

    score += (
        temporal_rotation_z
        >
        cfg.temporal_rotation_z
    ).astype(
        np.float64
    )

    trigger_mask = (
        score
        >=
        cfg.trigger_score_bad
    )

    severe_mask = (
        score
        >=
        cfg.trigger_score_severe
    )

    event_reliability = np.ones(
        n_frames,
        dtype=np.float64,
    )

    event_reliability[
        trigger_mask
    ] = (
        cfg.event_reliability_bad
    )

    event_reliability[
        severe_mask
    ] = (
        cfg.event_reliability_severe
    )

    return {
        "trigger_mask":
            trigger_mask.astype(
                np.uint8
            ),
        "severe_mask":
            severe_mask.astype(
                np.uint8
            ),
        "event_score":
            score,
        "event_reliability":
            event_reliability,
        "predictive_reliability":
            predictive_reliability,
        "quality_frame":
            lidar_quality_frame,
        "translation_disagreement":
            trans_disagreement,
        "rotation_disagreement_deg":
            rot_disagreement,
        "temporal_translation_z":
            temporal_translation_z,
        "temporal_rotation_z":
            temporal_rotation_z,
    }
