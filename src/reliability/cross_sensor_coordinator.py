from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional

import numpy as np


SENSORS = ("gps", "imu", "lidar", "camera")


@dataclass(frozen=True)
class CoordinatorConfig:
    """
    Conservative V1 cross-sensor coordinator.

    Design goals
    ------------
    1. Do not retrain or overwrite the existing V1 predictor.
    2. Keep each sensor's own predicted reliability.
    3. Coordinate the four reliabilities before covariance mapping.
    4. Protect the strong LiDAR geometric constraint from aggressive
       down-weighting, because the current sensor-wise ablation shows
       that the LiDAR predictive branch can damage the final graph.
    5. Preserve the empirically strong GPS+IMU+Camera interaction.

    All values are intentionally explicit so later sensitivity
    experiments can change them without changing the algorithm.
    """

    # How much a sensor may follow the cross-sensor consensus.
    consensus_gain_gps: float = 0.18
    consensus_gain_imu: float = 0.22
    consensus_gain_lidar: float = 0.08
    consensus_gain_camera: float = 0.18

    # Reliability lower bounds.  LiDAR is deliberately protected.
    floor_gps: float = 0.05
    floor_imu: float = 0.25
    floor_lidar: float = 0.72
    floor_camera: float = 0.08

    # Maximum deviation from the original V1 prediction.
    max_delta_gps: float = 0.12
    max_delta_imu: float = 0.10
    max_delta_lidar: float = 0.08
    max_delta_camera: float = 0.12

    # LiDAR gate: when LiDAR prediction is low but the other sensors
    # do not simultaneously indicate a bad period, avoid collapsing
    # LiDAR trust.
    lidar_disagreement_threshold: float = 0.20
    lidar_recovery_gain: float = 0.65

    # Camera may cooperate with GPS/IMU, which was the strongest
    # learned subset in the completed ablation.
    camera_gps_imu_gain: float = 0.12

    # Temporal smoothing.  This is causal: only past/current values
    # are used, so it does not leak future information.
    ema_beta: float = 0.82


def _as_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size == 0:
        raise ValueError("Reliability array must not be empty.")
    if not np.all(np.isfinite(x)):
        raise ValueError("Reliability array contains NaN/Inf.")
    return np.clip(x, 0.0, 1.0)


def _causal_ema(x: np.ndarray, beta: float) -> np.ndarray:
    x = _as_1d(x)
    beta = float(np.clip(beta, 0.0, 0.999999))
    y = np.empty_like(x)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = beta * y[i - 1] + (1.0 - beta) * x[i]
    return y


def _weighted_consensus(
    reliabilities: Mapping[str, np.ndarray],
) -> np.ndarray:
    """
    Robust cross-sensor consensus.

    GPS/IMU/Camera receive slightly larger influence than LiDAR in
    the consensus calculation.  This does NOT mean LiDAR is weak;
    LiDAR is protected separately because the graph already obtains
    strong geometric information from it.
    """
    weights = {
        "gps": 1.00,
        "imu": 0.95,
        "lidar": 0.70,
        "camera": 0.95,
    }
    stack = np.stack([reliabilities[s] for s in SENSORS], axis=1)
    w = np.asarray([weights[s] for s in SENSORS], dtype=np.float64)
    return np.sum(stack * w[None, :], axis=1) / np.sum(w)


def _bounded_update(
    original: np.ndarray,
    proposed: np.ndarray,
    max_delta: float,
    floor: float,
) -> np.ndarray:
    lo = original - max_delta
    hi = original + max_delta
    out = np.clip(proposed, lo, hi)
    return np.clip(out, floor, 1.0)


def coordinate_reliabilities(
    reliabilities: Mapping[str, np.ndarray],
    config: Optional[CoordinatorConfig] = None,
) -> Dict[str, np.ndarray]:
    """
    Coordinate four target-aligned reliability sequences.

    Parameters
    ----------
    reliabilities:
        Dict containing gps/imu/lidar/camera arrays of identical length.
    config:
        Optional CoordinatorConfig.

    Returns
    -------
    dict
        Coordinated arrays, still in [0, 1], with exactly the same length.
    """
    cfg = config or CoordinatorConfig()

    r = {s: _as_1d(reliabilities[s]) for s in SENSORS}
    n = len(r["gps"])
    for s in SENSORS:
        if len(r[s]) != n:
            raise ValueError(
                f"All reliability arrays must have equal length; "
                f"gps={n}, {s}={len(r[s])}"
            )

    smooth = {s: _causal_ema(r[s], cfg.ema_beta) for s in SENSORS}
    consensus = _weighted_consensus(smooth)

    gains = {
        "gps": cfg.consensus_gain_gps,
        "imu": cfg.consensus_gain_imu,
        "lidar": cfg.consensus_gain_lidar,
        "camera": cfg.consensus_gain_camera,
    }
    floors = {
        "gps": cfg.floor_gps,
        "imu": cfg.floor_imu,
        "lidar": cfg.floor_lidar,
        "camera": cfg.floor_camera,
    }
    deltas = {
        "gps": cfg.max_delta_gps,
        "imu": cfg.max_delta_imu,
        "lidar": cfg.max_delta_lidar,
        "camera": cfg.max_delta_camera,
    }

    proposed = {}
    for s in SENSORS:
        # Blend original instantaneous prediction with causal consensus.
        proposed[s] = (
            (1.0 - gains[s]) * r[s]
            + gains[s] * consensus
        )

    # GPS+IMU+Camera cooperative term.
    # If GPS and IMU jointly indicate degradation, camera is allowed
    # to become a little more conservative as well.
    gi_support = 0.5 * (smooth["gps"] + smooth["imu"])
    camera_penalty = np.maximum(
        0.0,
        smooth["camera"] - gi_support,
    )
    proposed["camera"] -= (
        cfg.camera_gps_imu_gain * camera_penalty
    )

    # LiDAR protection:
    # If LiDAR alone predicts low reliability while the other three
    # remain considerably healthier, recover part of LiDAR trust.
    others = (
        smooth["gps"] + smooth["imu"] + smooth["camera"]
    ) / 3.0
    disagreement = others - smooth["lidar"]
    recovery_mask = disagreement > cfg.lidar_disagreement_threshold
    recovery = np.zeros(n, dtype=np.float64)
    recovery[recovery_mask] = (
        cfg.lidar_recovery_gain
        * (
            disagreement[recovery_mask]
            - cfg.lidar_disagreement_threshold
        )
    )
    proposed["lidar"] += recovery

    out = {}
    for s in SENSORS:
        out[s] = _bounded_update(
            original=r[s],
            proposed=proposed[s],
            max_delta=deltas[s],
            floor=floors[s],
        )

    return out


def coordination_diagnostics(
    original: Mapping[str, np.ndarray],
    coordinated: Mapping[str, np.ndarray],
) -> Dict[str, Dict[str, float]]:
    result: Dict[str, Dict[str, float]] = {}
    for s in SENSORS:
        a = _as_1d(original[s])
        b = _as_1d(coordinated[s])
        d = b - a
        result[s] = {
            "original_min": float(np.min(a)),
            "original_max": float(np.max(a)),
            "original_mean": float(np.mean(a)),
            "coordinated_min": float(np.min(b)),
            "coordinated_max": float(np.max(b)),
            "coordinated_mean": float(np.mean(b)),
            "mean_abs_change": float(np.mean(np.abs(d))),
            "max_abs_change": float(np.max(np.abs(d))),
        }
    return result
