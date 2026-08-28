from __future__ import annotations

import os
import sys
from typing import Dict

import numpy as np


ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


GT_CANDIDATES = [
    os.path.join(
        ROOT,
        "results",
        "ground_truth",
        "trajectory.txt",
    ),
    os.path.join(
        ROOT,
        "results",
        "gps_ground_truth",
        "trajectory.txt",
    ),
]

METHODS = {
    "Degraded Fixed Four-Sensor": os.path.join(
        ROOT,
        "results",
        "degraded_four_sensor_fixed_fg",
        "trajectory.txt",
    ),
    "Predicted Factor Reliability V1": os.path.join(
        ROOT,
        "results",
        "predicted_factor_reliability_fg",
        "trajectory.txt",
    ),
    "GPS+IMU+Camera Predictive": os.path.join(
        ROOT,
        "results",
        "sensorwise_predictive_ablation",
        "gps_imu_camera",
        "trajectory.txt",
    ),
    "Cross-Sensor Coordinated V1": os.path.join(
        ROOT,
        "results",
        "cross_sensor_coordinated_fg_v1",
        "trajectory.txt",
    ),
    "Oracle Factor Reliability": os.path.join(
        ROOT,
        "results",
        "oracle_four_sensor_fg",
        "trajectory.txt",
    ),
}


def _load_gt():
    for path in GT_CANDIDATES:
        if os.path.isfile(path):
            return np.loadtxt(path, dtype=np.float64), path
    raise FileNotFoundError(
        "Ground-truth trajectory not found. Checked:\n  "
        + "\n  ".join(GT_CANDIDATES)
    )


def _metrics(gt: np.ndarray, est: np.ndarray) -> Dict[str, float]:
    gt = np.asarray(gt, dtype=np.float64)
    est = np.asarray(est, dtype=np.float64)

    n = min(len(gt), len(est))
    if n == 0:
        raise ValueError("Empty trajectory.")

    gt = gt[:n, :3]
    est = est[:n, :3]

    e = est - gt
    e3 = np.linalg.norm(e, axis=1)
    e2 = np.linalg.norm(e[:, :2], axis=1)

    return {
        "ATE3D": float(np.sqrt(np.mean(e3 ** 2))),
        "ATE2D": float(np.sqrt(np.mean(e2 ** 2))),
        "Mean3D": float(np.mean(e3)),
        "Max3D": float(np.max(e3)),
        "ZRMSE": float(np.sqrt(np.mean(e[:, 2] ** 2))),
    }


def main():
    print("=" * 120)
    print("CROSS-SENSOR COORDINATION V1 EVALUATION")
    print("=" * 120)

    gt, gt_path = _load_gt()
    print("Ground truth:", gt_path)
    print()

    rows = {}
    for name, path in METHODS.items():
        if not os.path.isfile(path):
            print(f"{name:40s}: NOT RUN")
            continue

        est = np.loadtxt(path, dtype=np.float64)
        m = _metrics(gt, est)
        rows[name] = m
        print(
            f"{name:40s} "
            f"ATE3D={m['ATE3D']:.6f} "
            f"ATE2D={m['ATE2D']:.6f} "
            f"Mean3D={m['Mean3D']:.6f} "
            f"Max3D={m['Max3D']:.6f} "
            f"ZRMSE={m['ZRMSE']:.6f}"
        )

    fixed = rows.get("Degraded Fixed Four-Sensor")
    coord = rows.get("Cross-Sensor Coordinated V1")
    best_subset = rows.get("GPS+IMU+Camera Predictive")
    oracle = rows.get("Oracle Factor Reliability")

    print()
    if fixed and coord:
        imp = 100.0 * (
            fixed["ATE3D"] - coord["ATE3D"]
        ) / fixed["ATE3D"]
        print(
            "Coordinated improvement vs Fixed: "
            f"{imp:+.2f}%"
        )

    if best_subset and coord:
        imp = 100.0 * (
            best_subset["ATE3D"] - coord["ATE3D"]
        ) / best_subset["ATE3D"]
        print(
            "Coordinated improvement vs GPS+IMU+Camera: "
            f"{imp:+.2f}%"
        )

    if oracle and coord:
        gap = 100.0 * (
            coord["ATE3D"] - oracle["ATE3D"]
        ) / oracle["ATE3D"]
        print(
            "Coordinated gap from Oracle: "
            f"{gap:+.2f}%"
        )

    out = os.path.join(
        ROOT,
        "results",
        "cross_sensor_coordination_v1_evaluation.txt",
    )
    os.makedirs(os.path.dirname(out), exist_ok=True)

    with open(out, "w", encoding="utf-8") as f:
        for name, m in rows.items():
            f.write(
                f"{name},"
                f"{m['ATE3D']:.10f},"
                f"{m['ATE2D']:.10f},"
                f"{m['Mean3D']:.10f},"
                f"{m['Max3D']:.10f},"
                f"{m['ZRMSE']:.10f}\n"
            )

    print()
    print("Saved:", out)


if __name__ == "__main__":
    main()
