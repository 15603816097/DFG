from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.reliability.oracle_factor_reliability import OracleReliabilityConfig

SENSORS = ("gps", "imu", "lidar", "camera")


def corr(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def main():
    path = (
        ROOT
        / "results/predictive_factor_reliability_v3_same_source"
        / "factor_reliability_training_data.npz"
    )
    if not path.exists():
        raise FileNotFoundError(path)

    d = np.load(path, allow_pickle=False)
    cfg = OracleReliabilityConfig()

    dims = {"gps": 21, "imu": 20, "lidar": 19, "camera": 16}

    print("=" * 110)
    print("VALIDATE SAME-SOURCE FACTOR LABELS")
    print("=" * 110)

    n = None
    for s, dim in dims.items():
        x = np.asarray(d[f"{s}_features"])
        yc = np.asarray(d[f"{s}_current_factor_reliability"])
        yf = np.asarray(d[f"{s}_future_factor_reliability"])

        assert x.ndim == 2 and x.shape[1] == dim, (s, x.shape)
        assert len(x) == len(yc) == len(yf)
        assert np.all(np.isfinite(x))
        assert np.all(np.isfinite(yc))
        assert np.all((yc >= 0.01) & (yc <= 1.0))

        n = len(x) if n is None else n
        assert len(x) == n

        print(
            f"{s:8s} X={x.shape} "
            f"current mean={yc.mean():.6f} std={yc.std():.6f} "
            f"future mean={yf.mean():.6f} std={yf.std():.6f}"
        )

    gps_e = d["diagnostic_gps_error_m"]
    imu_e = d["diagnostic_imu_rotation_error_deg"]
    lidar_te = d["diagnostic_lidar_translation_error_m"]
    lidar_re = d["diagnostic_lidar_rotation_error_deg"]
    camera_te = d["diagnostic_camera_translation_error_m"]
    camera_re = d["diagnostic_camera_rotation_error_deg"]

    # Sanity: reliability should be negatively associated with actual error.
    gps_r = d["gps_current_factor_reliability"]
    imu_r = d["imu_current_factor_reliability"]
    lidar_r = d["lidar_current_factor_reliability"]
    camera_r = d["camera_current_factor_reliability"]

    print()
    print("Error/reliability correlations (negative is expected):")
    print("GPS error vs reliability:", corr(gps_e, gps_r))
    print("IMU error vs reliability:", corr(imu_e, imu_r))
    print("LiDAR t error vs reliability:", corr(lidar_te, lidar_r))
    print("LiDAR r error vs reliability:", corr(lidar_re, lidar_r))
    print("Camera t error vs reliability:", corr(camera_te, camera_r))
    print("Camera r error vs reliability:", corr(camera_re, camera_r))

    assert corr(gps_e, gps_r) < 0.0
    assert corr(imu_e, imu_r) < 0.0

    # Pose reliability is a product of translation and rotation reliability,
    # so either component can dominate. Require at least one negative relation.
    assert min(corr(lidar_te, lidar_r), corr(lidar_re, lidar_r)) < 0.0
    assert min(corr(camera_te, camera_r), corr(camera_re, camera_r)) < 0.0

    print()
    print("PASS: features and labels are dimensionally aligned and labels track actual same-source factor errors.")


if __name__ == "__main__":
    main()
