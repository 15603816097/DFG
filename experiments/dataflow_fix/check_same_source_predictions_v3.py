from __future__ import annotations

from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/predictive_factor_reliability_v3_same_source"

SENSORS = ("gps", "imu", "lidar", "camera")


def main():
    train = np.load(
        OUT / "factor_reliability_training_data.npz",
        allow_pickle=False,
    )

    n = len(
        train["gps_current_factor_reliability"]
    )

    print("=" * 110)
    print("CHECK TARGET-ALIGNED SAME-SOURCE PREDICTIONS")
    print("=" * 110)

    for s in SENSORS:
        path = (
            OUT
            / f"{s}_predictive_prior_target_aligned.txt"
        )
        if not path.exists():
            raise FileNotFoundError(path)

        p = np.loadtxt(
            path, dtype=np.float64
        ).reshape(-1)

        assert len(p) == n, (
            s, len(p), n
        )
        assert np.all(np.isfinite(p))
        assert np.all(
            (p >= 0.0) & (p <= 1.0)
        )

        print(
            f"{s:8s} "
            f"min/max/mean/std = "
            f"{p.min():.6f} / "
            f"{p.max():.6f} / "
            f"{p.mean():.6f} / "
            f"{p.std():.6f}"
        )

    print()
    print("PASS: factor-graph prediction files are complete and frame aligned.")


if __name__ == "__main__":
    main()
