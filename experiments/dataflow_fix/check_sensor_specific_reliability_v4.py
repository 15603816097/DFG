from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/sensor_specific_reliability_v4"
DATA = (
    ROOT
    / "results/predictive_factor_reliability_v3_same_source"
    / "factor_reliability_training_data.npz"
)
SENSORS = ("gps", "imu", "camera")


def main():
    if not DATA.exists():
        raise FileNotFoundError(DATA)

    raw = np.load(DATA, allow_pickle=False)
    n = len(raw["gps_features"])

    print("=" * 100)
    print("CHECK SENSOR-SPECIFIC V4")
    print("=" * 100)

    summary = json.loads(
        (OUT / "training_summary.json").read_text(
            encoding="utf-8"
        )
    )

    for s in SENSORS:
        p = np.loadtxt(
            OUT / f"{s}_predictive_prior_target_aligned.txt"
        ).reshape(-1)
        src = np.loadtxt(
            OUT / f"{s}_prediction_source_frame.txt",
            dtype=np.int64,
        ).reshape(-1)

        assert len(p) == n
        assert len(src) == n
        assert np.all(np.isfinite(p))
        assert np.all((p >= 0.0) & (p <= 1.0))
        assert (OUT / s / "best_model.pt").exists()
        assert (OUT / s / "normalization.npz").exists()

        corr = summary["results"][s]["test"]["corr"]
        print(
            f"{s:8s} corr={corr:+.6f} | "
            f"pred min/max/mean/std="
            f"{p.min():.6f}/{p.max():.6f}/"
            f"{p.mean():.6f}/{p.std():.6f}"
        )

    print()
    print("PASS: V4 model files and target-aligned predictions are complete.")
    print("NOTE: PASS here validates files/alignment, not scientific performance.")
    print("Send the terminal output before wiring V4 into E7/E8.")


if __name__ == "__main__":
    main()
