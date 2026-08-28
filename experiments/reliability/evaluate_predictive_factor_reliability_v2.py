from __future__ import annotations
import os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SENSORS = ("gps", "imu", "lidar", "camera")

ORACLE_PATH = os.path.join(ROOT, "results", "oracle_factor_reliability", "oracle_factor_reliability.npz")
PRED_DIR = os.path.join(ROOT, "results", "predictive_factor_reliability_v2")

HORIZON = 3
SEQUENCE_LENGTH = 64

def metrics(y, p):
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    mae = float(np.mean(np.abs(y - p)))
    rmse = float(np.sqrt(np.mean((y - p) ** 2)))
    corr = float(np.corrcoef(y, p)[0, 1]) if np.std(y) > 1e-12 and np.std(p) > 1e-12 else 0.0
    bad = y < 0.5
    severe = y < 0.2
    bad_recall = float(np.mean(p[bad] < 0.5)) if np.any(bad) else 1.0
    severe_recall = float(np.mean(p[severe] < 0.2)) if np.any(severe) else 1.0
    return mae, rmse, corr, bad_recall, severe_recall

def main():
    oracle = np.load(ORACLE_PATH, allow_pickle=False)
    n = len(oracle["gps_reliability"])
    test_start = int(n * 0.85)
    start = max(test_start, SEQUENCE_LENGTH - 1 + HORIZON)

    print("=" * 112)
    print("STRICT TEST-ONLY PREDICTIVE FACTOR RELIABILITY V2")
    print("=" * 112)
    print("Test target-frame range:", start, "->", n - 1)

    for sensor in SENSORS:
        target = np.asarray(oracle[f"{sensor}_reliability"], dtype=np.float64)
        prediction = np.loadtxt(
            os.path.join(PRED_DIR, f"{sensor}_predictive_prior_target_aligned.txt"),
            dtype=np.float64,
        ).reshape(-1)

        end = min(len(target), len(prediction))
        mae, rmse, corr, bad_recall, severe_recall = metrics(
            target[start:end],
            prediction[start:end],
        )

        print()
        print(sensor.upper())
        print(f"MAE               : {mae:.6f}")
        print(f"RMSE              : {rmse:.6f}")
        print(f"Correlation       : {corr:.6f}")
        print(f"BadRecall@0.5     : {bad_recall:.6f}")
        print(f"SevereRecall@0.2  : {severe_recall:.6f}")

if __name__ == "__main__":
    main()
