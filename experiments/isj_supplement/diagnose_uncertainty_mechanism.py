from __future__ import annotations

"""Read-only diagnostic of evidence uncertainty, factor noise, and pose error.

Inputs: FROZEN V5/V6 reliability/uncertainty sequences, Stage-7 measurements,
GROUND TRUTH FOR ANALYSIS ONLY, and completed uniform-convergence poses from:
    FINAL_S1_v5_gru / FINAL_S2_v6_all_mamba
    ABL_V5_GRU_RELIABILITY_ONLY / ABL_V6_MAMBA_RELIABILITY_ONLY

Writes descriptive analysis CSVs without editing any original Stage-7 results.
Mechanism analysis is post-hoc, non-causal: comparing the same frame's
uncertainty with optimized pose-error difference does not attribute error
to a particular sensor. All three sequences share one collection date;
OXTS/GPS-derived reference is not independent of GPS measurements.
"""

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.cross_sequence import run_stage7_factor_graph as s7
from experiments.cross_sequence.stage7_config import (
    OUT_ROOT, TEST_SEQUENCES, WINDOW, sequence_out
)
from src.factor_graph.sensor_factor_config import FourSensorFactorConfig


CASES = {
    "GRU": {
        "source": "v5",
        "full": "FINAL_S1_v5_gru",
        "reliability": "ABL_V5_GRU_RELIABILITY_ONLY",
    },
    "Mamba": {
        "source": "v6",
        "full": "FINAL_S2_v6_all_mamba",
        "reliability": "ABL_V6_MAMBA_RELIABILITY_ONLY",
    },
}

SENSORS = ("gps", "imu", "camera")


def read_poses(seq: str, case: str, n: int) -> np.ndarray:
    file = (sequence_out(seq) / "factor_graph"
            / ("ISJ_CONVERGENCE_" + case) / "poses.npy")
    if not file.is_file():
        raise FileNotFoundError(
            f"Missing convergence-controlled poses: {file}\n"
            "Run run_uniform_convergence.py --sequence all "
            "--family core and --family ablations before this diagnostic."
        )
    poses = np.load(file, allow_pickle=False)
    if poses.shape != (n, 4, 4) or not np.isfinite(poses).all():
        raise ValueError(f"{file}: invalid pose array {poses.shape}")
    return poses[:, :3, 3]


def compare_error_curves(gt: np.ndarray, full: np.ndarray,
                         reliability: np.ndarray) -> dict[str, np.ndarray]:
    """Positive per-frame gain = Dynamic U lower error than Reliability Only."""
    a, b, c = (np.asarray(x, dtype=float) for x in
               (gt, full, reliability))
    if a.shape != b.shape or a.shape != c.shape or a.ndim != 2 or a.shape[1] != 3:
        raise ValueError(
            f"Expected equal Nx3 arrays, got {a.shape},{b.shape},{c.shape}"
        )
    if not all(np.isfinite(x).all() for x in (a, b, c)):
        raise ValueError("Non-finite evaluation data")
    full_err = np.linalg.norm(b - a, axis=1)
    rel_err = np.linalg.norm(c - a, axis=1)
    return {
        "dynamic_error_m": full_err,
        "reliability_error_m": rel_err,
        "dynamic_gain_per_frame_m": rel_err - full_err,
    }


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Descriptive Pearson coefficient; NaN when a signal is constant."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.ndim != 1 or b.shape != a.shape or len(a) < 3:
        return float("nan")
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def high_low_effect(u: np.ndarray, full_error: np.ndarray,
                    reliability_error: np.ndarray) -> dict:
    """Separate lowest/highest uncertainty quartiles, no hypothesis testing."""
    u = np.asarray(u, dtype=float)
    fe = np.asarray(full_error, dtype=float)
    re = np.asarray(reliability_error, dtype=float)
    if not (u.shape == fe.shape == re.shape) or len(u) == 0:
        raise ValueError("Bad uncertainty/error arrays")
    if not all(np.isfinite(x).all() for x in (u, fe, re)):
        raise ValueError("Nonfinite uncertainty/error arrays")
    q25, q75 = np.quantile(u, [0.25, 0.75])
    if abs(q75 - q25) < 1e-12:
        return {
            "quartiles_distinct": False,
            "low_count": 0, "high_count": 0,
            "low_dynamic_minus_reliability_rmse_m": float("nan"),
            "high_dynamic_minus_reliability_rmse_m": float("nan"),
        }
    low, high = (u <= q25), (u >= q75)
    def rmse(v, mask):
        return float(np.sqrt(np.mean(np.square(v[mask]))))
    return {
        "quartiles_distinct": True,
        "low_count": int(np.sum(low)),
        "high_count": int(np.sum(high)),
        "low_dynamic_minus_reliability_rmse_m": rmse(fe, low) - rmse(re, low),
        "high_dynamic_minus_reliability_rmse_m": rmse(fe, high) - rmse(re, high),
    }


def sigmas(cfg: FourSensorFactorConfig, sensor: str,
           r: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Replicate the exact Stage-7 noise mapping for a single sensor."""
    full, rel = [], []
    for ri, ui in zip(r, u):
        if sensor == "gps":
            full.append([s7.gps_sigma(cfg, ri, ui)])
            rel.append([s7.gps_sigma(cfg, ri, 0.0)])
        elif sensor == "imu":
            full.append([s7.imu_sigma(cfg, ri, ui)])
            rel.append([s7.imu_sigma(cfg, ri, 0.0)])
        elif sensor == "camera":
            full.append(list(s7.cam_sigma(cfg, ri, ui)))
            rel.append(list(s7.cam_sigma(cfg, ri, 0.0)))
        else:
            raise ValueError(f"Unknown sensor {sensor!r}")
    return np.asarray(full, float), np.asarray(rel, float)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No records for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(seq: str, model: str, output_root: Path) -> list[dict]:
    spec = CASES[model]
    m = s7.load_measurements(seq)
    n = len(m.gps_local)
    gt_path = sequence_out(seq) / "ground_truth.txt"
    gt = np.loadtxt(gt_path, dtype=float)
    if gt.ndim != 2 or len(gt) != n or gt.shape[1] < 3:
        raise ValueError(f"{seq}: ground truth expected N x >=3, got {gt.shape}")
    gt = gt[:, :3]
    full = read_poses(seq, spec["full"], n)
    reliability = read_poses(seq, spec["reliability"], n)
    comparison = compare_error_curves(gt, full, reliability)

    cfg = FourSensorFactorConfig()
    # Encoder window is [t-63,...,t]. The first 63 predictions are warm-up.
    valid_start = WINDOW - 1
    if n <= valid_start:
        raise ValueError(f"{seq}: n={n} too small for WINDOW={WINDOW}")

    frame = [{
        "sequence": seq, "model": model, "frame": i,
        "timestamp_s": float(m.timestamps[i]),
        "is_model_valid_window": i >= valid_start,
        "dynamic_error_m": float(comparison["dynamic_error_m"][i]),
        "reliability_error_m": float(comparison["reliability_error_m"][i]),
        "dynamic_gain_per_frame_m": float(
            comparison["dynamic_gain_per_frame_m"][i]
        ),
    } for i in range(n)]

    report = []
    for sensor in SENSORS:
        r, u = s7.pred(seq, spec["source"], sensor, n)
        noise_full, noise_rel = sigmas(cfg, sensor, r, u)
        valid = slice(valid_start, None)
        diff = noise_full[valid] - noise_rel[valid]
        effect = high_low_effect(
            u[valid], comparison["dynamic_error_m"][valid],
            comparison["reliability_error_m"][valid]
        )
        report.append({
            "sequence": seq,
            "model": model,
            "sensor": sensor,
            "valid_n": n - valid_start,
            "uncertainty_mean": float(np.mean(u[valid])),
            "uncertainty_std": float(np.std(u[valid])),
            "reliability_mean": float(np.mean(r[valid])),
            "uncertainty_error_gain_pearson": safe_corr(
                u[valid], comparison["dynamic_gain_per_frame_m"][valid]
            ),
            "mean_abs_sigma_difference_dynamic_vs_reliability":
                float(np.mean(np.abs(diff))),
            "sigma_diff_positive_fraction": float(np.mean(diff > 1e-10)),
            "sigma_diff_negative_fraction": float(np.mean(diff < -1e-10)),
            "overall_dynamic_ATE3D_m": float(np.sqrt(np.mean(
                np.square(comparison["dynamic_error_m"])
            ))),
            "overall_reliability_ATE3D_m": float(np.sqrt(np.mean(
                np.square(comparison["reliability_error_m"])
            ))),
            **effect,
            "interpretation": "descriptive_noncausal_post_hoc",
        })
        for i in range(n):
            frame[i][f"{sensor}_reliability"] = float(r[i])
            frame[i][f"{sensor}_uncertainty"] = float(u[i])
            # IMU/Camera use the frame i signal for factor i-1 -> i.
            frame[i][f"{sensor}_sigma_dynamic"] = ";".join(
                f"{x:.9g}" for x in noise_full[i]
            )
            frame[i][f"{sensor}_sigma_reliability_only"] = ";".join(
                f"{x:.9g}" for x in noise_rel[i]
            )

    seq_key = seq.split("_drive_")[1][:4]
    filename = output_root / f"isj_uncertainty_frames_{seq_key}_{model.lower()}.csv"
    write_csv(filename, frame)
    print(f"{seq_key}/{model}: dynamic ATE3D={report[0]['overall_dynamic_ATE3D_m']:.4f}m,"
          f" reliability only={report[0]['overall_reliability_ATE3D_m']:.4f}m")
    for row in report:
        print(f"  {row['sensor']:<6} u_std={row['uncertainty_std']:.5f} "
              f"corr(u, frame_gain)={row['uncertainty_error_gain_pearson']:+.4f} "
              f"low-quartile [dyn-rel]={row['low_dynamic_minus_reliability_rmse_m']:+.4f}m "
              f"high-quartile={row['high_dynamic_minus_reliability_rmse_m']:+.4f}m")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sequence", choices=("all",) + TEST_SEQUENCES, default="all")
    p.add_argument("--model", choices=("all",) + tuple(CASES), default="all")
    p.add_argument("--output-root", type=Path, default=OUT_ROOT)
    args = p.parse_args()
    sequences = TEST_SEQUENCES if args.sequence == "all" else (args.sequence,)
    models = tuple(CASES) if args.model == "all" else (args.model,)
    all_rows = []
    for seq in sequences:
        for model in models:
            all_rows.extend(run(seq, model, args.output_root))
    out = args.output_root / "isj_uncertainty_mechanism_summary.csv"
    write_csv(out, all_rows)
    print(f"\nSaved descriptive sensor analysis: {out}")
    print("WARNING: high/low quartiles and correlations are descriptive only;"
          " sensor contribution is NOT causally identified.")
    print("Ground-truth OXTS/GPS provenance and post-hoc study limitations apply.")


if __name__ == "__main__":
    main()
