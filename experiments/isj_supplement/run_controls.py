from __future__ import annotations

"""ISJ exploratory controls using frozen Stage-7 predictions and warm starts.

Controls (no training or hyperparameter search):
- Fixed-FGO, solved again from the exact Stage-1 fixed trajectory.
- Constant-mean uncertainty per sensor, retaining the frozen reliability.
- Permuted uncertainty within valid timestamps, retaining the marginal distribution.

Neither the original Stage-7 metrics nor checkpoint files are modified.
These are post-hoc controls, NOT independent prospective held-out evidence.
"""

import argparse
import csv
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.cross_sequence import run_stage7_factor_graph as s7
from experiments.cross_sequence.stage7_config import TEST_SEQUENCES, WINDOW, sequence_out

METHODS = {"GRU": "v5", "MAMBA": "v6"}
DEFAULT_SEEDS = (11, 23, 47)


def transform_uncertainty(u, mode, seed, source, sensor, window=WINDOW):
    """Only alter timestamps whose temporal encoder has a complete input window."""
    arr = np.asarray(u, dtype=np.float64).reshape(-1)
    if not np.isfinite(arr).all() or np.any((arr < 0) | (arr > 1)):
        raise ValueError("Expected finite uncertainty values in [0, 1]")
    start = window - 1
    if start < 0 or len(arr) <= start:
        raise ValueError(f"Too few samples: len={len(arr)}, window={window}")
    output = arr.copy()
    valid = arr[start:]
    if mode == "constant_mean":
        output[start:] = float(np.mean(valid))
    elif mode == "permuted":
        stable_offset = zlib.crc32(f"{source}/{sensor}".encode("utf-8"))
        rng = np.random.default_rng(int(seed) + stable_offset)
        output[start:] = rng.permutation(valid)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return output


def preflight(seq):
    out = sequence_out(seq)
    fixed_path = out / "factor_graph" / "FINAL_S0_fixed" / "poses.npy"
    summary_path = out / "stage7_two_stage_results.csv"
    required = [
        out / "sensor_data.npz",
        out / "lidar_factor_data.npz",
        out / "camera_factor_data.npz",
        out / "ground_truth.txt",
        fixed_path,
        summary_path,
    ]
    for source in METHODS.values():
        for sensor in ("gps", "imu", "camera"):
            required += [
                out / "predictions" / source / f"{sensor}_reliability.txt",
                out / "predictions" / source / f"{sensor}_uncertainty.txt",
            ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Stage-7 inputs are not ready. Missing:\n  " + "\n  ".join(missing)
            + "\nFirst restore original KITTI data and run Stage-7 preprocessing,"
            + " V5/V6 inference, and run_stage7_two_stage.py."
        )
    m = s7.load_measurements(seq)
    n = len(m.gps_local)
    poses = np.load(fixed_path, allow_pickle=False)
    if poses.shape != (n, 4, 4) or not np.isfinite(poses).all():
        raise ValueError(f"Invalid fixed warm-start poses: {poses.shape}, n={n}")
    with summary_path.open(newline="", encoding="utf-8") as stream:
        reference = {row["case"]: row for row in csv.DictReader(stream)}
    for key in ("S0_fixed", "S1_v5_gru", "S2_v6_all_mamba"):
        if key not in reference:
            raise KeyError(f"Missing original two-stage result {key}")
    return poses, reference


def run_case_cached(seq, name, gps_src=None, imu_src=None, cam_src=None, rerun=False):
    path = sequence_out(seq) / "factor_graph" / name / "metrics.json"
    if path.exists() and not rerun:
        return json.loads(path.read_text(encoding="utf-8")), None, "cached"
    begin = time.perf_counter()
    metrics = s7.run_case(
        seq=seq, name=name, gps_src=gps_src, imu_src=imu_src, cam_src=cam_src
    )
    return metrics, time.perf_counter() - begin, "computed"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sequence", default="all", choices=("all",) + TEST_SEQUENCES)
    p.add_argument("--seeds", default="11,23,47",
                   help="Predeclared permutation seeds, comma-separated.")
    p.add_argument("--controls", default="all",
                   choices=("all", "fixed2", "constant_mean", "permuted"))
    p.add_argument("--rerun", action="store_true",
                   help="Recompute ONLY ISJ_CTRL_* results, never original Stage-7.")
    args = p.parse_args()
    seeds = tuple(int(s.strip()) for s in args.seeds.split(",") if s.strip())
    if not seeds or len(set(seeds)) != len(seeds) or any(x < 0 for x in seeds):
        p.error("--seeds must contain distinct nonnegative integers")
    sequences = TEST_SEQUENCES if args.sequence == "all" else (args.sequence,)

    for seq in sequences:
        fixed_poses, reference = preflight(seq)
        out = sequence_out(seq)
        rows = []
        original_init, original_pred = s7.initial_trajectory_from_pose, s7.pred

        def frozen_initializer(gtsam, measurements, initial_pose):
            if len(measurements.gps_local) != len(fixed_poses):
                raise ValueError("Stage-1 trajectory length mismatch")
            return [s7._pose3_from_matrix(gtsam, mat) for mat in fixed_poses]

        def add_row(name, method, control, seed, metrics, seconds, status):
            rows.append({
                "sequence": seq, "case": name, "model": method, "control": control,
                "seed": seed if seed is not None else "",
                "ATE3D": float(metrics["ATE3D"]), "ATE2D": float(metrics["ATE2D"]),
                "ZRMSE": float(metrics["ZRMSE"]),
                "vs_original_fixed_pct": 100.0 *
                    (float(reference["S0_fixed"]["ATE3D"]) - float(metrics["ATE3D"]))
                    / float(reference["S0_fixed"]["ATE3D"]),
                "solver_seconds": round(seconds, 3) if seconds is not None else "",
                "status": status,
            })

        try:
            s7.initial_trajectory_from_pose = frozen_initializer
            if args.controls in ("all", "fixed2"):
                metrics, seconds, status = run_case_cached(
                    seq, "ISJ_CTRL_FIXED_SECOND_PASS", rerun=args.rerun
                )
                add_row("ISJ_CTRL_FIXED_SECOND_PASS", "NONE", "fixed2", None,
                        metrics, seconds, status)

            for model, source in METHODS.items():
                if args.controls == "fixed2":
                    break
                cases = []
                if args.controls in ("all", "constant_mean"):
                    cases.append(("constant_mean", None))
                if args.controls in ("all", "permuted"):
                    cases += [("permuted", seed) for seed in seeds]

                for mode, seed in cases:
                    def controlled_pred(s, src, sensor, n, _mode=mode, _seed=seed):
                        r, u = original_pred(s, src, sensor, n)
                        return r, transform_uncertainty(
                            u, _mode, 0 if _seed is None else _seed, src, sensor
                        )
                    s7.pred = controlled_pred
                    suffix = "MEAN_U" if mode == "constant_mean" else f"PERM_U_SEED_{seed}"
                    name = f"ISJ_CTRL_{model}_{suffix}"
                    metrics, seconds, status = run_case_cached(
                        seq, name, source, source, source, rerun=args.rerun
                    )
                    add_row(name, model, mode, seed, metrics, seconds, status)
        finally:
            s7.initial_trajectory_from_pose = original_init
            s7.pred = original_pred

        csv_path = out / "isj_controls_results.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        protocol = {
            "classification": "exploratory_post_hoc_not_blind",
            "sequence": seq,
            "fixed_warmstart": "FINAL_S0_fixed/poses.npy",
            "model_training": False,
            "model_parameters_changed": False,
            "noise_mapping_changed": False,
            "lidar_weight": "fixed",
            "comparison": "Original S0/S1/S2 from stage7_two_stage_results.csv",
            "warmup_frames_preserved": WINDOW - 1,
            "constant_mean": "Sensor-specific mean of valid uncertainty frames",
            "permutation": "Sensor-specific temporal shuffle preserving distribution",
            "permutation_seeds": list(seeds),
            "caveats": [
                "OXTS/GPS-derived reference is not independent of GPS inputs",
                "All sequences are from 2011_10_03",
                "Any interpretation of these controls is exploratory",
            ],
        }
        (out / "isj_controls_protocol.json").write_text(
            json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\nCompleted: {seq}\nResults: {csv_path}")
        for row in rows:
            print(f"{row['case']:42s} ATE3D={row['ATE3D']:.6f} m")

if __name__ == "__main__":
    main()
