from __future__ import annotations
"""Stage-7 conventional residual-adaptive covariance FGO comparisons.

ONE frozen outer-loop residual-based noise adaptation from the saved Stage-1
fixed FGO estimate, followed by the SAME LM iteration and initialization
protocol as GRU/Mamba Stage-7 second stage. No neural network, GT weighting,
training, or parameter search. Exploratory/post-hoc Stage-7 comparison.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.cross_sequence import run_stage7_factor_graph as s7
from experiments.cross_sequence.stage7_config import TEST_SEQUENCES, sequence_out
from experiments.isj_supplement.adaptive_fg_common import (
    CHI_3D_95, CHI_6D_95, MAX_SIGMA_MULTIPLIER, build_factor_graph,
    export_poses, load_first_stage, optimize_graph,
)

CASES = {
    "gps": ("ISJ_RESADAPT_GPS", ("gps",)),
    "all": ("ISJ_RESADAPT_GPS_IMU_CAMERA", ("gps", "imu", "camera")),
}


def run(seq: str, mode: str, rerun: bool) -> dict:
    name, selected = CASES[mode]
    directory = sequence_out(seq)
    saved = directory / "factor_graph" / name
    metrics_file = saved / "metrics.json"

    if metrics_file.exists() and not rerun:
        print(f"SKIP cached: {seq} {name}")
        payload = json.loads(metrics_file.read_text(encoding="utf-8"))
        return {
            "sequence": seq, "case": name, "ATE3D": payload["metrics"]["ATE3D"],
            "ATE2D": payload["metrics"]["ATE2D"],
            "ZRMSE": payload["metrics"]["ZRMSE"],
            "solver_seconds": payload["solver_seconds"],
            "cache": "cached",
        }

    m, first = load_first_stage(seq)
    start = time.perf_counter()
    graph, initial, stats = build_factor_graph(m, first, selected)
    graph_seconds = time.perf_counter() - start
    initial_objective = float(graph.error(initial))

    start = time.perf_counter()
    optimized = optimize_graph(graph, initial)
    solver_seconds = time.perf_counter() - start
    final_objective = float(graph.error(optimized))
    if not np.isfinite([initial_objective, final_objective]).all():
        raise RuntimeError("Nonfinite graph objective")
    poses = export_poses(optimized, len(first))
    if not np.isfinite(poses).all():
        raise RuntimeError("Nonfinite estimated poses")

    metrics = s7.evaluate(seq, poses[:, :3, 3])
    saved.mkdir(parents=True, exist_ok=True)
    np.save(saved / "poses.npy", poses)
    np.savetxt(saved / "trajectory.txt", poses[:, :3, 3], fmt="%.9f")

    metadata = {
        "sequence": seq,
        "case": name,
        "kind": "one_step_residual_adaptive_covariance_factor_graph",
        "selected_sensors": list(selected),
        "lidar_covariance": "fixed",
        "warm_start": "factor_graph/FINAL_S0_fixed/poses.npy",
        "reference_used_for_weighting_or_initialization": False,
        "deep_model_inference": False,
        "kernel": "sigma_multiplier=clip(sqrt(max(1, ||whitened_residual||/c)), 1, 4)",
        "c_3d": CHI_3D_95,
        "c_6d": CHI_6D_95,
        "sigma_multiplier_max": MAX_SIGMA_MULTIPLIER,
        "weights_computed_once": True,
        "lm_max_iterations": 100,
        "lm_relative_error_tolerance": 1e-7,
        "graph_initial_objective": initial_objective,
        "graph_final_objective": final_objective,
        "graph_build_seconds": graph_seconds,
        "solver_seconds": solver_seconds,
        "sigma_statistics": stats,
        "metrics": metrics,
        "protocol": "exploratory_post_hoc",
        "notes": [
            "This is one conventional residual-weighted baseline; not an exhaustive state of the art comparison.",
            "Stored OXTS/GPS reference is not independent of GPS sensor measurements.",
            "Stage-7 test sequences share the same date.",
            "No hyperparameters selected based on Stage-7 ATE results.",
        ],
    }
    metrics_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"{seq} {name} ATE3D={metrics['ATE3D']:.6f} m "
          f"objective={initial_objective:.3f}->{final_objective:.3f} "
          f"solver={solver_seconds:.2f}s")
    return {
        "sequence": seq, "case": name,
        "ATE3D": metrics["ATE3D"], "ATE2D": metrics["ATE2D"],
        "ZRMSE": metrics["ZRMSE"], "solver_seconds": solver_seconds,
        "cache": "computed",
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sequence", choices=("all",) + TEST_SEQUENCES, default="all")
    p.add_argument("--mode", choices=("both",) + tuple(CASES), default="both")
    p.add_argument("--rerun", action="store_true")
    args = p.parse_args()

    sequences = TEST_SEQUENCES if args.sequence == "all" else (args.sequence,)
    modes = tuple(CASES) if args.mode == "both" else (args.mode,)
    for seq in sequences:
        rows = [run(seq, mode, args.rerun) for mode in modes]
        file = sequence_out(seq) / "isj_residual_adaptive_results.csv"
        with file.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Results: {file}")


if __name__ == "__main__":
    main()
