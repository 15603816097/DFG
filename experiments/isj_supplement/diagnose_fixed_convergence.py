from __future__ import annotations
"""Diagnose Stage-7 fixed FGO repeat-optimization, especially 0042.

Rebuild the same fixed-weight factors and evaluate their *objective function*
at saved Stage-1 and Stage-2 poses. Optional third pass tests remaining
progress without touching original Stage-7 or ISJ_CTRL directories.
"""
import argparse
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
    build_factor_graph, export_poses, load_first_stage, optimize_graph,
    validate_poses, values_from_poses,
)


def run(seq: str, third_pass: bool) -> dict:
    m, first_poses = load_first_stage(seq)
    n = len(first_poses)
    out = sequence_out(seq)
    second_path = out / "factor_graph" / "ISJ_CTRL_FIXED_SECOND_PASS" / "poses.npy"
    if not second_path.is_file():
        raise FileNotFoundError(
            f"Missing {second_path}; run run_controls.py --controls fixed2 first")
    second_poses = np.load(second_path, allow_pickle=False)
    validate_poses(second_poses, n)

    graph, _init, stats = build_factor_graph(m, first_poses, adaptive_sensors=())
    first_objective = float(graph.error(values_from_poses(first_poses)))
    second_objective = float(graph.error(values_from_poses(second_poses)))
    if not np.isfinite([first_objective, second_objective]).all():
        raise ValueError("Fixed graph produced non-finite objective")

    shift = np.linalg.norm(
        first_poses[:, :3, 3] - second_poses[:, :3, 3], axis=1)
    result = {
        "sequence": seq, "n_frames": n, "factor_counts": stats["factor_counts"],
        "graph_factor_count": stats["graph_factor_count"],
        "fixed_objective_stage1": first_objective,
        "fixed_objective_stage2": second_objective,
        "fixed_objective_relative_reduction_pct": (
            100.0 * (first_objective - second_objective)
            / max(first_objective, 1e-12)),
        "fixed_ATE3D_stage1": s7.evaluate(seq, first_poses[:, :3, 3])["ATE3D"],
        "fixed_ATE3D_stage2": s7.evaluate(seq, second_poses[:, :3, 3])["ATE3D"],
        "stage1_to_stage2_mean_translation_shift_m": float(np.mean(shift)),
        "stage1_to_stage2_max_translation_shift_m": float(np.max(shift)),
        "optimizer_configuration": "GTSAM Levenberg-Marquardt; maxIterations=100; relativeErrorTol=1e-7",
        "third_pass_requested": third_pass,
        "interpretation_warning": (
            "Change in ATE does not itself establish optimizer convergence; "
            "check objective reduction and repeated optimizer passes.")
    }

    if third_pass:
        start = time.perf_counter()
        third = optimize_graph(graph, values_from_poses(second_poses))
        elapsed = time.perf_counter() - start
        third_poses = export_poses(third, n)
        third_objective = float(graph.error(third))
        result.update({
            "fixed_objective_stage3": third_objective,
            "fixed_ATE3D_stage3": s7.evaluate(seq, third_poses[:, :3, 3])["ATE3D"],
            "third_pass_solver_seconds": elapsed,
            "stage2_to_stage3_objective_relative_reduction_pct":
                100.0 * (second_objective - third_objective)
                / max(second_objective, 1e-12),
        })
        dest = out / "factor_graph" / "ISJ_DIAG_FIXED_THIRD_PASS"
        dest.mkdir(parents=True, exist_ok=True)
        np.save(dest / "poses.npy", third_poses)
        np.savetxt(dest / "trajectory.txt", third_poses[:, :3, 3], fmt="%.9f")

    json_path = out / "isj_fixed_convergence_diagnostic.json"
    json_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSEQUENCE: {seq}")
    print(f"  Fixed objective: {first_objective:.9f} -> {second_objective:.9f}")
    print(f"  ATE3D (m):       {result['fixed_ATE3D_stage1']:.6f} -> "
          f"{result['fixed_ATE3D_stage2']:.6f}")
    print(f"  Translation displacement mean/max (m): "
          f"{np.mean(shift):.6f}/{np.max(shift):.6f}")
    if third_pass:
        print(f"  Third objective: {result['fixed_objective_stage3']:.9f}")
        print(f"  Third ATE3D (m): {result['fixed_ATE3D_stage3']:.6f}")
    print(f"  Saved: {json_path}")
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sequence", choices=("all",) + TEST_SEQUENCES, default="all")
    p.add_argument("--third-pass", action="store_true",
                   help="Run a third pass from saved second-pass poses.")
    args = p.parse_args()
    seqs = TEST_SEQUENCES if args.sequence == "all" else (args.sequence,)
    for seq in seqs:
        run(seq, args.third_pass)


if __name__ == "__main__":
    main()
