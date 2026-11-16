from __future__ import annotations

"""Uniform continuation-to-tolerance audit of Stage-7 saved FGO solutions.

All methods are loaded from their original, untouched Stage-7 second-stage
outputs. Rebuilds their OWN fixed/noise-adjusted graph and continues unchanged
GTSAM LM optimization under the same numeric stopping rule.

- Fixed2: ISJ_CTRL_FIXED_SECOND_PASS
- Dynamic: FINAL_S1_v5_gru, FINAL_S2_v6_all_mamba
- Reliability-only: ABL_V5_GRU_RELIABILITY_ONLY / ABL_V6_MAMBA_RELIABILITY_ONLY
- Constant/permuted uncertainty: all 8 ISJ_CTRL_* controls

No KITTI Raw, Mamba installation or model retraining needed. Every continuation
starts from the already saved second-stage solution (thus NOT a fresh 2-stage
run). Original result folders and experiment CSVs are never modified.
Exploratory post-hoc diagnostic; convergence to a local stationary result
does not prove global optimality.
"""

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.cross_sequence.run_stage7_factor_graph as s7
from experiments.cross_sequence.stage7_config import TEST_SEQUENCES, sequence_out
from experiments.isj_supplement.audit_fixed_convergence import graph_fixed
from experiments.isj_supplement.audit_dynamic_convergence import (
    build_dynamic_graph,
    check_saved_pose,
    poses_from_values,
    values_from_poses,
)
from experiments.isj_supplement.run_controls import transform_uncertainty

DEFAULT_SEEDS = (11, 23, 47)


def available_cases(family: str) -> list[dict]:
    cases = [
        {"name": "ISJ_CTRL_FIXED_SECOND_PASS", "type": "fixed"},
        {"name": "FINAL_S1_v5_gru", "type": "dynamic", "source": "v5"},
        {"name": "FINAL_S2_v6_all_mamba", "type": "dynamic", "source": "v6"},
    ]
    if family == "core":
        return cases
    reliability = [
        {"name": "ABL_V5_GRU_RELIABILITY_ONLY",
         "type": "reliability", "source": "v5"},
        {"name": "ABL_V6_MAMBA_RELIABILITY_ONLY",
         "type": "reliability", "source": "v6"},
    ]
    if family == "reliability":
        return reliability
    if family not in ("controls", "all", "ablations"):
        raise ValueError(f"Unknown family {family}")
    controls = []
    for model, source in (("GRU", "v5"), ("MAMBA", "v6")):
        controls.append({
            "name": f"ISJ_CTRL_{model}_MEAN_U",
            "type": "uncertainty",
            "source": source,
            "mode": "constant_mean",
            "seed": None,
        })
        for seed in DEFAULT_SEEDS:
            controls.append({
                "name": f"ISJ_CTRL_{model}_PERM_U_SEED_{seed}",
                "type": "uncertainty",
                "source": source,
                "mode": "permuted",
                "seed": seed,
            })
    if family == "controls":
        return controls
    if family == "ablations":
        return reliability + controls
    return cases + reliability + controls


def same_graph(gtsam, seq: str, m, spec: dict):
    kind = spec["type"]
    if kind == "fixed":
        graph, counts = graph_fixed(gtsam, m)
        return graph, counts
    if kind == "dynamic":
        return build_dynamic_graph(gtsam, seq, m, spec["source"]), None
    if kind == "reliability":
        original_confidence = s7.confidence
        try:
            s7.confidence = lambda uncertainty: 1.0
            graph = build_dynamic_graph(gtsam, seq, m, spec["source"])
        finally:
            s7.confidence = original_confidence
        return graph, None

    # Exactly matches the frozen transformations in run_controls.py. The
    # reliability time-series remains untouched. The patch is always undone.
    original_pred = s7.pred

    def controlled_pred(seq_, source_, sensor, n):
        r, u = original_pred(seq_, source_, sensor, n)
        return r, transform_uncertainty(
            u, spec["mode"], spec["seed"] or 0, source_, sensor
        )

    try:
        s7.pred = controlled_pred
        graph = build_dynamic_graph(gtsam, seq, m, spec["source"])
    finally:
        s7.pred = original_pred
    return graph, None


def refine_case(gtsam, seq, m, spec, max_passes, objective_rtol, shift_tol):
    out = sequence_out(seq)
    name = spec["name"]
    source_dir = out / "factor_graph" / name
    original_poses = check_saved_pose(
        source_dir / "poses.npy", len(m.gps_local)
    )

    graph, factors = same_graph(gtsam, seq, m, spec)
    current = values_from_poses(gtsam, original_poses)
    current_poses = original_poses
    cost = float(graph.error(current))
    original_metrics = s7.evaluate(seq, original_poses[:, :3, 3])
    if not math.isfinite(cost):
        raise ValueError(f"{name}: nonfinite initial cost")

    original_file = source_dir / "metrics.json"
    if not original_file.is_file():
        raise FileNotFoundError(original_file)
    saved_metrics = json.loads(original_file.read_text(encoding="utf-8"))
    if "metrics" in saved_metrics:
        saved_metrics = saved_metrics["metrics"]
    saved_ate = float(saved_metrics["ATE3D"])
    if abs(saved_ate - original_metrics["ATE3D"]) > 1e-4:
        raise ValueError(f"{name}: source poses/metrics mismatch. "
                         "Check original experimental file versions.")

    history = [{
        "extra_pass": 0,
        "objective": cost,
        "ATE3D": original_metrics["ATE3D"],
        "solver_seconds": 0,
        "relative_objective_drop": None,
        "position_shift_rmse_m": None,
    }]
    stationary = False
    warning = None
    for extra in range(1, max_passes + 1):
        params = gtsam.LevenbergMarquardtParams()
        params.setMaxIterations(100)
        params.setRelativeErrorTol(1e-7)
        started = time.perf_counter()
        result = gtsam.LevenbergMarquardtOptimizer(
            graph, current, params
        ).optimize()
        seconds = time.perf_counter() - started
        next_cost = float(graph.error(result))
        if not math.isfinite(next_cost):
            raise ValueError(f"{name}: nonfinite objective after pass {extra}")
        # Do not accept an objectively worse solution as an improvement.
        if next_cost > cost + max(1e-7, abs(cost) * 1e-10):
            warning = (f"Objective increased on continuation pass {extra}: "
                       f"{cost:.9f} -> {next_cost:.9f}")
            break

        next_poses = poses_from_values(result, len(m.gps_local))
        delta = next_poses[:, :3, 3] - current_poses[:, :3, 3]
        shift = float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))
        relative_drop = max(0.0, (cost - next_cost) / max(abs(cost), 1.0))
        next_ate = float(s7.evaluate(seq, next_poses[:, :3, 3])["ATE3D"])
        history.append({
            "extra_pass": extra,
            "objective": next_cost,
            "ATE3D": next_ate,
            "solver_seconds": seconds,
            "relative_objective_drop": relative_drop,
            "position_shift_rmse_m": shift,
        })
        print(f"    pass={extra}: objective={next_cost:.9f}, "
              f"ATE3D={next_ate:.6f}m, "
              f"rel_drop={relative_drop:.3g}, shift_RMSE={shift:.6f}m",
              flush=True)

        current, current_poses, cost = result, next_poses, next_cost
        if relative_drop <= objective_rtol and shift <= shift_tol:
            stationary = True
            break

    final_metrics = s7.evaluate(seq, current_poses[:, :3, 3])
    destination = out / "factor_graph" / ("ISJ_CONVERGENCE_" + name)
    destination.mkdir(parents=True, exist_ok=True)
    np.save(destination / "poses.npy", current_poses)
    np.savetxt(destination / "trajectory.txt",
               current_poses[:, :3, 3], fmt="%.9f")

    info = {
        "sequence": seq,
        "case": name,
        "method": spec,
        "protocol": "post_hoc_uniform_stage2_continuation_to_tolerance",
        "source_second_stage_directory": str(source_dir),
        "extra_LM_passes": len(history) - 1,
        "per_pass_max_iterations": 100,
        "per_pass_relative_error_tol": 1e-7,
        "outer_relative_objective_tol": objective_rtol,
        "outer_position_shift_RMSE_tol_m": shift_tol,
        "max_extra_passes": max_passes,
        "stationary_under_rule": stationary,
        "warning": warning,
        "graph_factor_count": int(graph.size()),
        "factor_counts": factors,
        "source_ATE3D": original_metrics["ATE3D"],
        "final_metrics": final_metrics,
        "history": history,
        "notes": [
            "Each result is a continuation of its existing Stage-7 2nd pass.",
            "Every model uses its own factor covariance graph; graph objective values across methods are not directly comparable.",
            "This is a numerical stopping test, not proof of global optimum.",
            "Never re-label post-hoc exploratory results as prospectively held out.",
            "OXTS/GPS-derived evaluation reference is not independent of GPS inputs.",
        ],
    }
    (destination / "metrics.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {
        "sequence": seq,
        "case": name,
        "type": spec["type"],
        "source_ATE3D": original_metrics["ATE3D"],
        "continued_ATE3D": float(final_metrics["ATE3D"]),
        "source_objective": float(history[0]["objective"]),
        "continued_objective": float(history[-1]["objective"]),
        "extra_LM_passes": len(history) - 1,
        "stationary_under_rule": stationary,
        "warning": warning or "",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", default="all",
                        choices=("all",) + TEST_SEQUENCES)
    parser.add_argument("--family", default="core",
                        choices=("core", "controls", "reliability", "ablations", "all"))
    parser.add_argument("--max-extra-passes", type=int, default=4)
    parser.add_argument("--objective-rtol", type=float, default=1e-7)
    parser.add_argument("--shift-tol", type=float, default=1e-3,
                        help="RMSE of translation movement between passes, meters")
    args = parser.parse_args()
    if args.max_extra_passes < 1:
        parser.error("--max-extra-passes must be >=1")
    if args.objective_rtol <= 0 or args.shift_tol <= 0:
        parser.error("Tolerances must be positive")

    import gtsam

    sequences = TEST_SEQUENCES if args.sequence == "all" else (args.sequence,)
    cases = available_cases(args.family)
    for seq in sequences:
        m = s7.load_measurements(seq)
        results = []
        for spec in cases:
            print(f"\n{seq} | {spec['name']}", flush=True)
            results.append(refine_case(
                gtsam, seq, m, spec,
                args.max_extra_passes, args.objective_rtol, args.shift_tol
            ))
        path = sequence_out(seq) / f"isj_convergence_{args.family}_results.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)
        print(f"\nSequence done: {seq} | Saved {path}")
        for r in results:
            status = "stable" if r["stationary_under_rule"] else "NOT_STABLE"
            print(f"  {r['case']:<43s} "
                  f"{r['source_ATE3D']:.6f} -> {r['continued_ATE3D']:.6f} m "
                  f"(extra={r['extra_LM_passes']}, {status})")


if __name__ == "__main__":
    main()
