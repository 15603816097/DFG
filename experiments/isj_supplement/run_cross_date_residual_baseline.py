from __future__ import annotations

"""Cross-date ONE-STEP conventional residual reweighting baseline.

Reuses the exact frozen Stage-7 factor types/noise anchors and the existing
ISJ residual covariance inflation implementation. The residuals are computed
once from saved Stage-1 Fixed FGO poses; there is NO neural prediction,
NO test-reference weighting, NO new trainable parameters, and NO model tuning.

The baseline is NOT a full robust-kernel Huber/Cauchy implementation and is
NOT causal/streaming: it uses a full-trajectory first-stage fixed-FGO solution.
It is a useful traditional two-stage competitor, not independent truth.

Outputs only to results/isj_cross_date/<sequence>/factor_graph/ISJ_RESADAPT_*.
Never edits the historical cross_sequence tree or any frozen network result.
All experiments on previously inspected 0016 are EXPLORATORY.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.isj_supplement.run_cross_date import (
    RESULT_ROOT, crossdate_bindings, date_for, require_processed,
    result_is_complete, validate_destination,
)
from experiments.isj_supplement.run_cross_date_beta_swap import (
    write_new_summary, save_protocol,
)
from experiments.isj_supplement.run_cross_date_controls import (
    cached_continuation,
)
from experiments.isj_supplement.adaptive_fg_common import (
    CHI_3D_95, CHI_6D_95, MAX_SIGMA_MULTIPLIER, build_factor_graph,
    optimize_graph, export_poses,
)

CASES = {
    "gps": {
        "name": "ISJ_RESADAPT_GPS",
        "type": "adaptive",
        "sensors": ["gps"],
    },
    "all": {
        "name": "ISJ_RESADAPT_GPS_IMU_CAMERA",
        "type": "adaptive",
        "sensors": ["gps", "imu", "camera"],
    },
}


def select_specs(mode: str) -> list[dict]:
    if mode == "both":
        return [dict(CASES["gps"]), dict(CASES["all"])]
    if mode not in CASES:
        raise ValueError(f"Unknown mode: {mode}")
    return [dict(CASES[mode])]


def saved_case(out: Path, seq: str, spec: dict, n: int) -> dict | None:
    """Fail closed if any part of a named result already exists but is invalid."""
    folder = out / "factor_graph" / spec["name"]
    if not folder.exists():
        return None
    if not folder.is_dir():
        raise RuntimeError(f"Unexpected result entry {folder}")
    poses_file = folder / "poses.npy"
    metadata_file = folder / "metrics.json"
    if not poses_file.is_file() or not metadata_file.is_file():
        raise RuntimeError(f"Incomplete result {folder}; refusing overwrite")
    poses = np.load(poses_file, allow_pickle=False)
    if poses.shape != (n, 4, 4) or not np.isfinite(poses).all():
        raise ValueError(f"Invalid saved poses: {poses_file}")
    meta = json.loads(metadata_file.read_text(encoding="utf-8"))
    required = {
        "sequence": seq,
        "case": spec["name"],
        "selected_sensors": list(spec["sensors"]),
        "warm_start": "factor_graph/FINAL_S0_fixed/poses.npy",
        "protocol": "exploratory_cross_date_one_step_residual_adaptive",
    }
    for key, expected in required.items():
        if meta.get(key) != expected:
            raise ValueError(f"{folder}: metadata mismatch on {key}")
    for key in ("graph_initial_objective", "graph_final_objective"):
        val = float(meta[key])
        if not math.isfinite(val) or val < 0:
            raise ValueError(f"{folder}: invalid {key}")
    for key in ("ATE3D", "ATE2D", "ZRMSE"):
        val = float(meta["metrics"][key])
        if not math.isfinite(val) or val < 0:
            raise ValueError(f"{folder}: invalid {key}")
    return meta


def first_stage(seq: str, out: Path):
    import experiments.cross_sequence.run_stage7_factor_graph as s7

    n = require_processed(out)
    if not result_is_complete(out, "FINAL_S0_fixed", n):
        raise FileNotFoundError(
            f"{out}/factor_graph/FINAL_S0_fixed is required"
        )
    poses = np.load(
        out / "factor_graph" / "FINAL_S0_fixed" / "poses.npy",
        allow_pickle=False,
    )
    fixed_meta = json.loads(
        (out / "factor_graph" / "FINAL_S0_fixed" / "metrics.json")
        .read_text(encoding="utf-8")
    )
    m = s7.load_measurements(seq)
    if len(m.gps_local) != n:
        raise ValueError("Fixed pose count != sensor measurement count")
    return m, poses, n, float(fixed_meta["ATE3D"])


def run_fgo(seq: str, out: Path, specs: list[dict]) -> None:
    import experiments.cross_sequence.run_stage7_factor_graph as s7

    m, fixed, n, fixed_ate = first_stage(seq, out)
    rows = []
    for spec in specs:
        meta = saved_case(out, seq, spec, n)
        if meta is None:
            folder = out / "factor_graph" / spec["name"]
            started = time.perf_counter()
            graph, initial, stats = build_factor_graph(
                m, fixed, adaptive_sensors=spec["sensors"]
            )
            build_seconds = time.perf_counter() - started
            initial_obj = float(graph.error(initial))
            started = time.perf_counter()
            solved = optimize_graph(graph, initial)
            solver_seconds = time.perf_counter() - started
            final_obj = float(graph.error(solved))
            if (not np.isfinite([initial_obj, final_obj]).all()
                    or min(initial_obj, final_obj) < 0):
                raise ValueError("Nonfinite/negative graph objective")
            if final_obj > initial_obj + max(1e-7, initial_obj * 1e-10):
                raise RuntimeError("Optimization objective increased; no result written")
            poses = export_poses(solved, n)
            if poses.shape != (n, 4, 4) or not np.isfinite(poses).all():
                raise ValueError("Invalid optimized poses")
            metrics = s7.evaluate(seq, poses[:, :3, 3])
            meta = {
                "sequence": seq,
                "case": spec["name"],
                "selected_sensors": list(spec["sensors"]),
                "kind": "frozen_first_stage_residual_covariance_inflation",
                "warm_start": "factor_graph/FINAL_S0_fixed/poses.npy",
                "protocol": "exploratory_cross_date_one_step_residual_adaptive",
                "reference_used_for_weighting": False,
                "first_stage_used_full_sequence": True,
                "not_online_or_independent_reference": True,
                "network_weights_used": False,
                "sigma_multiplier_formula":
                    "clip(sqrt(max(1, norm(whitened_residual)/cutoff)), 1, 4)",
                "c_3d": CHI_3D_95,
                "c_6d": CHI_6D_95,
                "max_sigma_multiplier": MAX_SIGMA_MULTIPLIER,
                "lidar_covariance": "fixed",
                "graph_initial_objective": initial_obj,
                "graph_final_objective": final_obj,
                "graph_build_seconds": build_seconds,
                "solver_seconds": solver_seconds,
                "sigma_statistics": stats,
                "metrics": metrics,
                "limitations": [
                    "Traditional residual reweighting is not a Huber/Cauchy factor kernel.",
                    "0016 is repeatedly inspected exploratory data, not untouched test.",
                    "Fixed stage and reweighting use same sequence; no online causality.",
                    "OXTS/GPS evaluation reference not independent from GPS input.",
                ],
            }
            folder.mkdir(parents=True, exist_ok=False)
            np.save(folder / "poses.npy", poses)
            np.savetxt(folder / "trajectory.txt", poses[:, :3, 3], fmt="%.9f")
            (folder / "metrics.json").write_text(
                json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print(f"{spec['name']}: FGO computed", flush=True)
        else:
            print(f"{spec['name']}: FGO cached", flush=True)

        ate = float(meta["metrics"]["ATE3D"])
        rows.append({
            "sequence": seq,
            "case": spec["name"],
            "ATE3D_m": ate,
            "vs_fixed_pct": 100.0 * (fixed_ate - ate) / fixed_ate,
        })
        print(f"  {spec['name']:<35} ATE3D={ate:.6f} m "
              f"fixed={fixed_ate:.6f} m", flush=True)

    label = "_".join("gps" if len(spec["sensors"]) == 1 else "all"
                     for spec in specs)
    write_new_summary(
        out / f"isj_cross_date_residual_fgo_{label}.csv", rows
    )
    save_protocol(
        out / f"isj_cross_date_residual_fgo_protocol_{label}.json",
        {
            "sequence": seq,
            "methods": specs,
            "baseline": "predeclared one-step residual noise inflation",
            "warm_start": "FINAL_S0_fixed",
            "reference_for_weighting": False,
            "classification": "exploratory_post_hoc",
        },
    )


def run_converge(seq: str, out: Path, specs: list[dict],
                 max_extra_passes: int) -> None:
    import gtsam
    from experiments.isj_supplement.run_uniform_convergence import refine_case

    m, _, n, _ = first_stage(seq, out)
    for spec in specs:
        if saved_case(out, seq, spec, n) is None:
            raise FileNotFoundError(
                f"{spec['name']}: run --stage fgo first"
            )
    rows = []
    for spec in specs:
        cached = cached_continuation(out, spec, n, max_extra_passes)
        if cached is not None:
            row = cached
            print(f"{spec['name']}: convergence cached", flush=True)
        else:
            target = out / "factor_graph" / ("ISJ_CONVERGENCE_" + spec["name"])
            if target.exists():
                raise RuntimeError(
                    f"Incomplete convergence folder {target}; refusing overwrite"
                )
            row = refine_case(
                gtsam, seq, m, spec, max_extra_passes, 1e-7, 1e-3
            )
        rows.append(row)
        print(
            f"  {row['case']:<35} "
            f"{row['source_ATE3D']:.6f} -> {row['continued_ATE3D']:.6f} m "
            f"{'STABLE' if row['stationary_under_rule'] else 'NOT_STABLE'}",
            flush=True,
        )

    label = "_".join("gps" if len(spec["sensors"]) == 1 else "all"
                     for spec in specs)
    write_new_summary(
        out / f"isj_cross_date_residual_convergence_{label}.csv", rows
    )
    save_protocol(
        out / f"isj_cross_date_residual_convergence_protocol_{label}.json",
        {
            "sequence": seq,
            "methods": specs,
            "max_extra_passes": max_extra_passes,
            "objective_rtol": 1e-7,
            "translation_shift_rmse_tol_m": 0.001,
            "all_stable": all(row["stationary_under_rule"] for row in rows),
            "classification": "exploratory_post_hoc",
        },
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sequence", default="2011_09_30_drive_0016_sync")
    p.add_argument("--stage", required=True, choices=("fgo", "converge"))
    p.add_argument("--mode", choices=("gps", "all", "both"), default="both")
    p.add_argument("--max-extra-passes", type=int, default=4)
    args = p.parse_args()
    date_for(args.sequence)
    if args.max_extra_passes < 1:
        p.error("--max-extra-passes must be positive")
    specs = select_specs(args.mode)
    out = validate_destination(args.sequence, RESULT_ROOT)
    with crossdate_bindings(
        args.sequence, RESULT_ROOT,
        "converge" if args.stage == "converge" else "fgo"
    ):
        if args.stage == "fgo":
            run_fgo(args.sequence, out, specs)
        else:
            run_converge(args.sequence, out, specs, args.max_extra_passes)


if __name__ == "__main__":
    main()
