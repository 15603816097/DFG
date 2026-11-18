from __future__ import annotations

"""Frozen, isolated cross-date controls for the *time ordering* of uncertainty.

For each frozen GRU/V5 or Mamba/V6 predictor, compare the original dynamic-U
factor graph with:
  - sensor-specific constant mean uncertainty on frames t >= 63;
  - sensor-specific shuffled uncertainty on t >= 63 (seeds 11, 23, 47).

Retains every reliability r[t], the original first 63 uncertainty values,
frozen sensor measurements, graph construction, and the SAME saved Fixed-FGO
warm start. The permutation preserves each sensor's marginal U distribution.

This is an exploratory post-hoc synthetic-degradation experiment. OXTS/GPS
evaluation reference is correlated with the degraded GPS factor input.
Never edits existing GRU/Mamba runs or historical cross-sequence outputs.
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.isj_supplement.run_cross_date import (
    RESULT_ROOT,
    MODELS,
    SENSORS,
    crossdate_bindings,
    date_for,
    prediction_preflight,
    require_processed,
    result_is_complete,
    validate_destination,
    write_protocol,
)
from experiments.isj_supplement.run_controls import transform_uncertainty

DEFAULT_SEEDS = (11, 23, 47)
WINDOW = 64


def parse_seeds(raw: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError("Seeds must be comma-separated integers") from exc
    if not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError("Seeds must be distinct nonnegative integers")
    return seeds


def control_specs(models: tuple[str, ...], modes: str, seeds: tuple[int, ...]) -> list[dict]:
    specs = []
    for model in models:
        if model not in MODELS:
            raise ValueError(f"Unknown model: {model}")
        src = MODELS[model]["source"]
        prefix = "GRU" if model == "gru" else "MAMBA"
        if modes in ("all", "constant_mean"):
            specs.append({
                "name": f"ISJ_CTRL_{prefix}_MEAN_U",
                "type": "uncertainty",
                "source": src,
                "mode": "constant_mean",
                "seed": None,
            })
        if modes in ("all", "permuted"):
            for seed in seeds:
                specs.append({
                    "name": f"ISJ_CTRL_{prefix}_PERM_U_SEED_{seed}",
                    "type": "uncertainty",
                    "source": src,
                    "mode": "permuted",
                    "seed": seed,
                })
    if not specs or len({row["name"] for row in specs}) != len(specs):
        raise ValueError("Empty or duplicate control cases")
    return specs


def preflight(out: Path, models: tuple[str, ...]) -> tuple[int, np.ndarray, float]:
    n = require_processed(out)
    prediction_preflight(out, models)
    if n <= WINDOW:
        raise ValueError(f"Need more than {WINDOW} frames; got {n}")
    if not result_is_complete(out, "FINAL_S0_fixed", n):
        raise FileNotFoundError("Saved FINAL_S0_fixed missing; run cross-date --stage fgo first")
    poses = np.load(out / "factor_graph" / "FINAL_S0_fixed" / "poses.npy",
                    allow_pickle=False)
    fixed_metrics = json.loads(
        (out / "factor_graph" / "FINAL_S0_fixed" / "metrics.json").read_text()
    )
    return n, poses, float(fixed_metrics["ATE3D"])


def controlled_prediction(original_pred, spec: dict):
    """Preserve r identically; transform only U[63:] with existing Stage-7 logic."""
    def wrapped(seq, source, sensor, n):
        if source != spec["source"] or sensor not in SENSORS:
            raise ValueError(f"Unexpected control source/sensor: {source}/{sensor}")
        r, u = original_pred(seq, source, sensor, n)
        if len(u) != n:
            raise ValueError("Uncertainty length does not match sequence")
        u_control = transform_uncertainty(
            u, spec["mode"], spec["seed"] or 0, source, sensor, window=WINDOW
        )
        if not np.array_equal(u_control[:WINDOW - 1], u[:WINDOW - 1]):
            raise AssertionError("Warmup uncertainty must remain untouched")
        return r, u_control
    return wrapped


def run_fgo(seq: str, out: Path, models: tuple[str, ...],
            specs: list[dict]) -> None:
    import experiments.cross_sequence.run_stage7_factor_graph as s7

    n, fixed_poses, fixed_ate = preflight(out, models)
    original_pred = s7.pred

    def warm_start(gtsam, measurements, initial_pose):
        if len(measurements.gps_local) != n:
            raise ValueError("Frozen fixed trajectory length mismatch")
        return [s7._pose3_from_matrix(gtsam, pose) for pose in fixed_poses]

    rows = []
    with patch.object(s7, "initial_trajectory_from_pose", warm_start):
        for spec in specs:
            name = spec["name"]
            folder = out / "factor_graph" / name
            if result_is_complete(out, name, n):
                status = "cached"
            else:
                if folder.exists():
                    raise RuntimeError(
                        f"Refusing to overwrite incomplete or unexpected folder: {folder}"
                    )
                with patch.object(s7, "pred",
                                  controlled_prediction(original_pred, spec)):
                    s7.run_case(
                        seq=seq, name=name,
                        gps_src=spec["source"],
                        imu_src=spec["source"],
                        cam_src=spec["source"],
                    )
                status = "computed"

            metrics = json.loads((folder / "metrics.json").read_text())
            ate = float(metrics["ATE3D"])
            if not math.isfinite(ate):
                raise RuntimeError(f"Invalid ATE for {name}")
            rows.append({
                "sequence": seq,
                "case": name,
                "source": spec["source"],
                "control": spec["mode"],
                "seed": spec["seed"] if spec["seed"] is not None else "",
                "ATE3D_m": ate,
                "vs_fixed_pct": 100 * (fixed_ate - ate) / fixed_ate,
                "status": status,
            })
            print(f"{name:<43} ATE3D={ate:.6f} m ({status})", flush=True)

    path = out / "isj_cross_date_controls_fgo.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    write_protocol(out, seq, {
        "cross_date_uncertainty_controls": {
            "status": "fgo_complete",
            "method": "constant mean and sensor-wise temporal permutation of frozen U",
            "models": list(models),
            "warmup_uncertainty_frames_untouched": WINDOW - 1,
            "reliability_modified": False,
            "checkpoint_training": False,
            "fixed_warm_start": "FINAL_S0_fixed",
            "seeds": sorted(set(spec["seed"] for spec in specs if spec["seed"] is not None)),
            "classification": "exploratory_post_hoc",
        }
    })
    print("Saved:", path)


def cached_continuation(out: Path, spec: dict, n: int, max_extra_passes: int):
    dest = out / "factor_graph" / ("ISJ_CONVERGENCE_" + spec["name"])
    if not dest.exists():
        return None
    info_file = dest / "metrics.json"
    poses_file = dest / "poses.npy"
    if not info_file.is_file() or not poses_file.is_file():
        raise RuntimeError(f"Incomplete convergence folder; refusing overwrite: {dest}")
    poses = np.load(poses_file, allow_pickle=False)
    if poses.shape != (n, 4, 4) or not np.isfinite(poses).all():
        raise RuntimeError(f"Invalid cached convergence poses: {poses_file}")
    info = json.loads(info_file.read_text())
    for key, expected in (
        ("case", spec["name"]),
        ("method", spec),
        ("max_extra_passes", max_extra_passes),
        ("outer_relative_objective_tol", 1e-7),
        ("outer_position_shift_RMSE_tol_m", 1e-3),
    ):
        if info.get(key) != expected:
            raise ValueError(f"Cached convergence protocol mismatch for {spec['name']}: {key}")
    return {
        "sequence": info["sequence"],
        "case": info["case"],
        "type": spec["type"],
        "source_ATE3D": float(info["source_ATE3D"]),
        "continued_ATE3D": float(info["final_metrics"]["ATE3D"]),
        "source_objective": float(info["history"][0]["objective"]),
        "continued_objective": float(info["history"][-1]["objective"]),
        "extra_LM_passes": int(info["extra_LM_passes"]),
        "stationary_under_rule": bool(info["stationary_under_rule"]),
        "warning": info.get("warning") or "",
    }


def run_convergence(seq: str, out: Path, models: tuple[str, ...],
                    specs: list[dict], max_extra_passes: int) -> None:
    import gtsam
    import experiments.cross_sequence.run_stage7_factor_graph as s7
    from experiments.isj_supplement.run_uniform_convergence import refine_case

    n, _, _ = preflight(out, models)
    for spec in specs:
        if not result_is_complete(out, spec["name"], n):
            raise FileNotFoundError(
                f"{spec['name']} FGO missing; run --stage fgo first"
            )
    m = s7.load_measurements(seq)
    rows = []
    for spec in specs:
        row = cached_continuation(out, spec, n, max_extra_passes)
        if row is None:
            print(f"\n{seq} | {spec['name']}", flush=True)
            row = refine_case(gtsam, seq, m, spec, max_extra_passes, 1e-7, 1e-3)
        else:
            print(f"{spec['name']}: convergence cached", flush=True)
        rows.append(row)
        stable = "STABLE" if row["stationary_under_rule"] else "NOT_STABLE"
        print(f"  {spec['name']:<40} "
              f"{row['source_ATE3D']:.6f} -> {row['continued_ATE3D']:.6f} m "
              f"{stable}", flush=True)

    path = out / "isj_cross_date_controls_convergence.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_protocol(out, seq, {
        "cross_date_uncertainty_controls_convergence": {
            "max_extra_passes": max_extra_passes,
            "relative_objective_tol": 1e-7,
            "translation_shift_rmse_tol_m": 1e-3,
            "all_stationary_under_rule": all(row["stationary_under_rule"] for row in rows),
            "classification": "exploratory_post_hoc",
        }
    })
    print("Saved:", path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sequence", default="2011_09_30_drive_0016_sync")
    p.add_argument("--stage", required=True, choices=("fgo", "converge"))
    p.add_argument("--model", choices=("gru", "mamba", "both"), default="both")
    p.add_argument("--controls", choices=("all", "constant_mean", "permuted"),
                   default="all")
    p.add_argument("--seeds", default="11,23,47")
    p.add_argument("--max-extra-passes", type=int, default=4)
    args = p.parse_args()

    date_for(args.sequence)
    if args.max_extra_passes < 1:
        p.error("--max-extra-passes must be positive")
    try:
        seeds = parse_seeds(args.seeds)
    except ValueError as exc:
        p.error(str(exc))
    models = ("gru", "mamba") if args.model == "both" else (args.model,)
    specs = control_specs(models, args.controls, seeds)
    out = validate_destination(args.sequence, RESULT_ROOT)
    with crossdate_bindings(args.sequence, RESULT_ROOT, args.stage):
        if args.stage == "fgo":
            run_fgo(args.sequence, out, models, specs)
        else:
            run_convergence(args.sequence, out, models, specs,
                            args.max_extra_passes)


if __name__ == "__main__":
    main()
