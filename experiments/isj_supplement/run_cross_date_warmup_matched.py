from __future__ import annotations

"""Frozen cross-date Reliability Only comparison with matched first-63 frames.

Original Dynamic U and Reliability Only had DIFFERENT weight schedules for
t<63: U=1 for Dynamic U => fixed sigma; Reliability Only forced confidence=1
=> reliability-based predictive sigma, even without a full 64-frame encoder
window. This script adds an independent, *warmup-matched* Reliability Only
control:
    t<63: U_eff=1 -> fixed factor sigmas, exactly like original Dynamic U
    t>=63: U_eff=0 -> reliability-only predictive sigmas
This U_eff is a CONTROL SWITCH, not a new Beta evidential prediction. It is
deliberately allowed to equal zero only for the reliability-only control.

All checkpoints, raw features, learned reliability r[t], frozen predictions,
fixed warm start, factor equations, and convergence rules remain unchanged.
Never overwrites an existing result directory or historical 2011_10_03 data.
Exploratory pilot; OXTS/GPS reference not independent of input degraded GPS.
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
    RESULT_ROOT, MODELS, SENSORS, crossdate_bindings, date_for,
    validate_destination, result_is_complete,
)
from experiments.isj_supplement.run_cross_date_controls import (
    WINDOW, preflight, cached_continuation,
)
from experiments.isj_supplement.run_cross_date_beta_swap import (
    write_new_summary, save_protocol,
)


def specs_for_models(models: tuple[str, ...]) -> list[dict]:
    specs = []
    for model in models:
        if model not in MODELS:
            raise ValueError(f"Invalid encoder: {model}")
        name = ("GRU" if model == "gru" else "MAMBA")
        specs.append({
            "name": f"ISJ_CTRL_{name}_WARMUP_MATCHED_REL_ONLY",
            "type": "dynamic",
            "source": MODELS[model]["source"],
            "control": "fixed_warmup_then_reliability_only",
        })
    return specs


def warmup_matched_prediction(original_pred, target_source: str):
    def wrapped(seq: str, source: str, sensor: str, n: int):
        if source != target_source or sensor not in SENSORS:
            raise ValueError(f"Unexpected prediction request: {source}/{sensor}")
        r, u = original_pred(seq, source, sensor, n)
        r = np.asarray(r, dtype=np.float64).reshape(-1)
        u = np.asarray(u, dtype=np.float64).reshape(-1)
        if len(r) != n or len(u) != n or n <= WINDOW:
            raise ValueError("Frozen prediction / window length mismatch")
        if not np.isfinite(r).all() or not np.isfinite(u).all():
            raise ValueError("Frozen prediction has nonfinite values")
        # Existing frozen encoders both use U=1 before t=63. If that
        # assumption stops being true, the method must not be labeled
        # as warmup matched.
        if not np.allclose(u[:WINDOW - 1], 1.0, atol=1e-9, rtol=0):
            raise ValueError(
                f"{source}/{sensor}: frozen warmup U is not 1.0"
            )
        new_u = np.zeros_like(u)
        new_u[:WINDOW - 1] = u[:WINDOW - 1]
        assert np.array_equal(new_u[:WINDOW - 1], u[:WINDOW - 1])
        assert np.all(new_u[WINDOW - 1:] == 0.0)
        return r, new_u
    return wrapped


def preflight_audit(seq: str, specs: list[dict]) -> None:
    import experiments.cross_sequence.run_stage7_factor_graph as s7
    from experiments.isj_supplement.run_cross_date import require_processed

    n = require_processed(RESULT_ROOT / seq)
    for spec in specs:
        wrapped = warmup_matched_prediction(s7.pred, spec["source"])
        for sensor in SENSORS:
            r, u = s7.pred(seq, spec["source"], sensor, n)
            new_r, new_u = wrapped(seq, spec["source"], sensor, n)
            assert np.array_equal(new_r, r)
            assert np.array_equal(new_u[:WINDOW-1], u[:WINDOW-1])
            print(
                f'{spec["name"]:<42} {sensor:<7} '
                f'fixed_warmup={WINDOW-1} valid_rel_only={n-WINDOW+1} '
                f'r_unchanged=True',
                flush=True
            )
    print("WARMUP MATCH AUDIT PASS")


def run_fgo(seq: str, out: Path, models: tuple[str, ...],
            specs: list[dict]) -> None:
    import experiments.cross_sequence.run_stage7_factor_graph as s7

    n, fixed_poses, fixed_ate = preflight(out, models)
    original_pred = s7.pred

    def fixed_start(gtsam, m, initial_pose):
        if len(m.gps_local) != n:
            raise ValueError("Fixed FGO warm start length mismatch")
        return [s7._pose3_from_matrix(gtsam, pose) for pose in fixed_poses]

    rows = []
    with patch.object(s7, "initial_trajectory_from_pose", fixed_start):
        for spec in specs:
            folder = out / "factor_graph" / spec["name"]
            if result_is_complete(out, spec["name"], n):
                status = "cached"
            else:
                if folder.exists():
                    raise RuntimeError(
                        f"Refusing to replace incomplete FGO folder: {folder}"
                    )
                with patch.object(
                    s7, "pred",
                    warmup_matched_prediction(original_pred, spec["source"])
                ):
                    s7.run_case(
                        seq=seq, name=spec["name"],
                        gps_src=spec["source"],
                        imu_src=spec["source"],
                        cam_src=spec["source"],
                    )
                status = "computed"

            metrics = json.loads((folder / "metrics.json").read_text())
            ate = float(metrics["ATE3D"])
            if not math.isfinite(ate):
                raise ValueError(f"{spec['name']}: nonfinite ATE")
            rows.append({
                "sequence": seq,
                "case": spec["name"],
                "source": spec["source"],
                "ATE3D_m": ate,
                "vs_fixed_pct": 100.0 * (fixed_ate - ate) / fixed_ate,
            })
            print(f'{spec["name"]:<44} ATE3D={ate:.6f} m ({status})')
    label = "_".join(models)
    write_new_summary(out / f"isj_cross_date_warmup_matched_fgo_{label}.csv", rows)
    save_protocol(
        out / f"isj_cross_date_warmup_matched_fgo_protocol_{label}.json",
        {
            "sequence": seq,
            "model_set": list(models),
            "frozen_warmup_start": 0,
            "frozen_warmup_end_inclusive": WINDOW - 2,
            "valid_prediction_start_inclusive": WINDOW - 1,
            "warmup": "U_eff=1, exactly the original dynamic-U fixed sigma",
            "valid_frames": "U_eff=0; equivalently confidence=1 for r-only",
            "U_eff_is_not_beta_prediction": True,
            "reliability_unchanged": True,
            "warm_start": "FINAL_S0_fixed",
            "controls_are_exploratory": True,
            "reference_not_independent_of_gps": True,
        },
    )


def run_converge(seq: str, out: Path, models: tuple[str, ...],
                 specs: list[dict], max_extra_passes: int) -> None:
    import gtsam
    import experiments.cross_sequence.run_stage7_factor_graph as s7
    from experiments.isj_supplement.run_uniform_convergence import refine_case

    n, _, _ = preflight(out, models)
    for spec in specs:
        if not result_is_complete(out, spec["name"], n):
            raise FileNotFoundError(
                f"{spec['name']}: FGO missing; run --stage fgo"
            )

    original_pred = s7.pred
    measurements = s7.load_measurements(seq)
    rows = []
    for spec in specs:
        row = cached_continuation(out, spec, n, max_extra_passes)
        if row is None:
            folder = out / "factor_graph" / ("ISJ_CONVERGENCE_" + spec["name"])
            if folder.exists():
                raise RuntimeError(
                    f"Refusing to overwrite convergence folder: {folder}"
                )
            with patch.object(
                s7, "pred",
                warmup_matched_prediction(original_pred, spec["source"])
            ):
                row = refine_case(
                    gtsam, seq, measurements, spec, max_extra_passes,
                    1e-7, 1e-3
                )
        rows.append(row)
        print(
            f'{spec["name"]:<44} '
            f'{row["source_ATE3D"]:.6f} -> {row["continued_ATE3D"]:.6f}m '
            f'{"STABLE" if row["stationary_under_rule"] else "NOT_STABLE"}'
        )

    label = "_".join(models)
    write_new_summary(
        out / f"isj_cross_date_warmup_matched_convergence_{label}.csv", rows
    )
    save_protocol(
        out / f"isj_cross_date_warmup_matched_convergence_protocol_{label}.json",
        {
            "sequence": seq,
            "model_set": list(models),
            "max_extra_passes": max_extra_passes,
            "objective_rtol": 1e-7,
            "translation_shift_rmse_tol_m": 0.001,
            "all_stable": all(row["stationary_under_rule"] for row in rows),
            "controls_are_exploratory": True,
        },
    )
    print("\n==== Matched-warmup comparison (converged ATE3D) ====")
    for model, spec in zip(models, specs):
        original = MODELS[model]["full"]
        dyn_file = (out / "factor_graph" /
                    ("ISJ_CONVERGENCE_" + original) / "metrics.json")
        if not dyn_file.is_file():
            print(f"{model}: original Dynamic U convergence metrics unavailable")
            continue
        dyn = float(json.loads(dyn_file.read_text())["final_metrics"]["ATE3D"])
        matched = float(next(
            item["continued_ATE3D"] for item in rows
            if item["case"] == spec["name"]
        ))
        benefit = 100.0 * (matched - dyn) / matched
        print(
            f"{model.upper():6} Dynamic U={dyn:.6f} m; "
            f"matched Rel Only={matched:.6f} m; "
            f"Dynamic U improvement={benefit:+.3f}%"
        )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sequence", default="2011_09_30_drive_0016_sync")
    p.add_argument("--stage", required=True,
                   choices=("audit", "fgo", "converge"))
    p.add_argument("--model", choices=("gru", "mamba", "both"),
                   default="both")
    p.add_argument("--max-extra-passes", type=int, default=4)
    args = p.parse_args()

    date_for(args.sequence)
    if args.max_extra_passes <= 0:
        p.error("max-extra-passes must be positive")

    models = ("gru", "mamba") if args.model == "both" else (args.model,)
    specs = specs_for_models(models)
    out = validate_destination(args.sequence, RESULT_ROOT)
    with crossdate_bindings(
        args.sequence, RESULT_ROOT,
        "converge" if args.stage == "converge" else "fgo"
    ):
        if args.stage == "audit":
            preflight_audit(args.sequence, specs)
        elif args.stage == "fgo":
            run_fgo(args.sequence, out, models, specs)
        else:
            run_converge(
                args.sequence, out, models, specs, args.max_extra_passes
            )


if __name__ == "__main__":
    main()
