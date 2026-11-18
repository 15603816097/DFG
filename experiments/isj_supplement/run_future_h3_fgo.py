from __future__ import annotations

"""Frozen H=3 anticipatory reliability vs persistence, with equal information.

At source time t (first valid t=63), both methods only see X[:t+1].
  persistence assigned to factor target j=t+3: head0[t]
  predictive  assigned to factor target j=t+3: head1[t]
Training heads: head0=current[t]; head1=current[t+3]; head2=current[t+2].
First 66 target frames (j<66) use exactly FIXED sigmas for both.
For j>=66 both use U_eff=0, so only predictive r is ablated;
U_eff is a deterministic covariance-control switch, NOT Beta uncertainty.
Uses SAME 0016 saved degraded measurements, graph, Fixed-FGO warm-start,
frozen checkpoint normalizers, solver, and convergence rule.
This is an already-explored synthetic-degradation pilot, NOT untouched test.
Do not use its ATE to tune a model and claim held-out generalization.
Never overwrites original V5/V6 predictions or FGO outputs.
"""

import argparse
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
    validate_destination, result_is_complete, require_processed,
)
from experiments.isj_supplement.run_cross_date_controls import (
    preflight, cached_continuation,
)
from experiments.isj_supplement.run_cross_date_beta_swap import (
    write_new_summary, save_protocol,
)
from experiments.isj_supplement.audit_future_prediction_vs_persistence import (
    WINDOW, LEAD, source_indices, load_model, predict, check_current_cache,
)

FIRST = WINDOW - 1 + LEAD  # target frame 66


def specs_for(models):
    specs = []
    for model in models:
        prefix = "GRU" if model == "gru" else "MAMBA"
        for mode in ("PERSIST", "PREDICT"):
            specs.append({
                "name": f"ISJ_H3_{prefix}_{mode}_R_ONLY",
                "type": "dynamic",
                "source": MODELS[model]["source"],
                "model": model,
                "horizon": LEAD,
                "mode": mode.lower(),
                "warmup": "fixed_j_0_to_65",
                "covariance": "r_only_Ueff_0_from_j_66",
            })
    return specs


def forecast_file(out: Path, source: str, sensor: str) -> Path:
    return out / "forecast_h3" / source / f"{sensor}_head1_r_target_aligned.txt"


def save_frozen_forecast(seq: str, out: Path, model: str, device: str, batch_size: int):
    import torch
    source = MODELS[model]["source"]
    n = require_processed(out)
    idx = source_indices(n, seq)
    dev = torch.device(device)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA not available; do not replace official Mamba model")
    folder = out / "forecast_h3" / source
    paths = [forecast_file(out, source, s) for s in SENSORS]
    if all(path.is_file() for path in paths):
        for sensor in SENSORS:
            read_forecast(out, source, sensor, n)
        print(f"FORECAST H3 CACHED: {folder}")
        return
    if any(path.exists() for path in paths) or (folder / "protocol.json").exists():
        raise RuntimeError(f"Partial forecast cache, refusing overwrite: {folder}")
    arrays = {}
    for sensor in SENSORS:
        with np.load(out / "features.npz", allow_pickle=False) as data:
            x = np.asarray(data[f"{sensor}_features"], np.float32)
        if len(x) != n:
            raise ValueError("Frozen feature length differs from sensor data")
        net, mean, scale, input_dim = load_model(model, sensor, dev)
        if x.shape[1] != input_dim:
            raise ValueError("Feature dimension does not match original checkpoint")
        now, future = predict(
            model, net, x, idx, mean, scale, dev, batch_size,
        )
        check_current_cache(out, sensor, model, idx, now)
        if np.min(future) < 0 or np.max(future) > 1:
            raise ValueError(f"{model}/{sensor}: invalid predicted reliability")
        full = np.full(n, np.nan, dtype=np.float64)
        full[idx + LEAD] = future
        if np.any(np.isfinite(full[:FIRST])) or not np.isfinite(full[FIRST:]).all():
            raise ValueError(f"{model}/{sensor}: horizon alignment failure")
        arrays[sensor] = full
        print(
            f"{model}/{sensor}: source t={idx[0]}..{idx[-1]} -> "
            f"target j={idx[0]+LEAD}..{idx[-1]+LEAD}; "
            f"future r mean={future.mean():.6f}; "
            "frozen head0 verified"
        )

    # No file is written until all sensors pass their model/cache checks.
    folder.mkdir(parents=True, exist_ok=True)
    for sensor in SENSORS:
        with forecast_file(out, source, sensor).open("x", encoding="utf-8") as f:
            np.savetxt(f, arrays[sensor], fmt="%.9f")
    save_protocol(folder / "protocol.json", {
        "sequence": seq, "model": model, "source": source,
        "head0": "current_reliability_at_t",
        "head1": "future_reliability_at_t_plus_3",
        "target_frame_start": FIRST,
        "forecast_source_start": WINDOW - 1,
        "checkpoint_training": False,
        "normalization": "frozen_checkpoint",
        "control": "persistence_uses_head0_at_same_source_t",
        "original_prediction_files_modified": False,
        "status": "exploratory_not_independent_test",
    })
    print("Saved frozen future-head predictions:", folder)


def read_forecast(out: Path, source: str, sensor: str, n: int) -> np.ndarray:
    path = forecast_file(out, source, sensor)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path}; run --stage predict --model "
            f"{'gru' if source == 'v5' else 'mamba'} first"
        )
    y = np.loadtxt(path, dtype=np.float64).reshape(-1)
    if (len(y) != n or n <= FIRST or
            not np.isnan(y[:FIRST]).all() or
            not np.isfinite(y[FIRST:]).all() or
            np.any(y[FIRST:] < 0) or np.any(y[FIRST:] > 1)):
        raise ValueError(f"{path}: invalid H3 target alignment or reliability values")
    return y


def build_h3_arrays(current_r: np.ndarray, future_r: np.ndarray,
                    mode: str) -> tuple[np.ndarray, np.ndarray]:
    current_r = np.asarray(current_r, float).reshape(-1)
    future_r = np.asarray(future_r, float).reshape(-1)
    n = len(current_r)
    if n != len(future_r) or n <= FIRST:
        raise ValueError("H3 arrays must have equal length >66")
    if not np.isfinite(current_r).all():
        raise ValueError("Nonfinite current frozen predictions")
    if not np.isnan(future_r[:FIRST]).all() or not np.isfinite(future_r[FIRST:]).all():
        raise ValueError("Future forecast lacks a correctly masked prewindow")
    if mode not in ("persist", "predict"):
        raise ValueError(f"Invalid H3 method {mode}")
    r = np.full(n, 0.5, dtype=float)
    u_eff = np.ones(n, dtype=float)
    if mode == "persist":
        r[FIRST:] = current_r[WINDOW-1:n-LEAD]
    else:
        r[FIRST:] = future_r[FIRST:]
    u_eff[FIRST:] = 0.0
    if not np.isfinite(r).all() or np.any(r < 0) or np.any(r > 1):
        raise ValueError("H3 predictor outputs not finite or outside [0,1]")
    return r, u_eff


def patched_predictor(original_pred, out: Path, spec: dict):
    def wrapped(seq: str, source: str, sensor: str, n: int):
        if source != spec["source"] or sensor not in SENSORS:
            raise ValueError(f"Wrong FGO prediction source {source}/{sensor}")
        current, _old_u = original_pred(seq, source, sensor, n)
        future = read_forecast(out, source, sensor, n)
        return build_h3_arrays(current, future, spec["mode"])
    return wrapped


def audit(seq: str, out: Path, models, specs):
    import experiments.cross_sequence.run_stage7_factor_graph as s7
    n, _, _ = preflight(out, models)
    for spec in specs:
        wrapped = patched_predictor(s7.pred, out, spec)
        for sensor in SENSORS:
            r,u = wrapped(seq,spec["source"],sensor,n)
            if (not np.array_equal(u[:FIRST],np.ones(FIRST)) or
                    not np.array_equal(u[FIRST:],np.zeros(n-FIRST)) or
                    not np.array_equal(r[:FIRST],np.full(FIRST,.5))):
                raise AssertionError("Unmatched H3 warmup, abort")
            print(
                f"{spec['name']:34} {sensor:7} "
                f"fixed_warmup={FIRST} predictive_frames={n-FIRST} "
                f"r_mean={r[FIRST:].mean():.6f}",
                flush=True,
            )
    print("H3 PERSISTENCE/PREDICTIVE AUDIT PASS")


def run_fgo(seq: str, out: Path, models, specs):
    import experiments.cross_sequence.run_stage7_factor_graph as s7
    n, poses, fixed_ate = preflight(out, models)
    audit(seq,out,models,specs)
    original = s7.pred
    def initial(gtsam, m, initial_pose):
        if len(m.gps_local) != n:
            raise ValueError("Fixed warm start size mismatch")
        return [s7._pose3_from_matrix(gtsam,pose) for pose in poses]
    rows = []
    with patch.object(s7,"initial_trajectory_from_pose",initial):
        for spec in specs:
            name = spec["name"]
            folder = out / "factor_graph" / name
            if result_is_complete(out,name,n):
                status = "cached"
            else:
                if folder.exists():
                    raise RuntimeError(f"Refusing overwrite of incomplete FGO output {folder}")
                with patch.object(s7,"pred",patched_predictor(original,out,spec)):
                    s7.run_case(
                        seq=seq, name=name,
                        gps_src=spec["source"],
                        imu_src=spec["source"],
                        cam_src=spec["source"],
                    )
                status = "computed"
            metrics = json.loads((folder/"metrics.json").read_text())
            ate = float(metrics["ATE3D"])
            if not math.isfinite(ate):
                raise ValueError("Nonfinite FGO ATE")
            rows.append({
                "sequence":seq,"case":name,"model":spec["model"],
                "mode":spec["mode"],"ATE3D_m":ate,
                "vs_fixed_pct":100*(fixed_ate-ate)/fixed_ate,
            })
            print(f"{name:<40} ATE3D={ate:.6f}m ({status})")
    label="_".join(models)
    write_new_summary(out/f"isj_h3_fgo_{label}.csv",rows)
    save_protocol(out/f"isj_h3_fgo_protocol_{label}.json",{
        "sequence":seq,"models":list(models),"source_t_first":WINDOW-1,
        "target_j_first":FIRST,"lead_frames":LEAD,
        "both_use_only_observations_through_source_t":True,
        "persistence":"head0_t_used_for_j_t_plus_3",
        "predict":"head1_t_used_for_j_t_plus_3",
        "warmup":"Ueff1_and_fixed_sigma_j_less_66",
        "active":"Ueff0_and_r_only_j_ge_66",
        "fixed_start":"FINAL_S0_fixed","frozen_no_retraining":True,
        "exploratory_not_untouched":True,
    })
    for model in models:
        a=next(r["ATE3D_m"] for r in rows if r["model"]==model and r["mode"]=="persist")
        b=next(r["ATE3D_m"] for r in rows if r["model"]==model and r["mode"]=="predict")
        print(f"H3 {model.upper()} predictive vs persistence: "
              f"{a:.6f} -> {b:.6f}m, benefit={(a-b)/a*100:+.3f}%")


def run_converge(seq: str, out: Path, models, specs, max_extra_passes: int):
    import gtsam
    import experiments.cross_sequence.run_stage7_factor_graph as s7
    from experiments.isj_supplement.run_uniform_convergence import refine_case
    n, _, _ = preflight(out, models)
    audit(seq,out,models,specs)
    m=s7.load_measurements(seq)
    original=s7.pred
    rows=[]
    for spec in specs:
        if not result_is_complete(out,spec["name"],n):
            raise FileNotFoundError(f"{spec['name']}: run --stage fgo first")
        cached=cached_continuation(out,spec,n,max_extra_passes)
        if cached is None:
            destination=out/"factor_graph"/("ISJ_CONVERGENCE_"+spec["name"])
            if destination.exists():
                raise RuntimeError(f"Refusing overwrite incomplete convergence {destination}")
            with patch.object(s7,"pred",patched_predictor(original,out,spec)):
                row=refine_case(gtsam,seq,m,spec,max_extra_passes,1e-7,1e-3)
        else:
            row=cached
        rows.append(row)
        print(f"{spec['name']:<40} "
              f"{row['source_ATE3D']:.6f} -> {row['continued_ATE3D']:.6f}m "
              f"{'STABLE' if row['stationary_under_rule'] else 'NOT_STABLE'}")
    label="_".join(models)
    write_new_summary(out/f"isj_h3_convergence_{label}.csv",rows)
    for model in models:
        a=next(row["continued_ATE3D"] for row in rows if
               row["case"]==next(s["name"] for s in specs
                                  if s["model"]==model and s["mode"]=="persist"))
        b=next(row["continued_ATE3D"] for row in rows if
               row["case"]==next(s["name"] for s in specs
                                  if s["model"]==model and s["mode"]=="predict"))
        print(f"CONVERGED H3 {model.upper()}: persistence={a:.6f}m, "
              f"predictive={b:.6f}m, benefit={(a-b)/a*100:+.3f}%")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sequence",default="2011_09_30_drive_0016_sync")
    p.add_argument("--stage",required=True,choices=("predict","audit","fgo","converge"))
    p.add_argument("--model",choices=("gru","mamba","both"),default="gru")
    p.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    p.add_argument("--batch-size",type=int,default=128)
    p.add_argument("--max-extra-passes",type=int,default=4)
    args=p.parse_args()
    date_for(args.sequence)
    if args.batch_size < 1 or args.max_extra_passes < 1:
        p.error("batch-size and max-extra-passes must be >=1")
    models=("gru","mamba") if args.model=="both" else (args.model,)
    out=validate_destination(args.sequence,RESULT_ROOT)
    if args.stage=="predict":
        for model in models:
            save_frozen_forecast(args.sequence,out,model,args.device,args.batch_size)
        return
    with crossdate_bindings(args.sequence,RESULT_ROOT,
                            "converge" if args.stage=="converge" else "fgo"):
        specs=specs_for(models)
        if args.stage=="audit":
            audit(args.sequence,out,models,specs)
        elif args.stage=="fgo":
            run_fgo(args.sequence,out,models,specs)
        elif args.stage=="converge":
            run_converge(args.sequence,out,models,specs,args.max_extra_passes)


if __name__=="__main__":
    main()
