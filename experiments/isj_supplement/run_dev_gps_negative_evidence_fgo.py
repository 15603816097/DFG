from __future__ import annotations

"""Isolated 0027 development FGO pilot, reusing EXISTING Stage-6 factor graph.

This is explicitly an exploratory *development* run, not a final ISJ
experiment or a convergence-controlled cross-drive result. The pilot
does NOT read or change 0016, 0020, frozen checkpoints or any historical
Stage-6/FGO result. Uses four sensors on the same 0027 measurements.

Within each GRU or Mamba backbone the three ablations differ ONLY in
GPS reliability: unchanged r; r+negative q without U; r+negative q
gated by model Beta uncertainty U. IMU and Camera use frozen r-only
weights in all three variants; LiDAR remains fixed. The original
Stage-6 GTSAM factor graph/optimizer code is reused directly by
in-memory patching only, eliminating independent graph rewrites.
First 63 and final 2 frames use fixed noise for every model/case.

No fitting on 0020. The same 0027 was used for model training, so all
outputs here are exploratory. q is heuristic and non-independent of
LiDAR factors. Error reference is OXTS/GPS based.
"""

import argparse
from contextlib import ExitStack
import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from experiments.factor_graph import run_stage6_v6_mamba_factor_graph_ablation as s6
from src.reliability.cross_modal_consistency import (
    measurement_consistency, CrossModalSettings,
)
from src.reliability.gps_negative_evidence_gate import (
    negative_evidence_gate, predicted_mask, NegativeEvidenceSettings,
)

OUT = ROOT/"results"/"isj_dev_gps_negative_evidence_v1"
SENSORS=("gps","imu","camera")
MODELS={"gru":"v5","mamba":"v6"}
VARIANTS=("reliability_only","no_u","with_u")
SETTINGS=NegativeEvidenceSettings()
CROSS=CrossModalSettings()


def make_inputs(model):
    if model not in MODELS:
        raise ValueError("Unsupported backbone")
    m=s6.load_measurements()
    n=len(m.gps_local)
    if n<100:
        raise ValueError("Unexpected short development sequence")
    pred={}
    for sensor in SENSORS:
        r,u=s6.load_prediction(MODELS[model],sensor,n)
        pred[sensor]=(r,u)
    q=measurement_consistency(
        gps_local=m.gps_local,imu_gyro=m.imu_gyro,
        timestamps_seconds=np.asarray(m.timestamps,float),
        lidar_between=m.lidar_between,
        lidar_valid=m.lidar_valid,
        lidar_quality=m.lidar_quality,
        settings=CROSS,
    )
    active=predicted_mask(n,SETTINGS)
    cases={}
    for v in VARIANTS:
        rstar,mult=negative_evidence_gate(
            pred["gps"][0],pred["gps"][1],
            q["gps_quality"],q["gps_valid"],
            variant=v,config=SETTINGS,
        )
        cases[v]=(rstar,mult)
    return m,pred,q,active,cases


def case_name(variant):
    return f"GPS_{variant.upper()}"


def report(model,inputs):
    m,pred,q,active,cases=inputs
    valid=active & q["gps_valid"]
    n=len(m.gps_local)
    if not np.any(valid):
        raise ValueError("No valid LiDAR/GPS feedback")
    print(f"0027 ONLY: model={model} n={n}, valid_pred={active.sum()}, "
          f"valid_feedback={valid.sum()}")
    print(f"  q<1={np.mean(q['gps_quality'][valid]<1-1e-8):.3%}, "
          f"q<.8={np.mean(q['gps_quality'][valid]<.8):.3%}")
    baseline=cases["reliability_only"][0]
    for variant,(rstar,mult) in cases.items():
        difference=rstar-baseline
        if np.any(difference>1e-9):
            raise AssertionError("Negative-evidence gate boosted reliability")
        print(f"  {variant:<17} changed={np.mean(abs(difference[valid])>1e-8):.3%}, "
              f"mean_dr={np.mean(difference[valid]):+.8f}, "
              f"min_mult={np.min(mult[valid]):.6f}")
    print("  Frozen non-GPS factors: IMU r-only, Camera r-only, LiDAR fixed")
    print("  Warmup first 63 frames and unpredicted tail 2 frames use fixed sigmas")
    print("  No test-drive reading, re-training, source file edits, or tuning")


def cache_check(directory:Path,spec:dict,n:int):
    if not directory.exists():
        return None
    pose=directory/"poses.npy"
    meta=directory/"metrics.json"
    if not pose.is_file() or not meta.is_file():
        raise RuntimeError(f"Partial result; refusing overwrite: {directory}")
    x=np.load(pose,allow_pickle=False)
    if x.shape!=(n,4,4) or not np.isfinite(x).all():
        raise ValueError(f"Invalid saved pose shape/values: {pose}")
    j=json.loads(meta.read_text(encoding="utf-8"))
    if j.get("spec")!=spec:
        raise ValueError("Cached development model/protocol mismatch")
    return j


def run_model(model:str,inputs,stage:str):
    m,pred,q,active,cases=inputs
    n=len(m.gps_local)
    if stage=="audit":
        return

    for variant in VARIANTS:
        spec={
            "source_model":model,
            "source_prediction":MODELS[model],
            "variant":variant,
            "protocol":"dev0027_gps_negative_evidence_v1",
            "gate_k":SETTINGS.contradiction_count,
            "no_u_constant":SETTINGS.no_u_constant,
            "gps_quality_window":CROSS.gps_window,
            "first_valid_model_frame":SETTINGS.warmup_frames,
            "last_valid_model_frame":n-SETTINGS.forecast_horizon,
            "other_sensor_policy":"IMU/Camera frozen reliability-only; LiDAR fixed",
            "same_initialization":"Stage6 LiDAR -> Camera odometry",
            "FGO_solver":"original Stage6 100 LM iterations; no continuation yet",
            "training_drive":"2011_10_03_drive_0027_sync",
            "heldout_drives_used":False,
        }
        target=OUT/model/case_name(variant)
        cached=cache_check(target,spec,n)
        if cached is not None:
            print(f"{model}/{variant}: CACHED ATE3D="
                  f"{cached['metrics']['ATE3D']:.6f}",flush=True)
            continue

        rstar,_=cases[variant]

        def frozen_prediction(source,sensor,count):
            if count!=n:
                raise ValueError("Frozen model frame count mismatch")
            if source==MODELS[model] and sensor in SENSORS:
                r=(rstar if sensor=="gps" else pred[sensor][0]).copy()
                # Stage6 maps u=0 to pure reliability sigma.
                # For invalid network windows u=1 => EXACT fixed sigma.
                u=np.ones(n,float)
                u[active]=0
                return r,u
            # Stage6 loads both backbones even though only one is
            # activated by its modes. Those other arrays are unused.
            return s6_original_load_prediction(source,sensor,count)

        s6_original_load_prediction=s6.load_prediction
        # These are in-memory patches of the EXISTING factor graph,
        # not new code for sensors or optimizer; always restored.
        with ExitStack() as stack:
            stack.enter_context(patch.object(s6,"OUT",OUT/model))
            stack.enter_context(patch.object(s6,"load_measurements",lambda:m))
            stack.enter_context(patch.object(s6,"load_lidar_risk",
                                            lambda length: np.zeros(length)))
            stack.enter_context(patch.object(s6,"load_prediction",
                                            frozen_prediction))
            print(f"\nFGO 0027 DEV: {model} / {variant}",flush=True)
            metrics=s6.run_case(
                name=case_name(variant),measurements=m,
                gps_mode=MODELS[model]+"_uncertainty",
                imu_mode=MODELS[model]+"_uncertainty",
                camera_mode=MODELS[model]+"_uncertainty",
                lidar_mode="fixed",
            )
        # Historical Stage6 runner saves poses/trajectory into our
        # isolated new directory. Fail closed if interrupted.
        pose=target/"poses.npy"
        if not pose.is_file():
            raise RuntimeError(f"Stage6 did not save poses: {pose}")
        manifest={
            "spec":spec,
            "metrics":metrics,
            "source":"original Stage6 factor graph reused in memory",
            "reference_caveat":"0027 OXTS/GPS-derived nonindependent GT",
            "feedback_caveat":"LiDAR disagreement q is a heuristic not proof of fault",
            "status":"development-only exploratory, not final convergence verified",
        }
        (target/"metrics.json").write_text(
            json.dumps(manifest,indent=2),encoding="utf-8"
        )
        print(f"{model}/{variant}: ATE3D={metrics['ATE3D']:.6f}m "
              f"(initial LM solve only)",flush=True)

    print("\nDEVELOPMENT-ONLY RELATIVE RESULTS")
    scores={}
    for variant in VARIANTS:
        f=OUT/model/case_name(variant)/"metrics.json"
        meta=json.loads(f.read_text(encoding="utf-8"))
        scores[variant]=float(meta["metrics"]["ATE3D"])
        print(f"  {variant:<17s} ATE3D={scores[variant]:.6f}m")
    print(f"  withU - noU = "
          f"{scores['with_u']-scores['no_u']:+.6f}m")
    print(f"  withU - rOnly = "
          f"{scores['with_u']-scores['reliability_only']:+.6f}m")
    print("Not yet a final paper comparison: convergence controls and "
          "prospectively frozen external drives still required.")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage",choices=("audit","fgo"),required=True)
    p.add_argument("--model",choices=("gru","mamba","both"),default="gru")
    args=p.parse_args()
    models=tuple(MODELS) if args.model=="both" else (args.model,)
    for model in models:
        inputs=make_inputs(model)
        report(model,inputs)
        run_model(model,inputs,args.stage)


if __name__=="__main__":
    main()
