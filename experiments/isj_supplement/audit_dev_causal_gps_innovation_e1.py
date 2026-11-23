from __future__ import annotations

"""E1 read-only 0027: test causal GPS 3D innovation BEFORE designing a gate.

Runs exclusively on historical development 2011_10_03_drive_0027_sync.
Construct feedback from degraded GPS + LiDAR/Camera *without clean GPS*.
Only after constructing feedback read clean GPS to label known synthetic
GPS injection for diagnostic discrimination. Ground truth/oracle/FGO
posterior are never passed to the prediction routine.

AUC scores use raw mismatch signals with identical matched frame masks:
new pre-update GPS vector innovation vs rejected k=5 displacement
magnitude mismatch. AUC is descriptive, not an independent test.
"No-fault" only means injected error < 5m relative to original OXTS GPS,
not a guaranteed correct independent GPS measurement. Long sustained
bias is not assured to be detected by a GPS-corrected causal estimator.

No FGO, no trained network, no parameter tuning, no files written.
0020/0016 are not opened. Metrics do not prove a paper contribution.
"""

import numpy as np
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from experiments.factor_graph import run_stage6_v6_mamba_factor_graph_ablation as s6
from src.reliability.causal_gps_position_innovation import (
    CausalInnovationSettings, causal_position_innovation,
)
from src.reliability.cross_modal_consistency import (
    CrossModalSettings, measurement_consistency,
)
from experiments.isj_supplement.audit_dev_uncertainty_risk_by_split import (
    rank_auc,finite_corr,
)

FAULT_THRESHOLD_M=5.0
INNOV=CausalInnovationSettings()
OLD=CrossModalSettings()


def performance(scores,label):
    scores=np.asarray(scores,float)
    label=np.asarray(label,bool)
    if scores.ndim!=1 or label.shape!=scores.shape:
        raise ValueError("Diagnostic score/label shape mismatch")
    if len(scores)<20 or not np.isfinite(scores).all():
        raise ValueError("Too few finite matched scores")
    pos=int(label.sum());neg=len(label)-pos
    if pos<5 or neg<5:
        raise ValueError(
            f"Not enough synthetic positive/negative examples: "
            f"positive={pos} negative={neg}"
        )
    auc=rank_auc(scores,label)
    # Diagnostic operating point: evaluate *post-hoc* negative-class 95th
    # percentile for comparison only. DO NOT deploy/tune this threshold
    # using external held-out data or call it an independent 5%-FPR.
    threshold=float(np.percentile(scores[~label],95))
    tpr=float(np.mean(scores[label]>threshold))
    fpr=float(np.mean(scores[~label]>threshold))
    return dict(
        n=len(scores),positive=pos,negative=neg,auc=float(auc),
        diagnostic_tpr_at_normal_p95=tpr,
        diagnostic_fpr=fpr,normal_p95_threshold=threshold,
    )


def _describe(name,values):
    a=np.asarray(values,float)
    if len(a)==0:return f"{name}: none"
    return (f"{name}: n={len(a)} mean={np.mean(a):.5f}, "
            f"p50={np.median(a):.5f}, p95={np.percentile(a,95):.5f}")


def main():
    m=s6.load_measurements()  # physical LiDAR + degraded 0027 GPS/IMU/Camera
    n=len(m.gps_local)
    if n!=4544:
        raise ValueError(f"Expected frozen development n=4544, got {n}")
    feedback=causal_position_innovation(
        gps_local=m.gps_local,
        lidar_between=m.lidar_between,lidar_valid=m.lidar_valid,
        lidar_quality=m.lidar_quality,
        camera_between=m.camera_between,camera_valid=m.camera_valid,
        camera_quality=m.camera_quality,settings=INNOV,
    )
    legacy=measurement_consistency(
        gps_local=m.gps_local,imu_gyro=m.imu_gyro,
        timestamps_seconds=np.asarray(m.timestamps,float),
        lidar_between=m.lidar_between,lidar_valid=m.lidar_valid,
        lidar_quality=m.lidar_quality,settings=OLD,
    )

    # This is the ONLY read of clean GPS, AFTER generating BOTH signals;
    # it cannot influence prediction, fitting, masking or propagation.
    reference=s6.BASE/"degraded_sensor_data.npz"
    with np.load(reference,allow_pickle=False) as z:
        clean=np.asarray(z["clean_gps_local"],float)
        sev=(np.asarray(z["gps_severity"],float)
             if "gps_severity" in z.files else None)
    if clean.shape!=m.gps_local.shape or not np.isfinite(clean).all():
        raise ValueError("Missing/malformed original OXTS-based GPS reference")
    injected=np.linalg.norm(m.gps_local-clean,axis=1)
    label=injected>=FAULT_THRESHOLD_M
    common=feedback["valid"] & legacy["gps_valid"]
    # E1 can score all frames >=warmup; restrict to frozen network
    # prediction-valid range to make later diagnostics comparable.
    valid_end=n-2
    common[valid_end:]=False
    if not np.any(common):
        raise ValueError("No shared valid E1 diagnostic frames")

    print("="*104)
    print("E1 0027 DEVELOPMENT-ONLY: CAUSAL PRE-UPDATE GPS VECTOR INNOVATION")
    print(f"n={n}; evaluated t={INNOV.warmup}..{valid_end-1}; "
          f"matched frames={common.sum()}; label=synthetic GPS error >=5m")
    print(f"Past-only GPS heading bootstrap yaw_rad={feedback['bootstrapped_heading_rad']:+.5f}")
    print("Current GPS not used in prediction until AFTER saving innovation")
    print("Physical LiDAR used for prior; camera fallback; no OXTS truth used for prior")
    print("Innovation normalized score uses a heuristic variance proxy; NOT NIS")
    print("Reference label is ORIGINAL OXTS GPS, not independent GT")
    print("0020 untouched, no write, no optimizer, no retraining")
    print("="*104)
    print(_describe("synthetic GPS error m",injected[common]))
    print(_describe("new vector innovation norm m",feedback["innovation_norm_m"][common]))
    print(_describe("new standardized score",feedback["standardized_score"][common]))
    print(_describe("old magnitude mismatch NIS",legacy["gps_nis"][common]))
    print("source modes (among all post-warmup):",{
        "lidar":int(np.sum(feedback["odometry_source"]==1)),
        "camera":int(np.sum(feedback["odometry_source"]==2)),
        "missing":int(np.sum(
            (feedback["odometry_source"]==0)&
            (np.arange(n)>=INNOV.warmup)
        )),
    })
    geometry=feedback["geometry_consistent"]
    print(f"LiDAR-Camera geometry agreement among matched frames: "
          f"{int((common & geometry).sum())}/{int(common.sum())} "
          f"({(common & geometry).sum()/common.sum():.2%})")

    for name,subset in (
        ("ALL_MATCHED",common),
        ("GEOMETRY_AGREES",common & geometry),
        ("GEOMETRY_CONFLICT_OR_UNAVAILABLE",common & ~geometry),
    ):
        if subset.sum()<20:
            print(name+": insufficient diagnostic support, skip")
            continue
        print(f"\n--- {name} n={subset.sum()} "
              f"fault_fraction={np.mean(label[subset]):.2%} ---")
        if sev is not None:
            print("Pearson(vector score, injected severity):",
                  f"{finite_corr(feedback['standardized_score'][subset],sev[subset]):+.4f}")
        for method,score in (
            ("VECTOR_PRE_UPDATE",feedback["standardized_score"]),
            ("VECTOR_METERS",feedback["innovation_norm_m"]),
            ("OLD_MAGNITUDE_ONLY",legacy["gps_nis"]),
        ):
            try:
                v=performance(score[subset],label[subset])
                print(
                    f"{method:<23} AUC={v['auc']:.4f} "
                    f"TPR_at_dev_normal_p95={v['diagnostic_tpr_at_normal_p95']:.4f} "
                    f"observed_FPR={v['diagnostic_fpr']:.4f} "
                    f"pos={v['positive']} neg={v['negative']}"
                )
            except ValueError as exc:
                print(f"{method:<23} unavailable: {exc}")
    print("\nE1 FINISHED (read-only); do not judge success by only one AUC.")
    print("If geometry false alarms/causal drift are large, redesign prior on 0027 only.")
    print("Do not evaluate/tune on 0016 or the untouched 0020 drive.")


if __name__=="__main__":
    main()
