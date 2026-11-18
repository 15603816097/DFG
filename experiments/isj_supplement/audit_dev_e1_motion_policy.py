from __future__ import annotations

"""READ-ONLY 0027 causal GPS E1: isolate orientation source.

Compare original E1 LiDAR-first prior against Camera-first and mixed
LiDAR-body-translation/Camera-body-rotation. All variants use the same
predeclared E1 GPS update, rough variance proxy and 63-frame *historical*
degraded-GPS yaw bootstrap; only the relative-pose source changes.
Each step uses factor i=t-1 only; current GPS enters AFTER innovation.
The mixed-pose experiment is a diagnostic, NOT a calibrated optimal
fusion or a completed ISJ method.

The original OXTS GPS is read only AFTER computing all causal signals
to label synthetic injected GPS error>=5m. Label is NOT independent
ground truth, and this intra-drive AUC is not a fresh external test.
No model predictions, FGO, optimizer or files written. Never touch
previously explored 0016 or prospective 0020 for this experiment.
"""

from dataclasses import replace

import numpy as np

from experiments.factor_graph import run_stage6_v6_mamba_factor_graph_ablation as s6
from experiments.isj_supplement.audit_dev_causal_gps_innovation_e1 import (
    performance,
)
from src.reliability.causal_gps_position_innovation import (
    CausalInnovationSettings, causal_position_innovation,
)
from src.reliability.cross_modal_consistency import (
    measurement_consistency, CrossModalSettings,
)

POLICIES=(
    "lidar_first",
    "camera_first",
    "camera_rotation_lidar_translation",
)
SCORES=("innovation_norm_m","standardized_score")
LABEL_ERROR_M=5.0
WARMUP=63
LAST_VALID_MARGIN=2


def collect(m):
    n=len(m.gps_local)
    cfg=CausalInnovationSettings()
    outs={}
    args={
        "gps_local":m.gps_local,
        "lidar_between":m.lidar_between,
        "lidar_valid":m.lidar_valid,
        "lidar_quality":m.lidar_quality,
        "camera_between":m.camera_between,
        "camera_valid":m.camera_valid,
        "camera_quality":m.camera_quality,
    }
    for policy in POLICIES:
        outs[policy]=causal_position_innovation(
            **args,settings=replace(cfg,motion_policy=policy)
        )
    # Reference old measurement discrepancy is built from noisy
    # measurements only. No clean GPS or synthetic labels in here.
    old=measurement_consistency(
        gps_local=m.gps_local,imu_gyro=m.imu_gyro,
        timestamps_seconds=np.asarray(m.timestamps,float),
        lidar_between=m.lidar_between,lidar_valid=m.lidar_valid,
        lidar_quality=m.lidar_quality,settings=CrossModalSettings(),
    )
    common=old["gps_valid"].copy()
    for policy in POLICIES:
        common &=outs[policy]["valid"]
    common[:WARMUP]=False
    common[n-LAST_VALID_MARGIN:]=False
    if common.sum()<100:
        raise ValueError(f"Insufficient common matched frames: {common.sum()}")
    return outs,old,common


def summarize(name,score,label,common):
    a=np.asarray(score,float)[common]
    y=np.asarray(label,bool)[common]
    if not np.isfinite(a).all():
        raise ValueError(f"Nonfinite {name} scores")
    result=performance(a,y)
    print(
        f"  {name:<43s} AUC={result['auc']:.4f} "
        f"TPR_at_dev_normal_p95={result['diagnostic_tpr_at_normal_p95']:.4f} "
        f"FPR={result['diagnostic_fpr']:.4f} "
        f"mean={np.mean(a):.4f} "
        f"median={np.median(a):.4f} "
        f"p95={np.percentile(a,95):.4f}",
        flush=True
    )


def main():
    m=s6.load_measurements()
    n=len(m.gps_local)
    if n!=4544:
        raise ValueError(f"Expected 0027 with 4544 frames, got {n}")
    outs,old,common=collect(m)

    # Reference is only used for a post-hoc synthetic-injection label.
    with np.load(s6.BASE/"degraded_sensor_data.npz",allow_pickle=False) as z:
        clean=np.asarray(z["clean_gps_local"],float)
    if clean.shape!=(n,3) or not np.isfinite(clean).all():
        raise ValueError("Invalid original OXTS/GPS reference")
    injected=np.linalg.norm(m.gps_local-clean,axis=1)
    label=injected>=LABEL_ERROR_M

    print("="*100)
    print("0027 E1 SOURCE-POLICY DIAGNOSTIC: NO TRAINING / NO FGO / READ ONLY")
    print("All policies evaluated on SAME matched frames; yaw bootstrap from first 63 DEGRADED GPS")
    print("Current-frame GPS never used in its own prior; original OXTS/GPS used for label ONLY")
    print("AUC labels mean synthetic injection>=5m, not independent real GNSS fault")
    print("Heuristic normalizer is NOT calibrated NIS; no parameters tuned")
    print("Only 0027 touched. 0016/0020 excluded. No files written.")
    print(f"n={n} matched={common.sum()} "
          f"fault_fraction={label[common].mean():.3%}")
    print("="*100)

    for policy in POLICIES:
        out=outs[policy]
        counts={
            code:int(np.sum((out["odometry_source"]==code)&common))
            for code in (0,1,2,3)
        }
        print(f"\n[{policy}] "
              f"bootstrap_yaw_deg={np.degrees(out['bootstrapped_heading_rad']):+.3f} "
              f"source_counts_0missing_1lidar_2camera_3hybrid={counts}")
        for metric in SCORES:
            summarize(f"{policy}.{metric}",out[metric],label,common)

    print("\n[OLD STRONG BASELINE, SAME MATCHED FRAMES]")
    summarize("5-step GPS/LiDAR raw magnitude mismatch",
              old["gps_nis"],label,common)

    print("\n[CHRONOLOGICAL QUARTER RAW METER INNOVATION]")
    eligible=np.flatnonzero(common)
    edges=np.linspace(eligible[0],eligible[-1]+1,5,dtype=int)
    for i in range(4):
        selected=common&(
            (np.arange(n)>=edges[i])&
            (np.arange(n)<edges[i+1])
        )
        print(f"  Q{i+1}: n={selected.sum()}")
        if selected.sum()<10:
            continue
        for policy in POLICIES:
            values=outs[policy]["innovation_norm_m"][selected]
            print(f"    {policy:<38s} "
                  f"mean={values.mean():.4f}m "
                  f"p95={np.percentile(values,95):.4f}m")
    print("\nDONE. Camera-first/HYBRID are *diagnostic* substitutions.")
    print("If either improves fault AUC, that DOES NOT establish FGO location gains.")
    print("Do not freeze a paper method until additional causal and fair FGO checks.")


if __name__=="__main__":
    main()
