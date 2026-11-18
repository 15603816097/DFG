from __future__ import annotations

"""0027 READ-ONLY: localization trade-off vs LiDAR/Camera rotation disagreement.

Consumes the FOUR FINISHED, UNIFORMLY CONTINUED, manifest-verified
Stage-6 fixed-FGO LiDAR rotational-sigma studies. Uses NO new FGO,
no model training, no tuning of noise weights and writes NO files.

Descriptively stratifies each pose trajectory by chronological quarters
and by a matched-pair (valid physical LiDAR + Camera) SO(3)
disagreement threshold at 0027's 75th percentile. This label comes
ONLY from physical measurements, not OXTS/GT or saved optimizer poses.
It is an exploratory DEVELOPMENT sequence, not held-out evidence.
The split threshold is not an online method; do not use as such
or tune on 0016/0020.

For the exact same frames in every static-sigma case, reports
  - ATE3D / ATE2D / ZRMSE (position error against original
    OXTS/GPS-derived reference, NOT independent GNSS ground truth);
  - RPE translation RMSE k=1,10 under a wholly contained chronological
    frame segment; within high/low discrepancy groups, k=1 only
    (pairwise disagreement corresponds to the current adjacent pair).
Evaluation references never enter factor measurements or masks.

No raw OXTS files needed. Only pre-existing 0027 Stage6 data/poses.
"""

from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from experiments.factor_graph import run_stage6_v6_mamba_factor_graph_ablation as s6
from experiments.isj_supplement import run_dev_lidar_rot_sigma_convergence as conv

N=conv.N
WARMUP=63
END_MARGIN=2
PERCENTILE=75.0
RPE_K=(1,10)


def angle_deg_lidar_camera(measurements):
    n=len(measurements.gps_local)
    L=np.asarray(measurements.lidar_between,float)
    C=np.asarray(measurements.camera_between,float)
    if L.shape!=(n-1,4,4) or C.shape!=(n-1,4,4):
        raise ValueError("Unexpected physical relative factor dimensions")
    if not np.isfinite(L).all() or not np.isfinite(C).all():
        raise ValueError("Nonfinite relative factor values")
    lv=np.asarray(measurements.lidar_valid,bool)
    cv=np.asarray(measurements.camera_valid,bool)
    lq=np.asarray(measurements.lidar_quality,float)
    cq=np.asarray(measurements.camera_quality,float)
    if any(x.shape!=(n-1,) for x in (lv,cv,lq,cq)):
        raise ValueError("Unexpected odometry quality/valid arrays")
    valid=lv & cv & (lq>=.1) & (cq>=.1)
    delta=np.einsum("tji,tjk->tik",L[:,:3,:3],C[:,:3,:3])
    cosine=np.clip((np.trace(delta,axis1=1,axis2=2)-1)*.5,-1.,1.)
    angle=np.rad2deg(np.arccos(cosine))
    return angle,valid


def sensor_only_quarter_profile(m, angle, valid, threshold, start, end):
    """Describe existing physical factors on [start,end); no pose or GT input.

    Quantile threshold from entire 0027 is descriptive only. A single
    sensor-pair angular difference cannot identify which sensor is at
    fault; do not use this as a deployed dynamic weight.
    """
    n=len(m.gps_local)
    if not (0<=start<end<=n-1):
        raise ValueError("Invalid quarter bounds")
    a=np.asarray(angle,float)
    v=np.asarray(valid,bool)
    if a.shape!=(n-1,) or v.shape!=(n-1,):
        raise ValueError("Sensor angle/mask dimension mismatch")
    selected=np.arange(start,end,dtype=int)
    common=selected[v[selected]]
    if len(common)<10:
        raise ValueError("Insufficient common physical odometry pairs")
    L=np.asarray(m.lidar_between,float)
    C=np.asarray(m.camera_between,float)
    lq=np.asarray(m.lidar_quality,float)
    cq=np.asarray(m.camera_quality,float)
    signed=np.rad2deg(np.arctan2(
        np.sin(
            np.arctan2(L[common,1,0],L[common,0,0])
            -np.arctan2(C[common,1,0],C[common,0,0])
        ),
        np.cos(
            np.arctan2(L[common,1,0],L[common,0,0])
            -np.arctan2(C[common,1,0],C[common,0,0])
        ),
    ))
    vals=a[common]
    return {
        "valid_pair_fraction":float(len(common)/len(selected)),
        "valid_pairs":int(len(common)),
        "high_disagreement_fraction":float(np.mean(vals>=threshold)),
        "angle_mean_deg":float(np.mean(vals)),
        "angle_median_deg":float(np.median(vals)),
        "angle_p95_deg":float(np.percentile(vals,95)),
        "signed_relative_yaw_mean_deg":float(np.mean(signed)),
        "lidar_quality_mean":float(np.mean(lq[common])),
        "camera_quality_mean":float(np.mean(cq[common])),
        "lidar_step_distance_mean_m":float(np.mean(np.linalg.norm(
            L[common,:3,3],axis=1
        ))),
        "camera_step_distance_mean_m":float(np.mean(np.linalg.norm(
            C[common,:3,3],axis=1
        ))),
    }


def print_sensor_profiles(m,angle,valid,threshold,edges):
    print("\nSENSOR-ONLY CHRONOLOGICAL QUARTER PROFILES")
    print("No reference GT, saved FGO pose, or ATE is used to calculate "
          "any profile below; Q75 cutoff is development-only DESCRIPTIVE.")
    for i in range(len(edges)-1):
        start,end=int(edges[i]),int(edges[i+1])
        info=sensor_only_quarter_profile(
            m,angle,valid,threshold,start,end,
        )
        print(
            f"  Q{i+1} {start}..{end-1}: "
            f"pairs={info['valid_pairs']} "
            f"pair_valid={info['valid_pair_fraction']:.2%} "
            f"high_angle_fraction={info['high_disagreement_fraction']:.2%} "
            f"angle_mean={info['angle_mean_deg']:.5f}deg "
            f"angle_p50={info['angle_median_deg']:.5f}deg "
            f"angle_p95={info['angle_p95_deg']:.5f}deg "
            f"signed_yaw_step_mean={info['signed_relative_yaw_mean_deg']:+.5f}deg "
            f"lidar_q_mean={info['lidar_quality_mean']:.4f} "
            f"camera_q_mean={info['camera_quality_mean']:.4f} "
            f"lidar_step_mean={info['lidar_step_distance_mean_m']:.4f}m "
            f"camera_step_mean={info['camera_step_distance_mean_m']:.4f}m",
            flush=True,
        )


def position_metrics(poses,gt,indices):
    poses=np.asarray(poses,float)
    gt=np.asarray(gt,float)
    ix=np.asarray(indices,int)
    if poses.ndim!=3 or poses.shape[1:]!=(4,4):
        raise ValueError("Invalid poses")
    if gt.shape!=(len(poses),3):
        raise ValueError("Invalid evaluation reference length")
    if ix.ndim!=1 or len(ix)<1 or np.any(ix<0) or np.any(ix>=len(poses)):
        raise ValueError("Invalid evaluation index mask")
    if not np.isfinite(poses).all() or not np.isfinite(gt).all():
        raise ValueError("Nonfinite pose/eval data")
    err=poses[ix,:3,3]-gt[ix,:3]
    return (
        float(np.sqrt(np.mean(np.sum(err**2,axis=1)))),
        float(np.sqrt(np.mean(np.sum(err[:,:2]**2,axis=1)))),
        float(np.sqrt(np.mean(err[:,2]**2))),
    )


def rpe_trans_rmse(poses,gt,starts,k):
    poses=np.asarray(poses,float)
    gt=np.asarray(gt,float)
    t=np.asarray(starts,int)
    n=len(poses)
    if (poses.shape!=(n,4,4) or gt.shape!=(n,3)
        or t.ndim!=1 or not len(t) or k<1
        or np.any(t<0) or np.any(t+k>=n)):
        raise ValueError("Invalid start indices for translational RPE")
    e=(poses[t+k,:3,3]-poses[t,:3,3])-(gt[t+k]-gt[t])
    return float(np.sqrt(np.mean(np.einsum("ij,ij->i",e,e))))


def get_cases():
    poses={}
    for name,sigma in conv.pilot.SPECS:
        manifest=conv.get_cached_result(
            conv.OUT/name,conv.desired_spec(name,sigma)
        )
        if manifest is None:
            raise FileNotFoundError(
                f"Missing required uniform convergence case {conv.OUT/name}"
            )
        if not manifest["continuation"]["stationary_under_rule"]:
            raise ValueError(f"{name} did not satisfy uniform stop criterion")
        poses[name]=np.load(conv.OUT/name/"poses.npy",allow_pickle=False)
        if poses[name].shape!=(N,4,4):
            raise ValueError(f"{name}: unexpected poses length")
        eval_metrics=s6.evaluate(poses[name][:,:3,3])
        if abs(eval_metrics["ATE3D"]-manifest["final_metrics"]["ATE3D"])>1e-4:
            raise ValueError(f"{name}: poses disagree with saved metrics")
    return poses


def tabulate(label,poses,gt,frames,rpe_starts):
    frames=np.asarray(frames,int)
    if not len(frames):
        print(f"\n{label}: NO FRAMES")
        return
    print(f"\n{label} | n_positions={len(frames)} "
          + " ".join(f"n_RPE_k{k}={len(v)}" for k,v in rpe_starts.items()))
    for name,_ in conv.pilot.SPECS:
        ate3,ate2,z=position_metrics(poses[name],gt,frames)
        rs=[]
        for k,starts in rpe_starts.items():
            if len(starts):
                rs.append(f"RPE_k{k}={rpe_trans_rmse(poses[name],gt,starts,k):.5f}")
        print(f"  {name:<28s} ATE3D={ate3:.5f}m "
              f"ATE2D={ate2:.5f}m ZRMSE={z:.5f}m "
              +(" ".join(rs) if rs else "RPE=N/A"),flush=True)


def main():
    m=s6.load_measurements()
    if len(m.gps_local)!=N:
        raise ValueError(f"Expected 0027 {N} frames only")
    trajectory=s6.GT
    gt=np.loadtxt(trajectory,dtype=float)
    if gt.ndim!=2 or gt.shape[0]!=N or gt.shape[1]<3:
        raise ValueError("0027 OXTS/GPS-derived evaluation path mismatch")
    gt=gt[:,-3:]
    poses=get_cases()

    angle,valid=angle_deg_lidar_camera(m)
    candidate=np.arange(WARMUP,N-END_MARGIN)
    valid_in_window=np.zeros(N,bool)
    valid_in_window[candidate]=True
    valid_pair=valid & valid_in_window[:-1]
    if valid_pair.sum()<100:
        raise ValueError("Insufficient common LiDAR/Camera observations")
    threshold=float(np.percentile(angle[valid_pair],PERCENTILE))
    high=valid_pair & (angle>=threshold)
    low=valid_pair & (angle<threshold)

    print("="*103)
    print("READ-ONLY 0027 LOCALIZATION TRADE-OFF vs ROTATION DISAGREEMENT")
    print(f"n={N} shared valid pairs={valid_pair.sum()}, "
          f"Q75 angle={threshold:.6f}deg")
    print(f"high_disagreement_pairs={high.sum()} "
          f"low_disagreement_pairs={low.sum()}")
    print("Angle threshold is a descriptive 0027 sensor-only quantile, "
          "NOT calibrated failure probability and NOT a deployable gate")
    print("All FOUR saved uniformly converged FGO cases are reused "
          "with SAME matched evaluation masks")
    print("Original OXTS/GPS reference used for EVALUATION ONLY, not independent GT")
    print("No 0016/0020, no optimization, no prediction retraining, NO files written")
    print("="*103)

    base_idx=candidate
    all_rpe={
        k:np.arange(WARMUP,N-END_MARGIN-k,dtype=int) for k in RPE_K
    }
    tabulate("ALL DEVELOPMENT FRAMES",poses,gt,base_idx,all_rpe)

    edges=np.linspace(WARMUP,N-END_MARGIN,5,dtype=int)
    print_sensor_profiles(m,angle,valid_pair,threshold,edges)
    for i in range(4):
        start,end=int(edges[i]),int(edges[i+1])
        mask=np.arange(start,end,dtype=int)
        starts={
            k:np.arange(start,end-k,dtype=int) if end-start>k else np.array([],int)
            for k in RPE_K
        }
        tabulate(f"Q{i+1} t={start}..{end-1}",poses,gt,mask,starts)

    for group_name,select in (
        ("LOW SENSOR DISAGREEMENT",low),
        ("HIGH SENSOR DISAGREEMENT",high),
    ):
        idx=np.flatnonzero(select)
        tabulate(group_name,poses,gt,idx,{1:idx})

    print("\nINTERPRETATION: ATE improvement in one subgroup alone does NOT "
          "prove an online rotation gate improves ATE/RPE; "
          "subgroups are observational and can correlate with motion regimes.")
    print("STOP before any new covariance formula: inspect whether the "
          "high-disagreement subgroup localizes error trade-offs at all.")


if __name__=="__main__":
    main()
