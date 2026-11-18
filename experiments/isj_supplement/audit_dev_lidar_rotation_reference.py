from __future__ import annotations

"""0027 READ-ONLY: diagnose LiDAR between-rotation bias and direction convention.

Use frozen 0027 *physical* LiDAR and Camera body-frame BetweenPose3
rotations and ORIGINAL KITTI OXTS roll/pitch/yaw only as independent-
of-our-odometry (but NOT independent of GNSS/INS) post-hoc reference.
For each common valid pair t->t+1 calculate SO(3) angle mismatch,
signed yaw increment mismatch, quarter-wise signed drift, and
the error if each relative factor rotation were accidentally inverted.

This audit does not create/correct factor graph weights, alter
measurements, run FGO or touch 0016/0020. Do NOT use reference
attitude or the diagnostic inverse test to construct a deployable
factor or optimize a paper method.
"""

import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from experiments.factor_graph import run_stage6_v6_mamba_factor_graph_ablation as s6
from src.reliability.causal_gps_position_innovation import (
    CausalInnovationSettings,
)

SEQUENCE=ROOT/"dataset"/"kitti"/"2011_10_03"/"2011_10_03_drive_0027_sync"
WARMUP=63
END_MARGIN=2
CFG=CausalInnovationSettings()


def rpy_to_matrix(roll,pitch,yaw):
    """KITTI OXTS world-from-IMU/body: Rz(yaw)@Ry(pitch)@Rx(roll)."""
    cr,sr=np.cos(roll),np.sin(roll)
    cp,sp=np.cos(pitch),np.sin(pitch)
    cy,sy=np.cos(yaw),np.sin(yaw)
    Rx=np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]],float)
    Ry=np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]],float)
    Rz=np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]],float)
    return Rz@Ry@Rx


def reference_imu_rotation_pairs(seq_path,n):
    oxts=seq_path/"oxts"/"data"
    names=sorted(oxts.glob("*.txt"))
    if len(names)!=n:
        raise ValueError(f"Expected {n} original OXTS files, got {len(names)}")
    rpys=np.empty((n,3))
    for t,name in enumerate(names):
        vals=np.asarray(np.loadtxt(name),float).reshape(-1)
        if len(vals)<6 or not np.isfinite(vals[3:6]).all():
            raise ValueError(f"Malformed OXTS attitude {name}")
        rpys[t]=vals[3:6]
    reference=np.stack([rpy_to_matrix(*rpy) for rpy in rpys])
    # Relative GTSAM BetweenPose3 increment at IMU/body:
    # R_B(t)_B(t+1) = R_W_B(t)^T @ R_W_B(t+1).
    return np.matmul(np.swapaxes(reference[:-1],1,2),reference[1:])


def relative_rotation_angle_rad(a,b):
    """Geodesic SO(3) residual between relative rotations."""
    a=np.asarray(a,float)
    b=np.asarray(b,float)
    if (a.ndim!=3 or b.shape!=a.shape or a.shape[1:]!=(3,3)
        or not np.isfinite(a).all() or not np.isfinite(b).all()):
        raise ValueError("Expected matching finite (N,3,3) rotations")
    delta=np.matmul(np.swapaxes(a,1,2),b)
    c=np.clip((np.trace(delta,axis1=1,axis2=2)-1)/2,-1.,1.)
    return np.arccos(c)


def relative_yaw_increments(rot):
    r=np.asarray(rot,float)
    if r.ndim!=3 or r.shape[1:]!=(3,3) or not np.isfinite(r).all():
        raise ValueError("Invalid relative rotation array")
    return np.arctan2(r[:,1,0],r[:,0,0])


def signed_yaw_difference(a,b):
    """Wrap(a-b) = sensor minus reference, in [-pi,pi]."""
    delta=relative_yaw_increments(a)-relative_yaw_increments(b)
    return np.arctan2(np.sin(delta),np.cos(delta))


def report(name,rot,gt,valid,limit):
    if rot.shape!=(limit-1,3,3) or gt.shape!=rot.shape:
        raise ValueError("Inconsistent rotation factor counts")
    if valid.shape!=(limit-1,):
        raise ValueError("Invalid boolean factor mask")
    m=np.arange(limit-1)>=WARMUP
    m &= np.arange(limit-1)<limit-END_MARGIN-1
    m &=valid
    n=int(m.sum())
    if n<100:
        raise ValueError(f"Too few matched rotations for {name}: {n}")
    err=relative_rotation_angle_rad(rot,gt)
    inv=relative_rotation_angle_rad(np.swapaxes(rot,1,2),gt)
    signed=np.degrees(signed_yaw_difference(rot,gt))
    selected=err[m]
    select_inv=inv[m]
    print(f"\n{name:<7s} valid_pairs={n} "
          f"mean_so3_error_deg={np.degrees(np.mean(selected)):.5f} "
          f"p95_so3_error_deg={np.degrees(np.percentile(selected,95)):.5f} "
          f"INVERSE_mean_so3_error_deg={np.degrees(np.mean(select_inv)):.5f}")
    print(f"  signed_yaw_step_mean={signed[m].mean():+.7f}deg "
          f"median={np.median(signed[m]):+.7f}deg "
          f"sum_signed={signed[m].sum():+.3f}deg")
    edges=np.linspace(WARMUP,limit-END_MARGIN,5,dtype=int)
    for j in range(4):
        local=m & (np.arange(limit-1)>=edges[j]) & (
            np.arange(limit-1)<edges[j+1]
        )
        if local.sum()==0:
            continue
        print(f"  Q{j+1} n={local.sum():4d} "
              f"step_signed_mean={signed[local].mean():+.7f}deg "
              f"step_so3_mean={np.degrees(np.mean(err[local])):.6f}deg "
              f"cumulative_signed={signed[local].sum():+.2f}deg",
              flush=True)
    return selected,select_inv,signed[m]


def main():
    m=s6.load_measurements()
    n=len(m.gps_local)
    if n!=4544:
        raise ValueError(f"Unexpected 0027 frame count {n}")
    gt=reference_imu_rotation_pairs(SEQUENCE,n)
    L=np.asarray(m.lidar_between,float)[:n-1,:3,:3]
    C=np.asarray(m.camera_between,float)[:n-1,:3,:3]
    lv=np.asarray(m.lidar_valid,bool)[:n-1] & (
        np.asarray(m.lidar_quality,float)[:n-1]>=CFG.lidar_quality_min
    )
    cv=np.asarray(m.camera_valid,bool)[:n-1] & (
        np.asarray(m.camera_quality,float)[:n-1]>=CFG.camera_quality_min
    )
    common=lv & cv

    print("="*104)
    print("0027 READ-ONLY OXTS/INS ROTATION INCREMENT CHECK")
    print("Original OXTS orientation is REFERENCE ONLY, not independent absolute GT")
    print("Stage-6 physical LiDAR/Camera between factors in calibrated IMU/body frame")
    print(f"reference={SEQUENCE}/oxts/data; frames={n}")
    print("SO(3) error and signed local yaw angle; inverse test is diagnostic only")
    print("Compare the SAME common valid timestamp pairs for both sensors")
    print("No weights, no FGO, no training, no 0016/0020, no file writes")
    print("="*104)

    lerr,linv,ly=report("LiDAR",L,gt,common,n)
    cerr,cinv,cy=report("Camera",C,gt,common,n)
    difference=np.degrees(relative_rotation_angle_rad(L,C))
    pool=common.copy()
    pool[:WARMUP]=False
    pool[n-END_MARGIN-1:]=False
    yawdiff=np.degrees(signed_yaw_difference(L,C))
    print(f"\nLiDAR versus Camera (same {pool.sum()} valid pairs):")
    print(f"  SO3 discrepancy mean={np.mean(difference[pool]):.6f}deg "
          f"p95={np.percentile(difference[pool],95):.6f}deg")
    print(f"  signed step yaw LiDAR-Camera mean={np.mean(yawdiff[pool]):+.7f}deg "
          f"sum={np.sum(yawdiff[pool]):+.3f}deg")
    print("  INVERSE tests: an inverted factor orientation may indicate a "
          "conversion convention error, not a learned noise problem.")
    if np.mean(linv)<.6*np.mean(lerr):
        print("  WARNING: LiDAR INVERSE agrees much better with OXTS attitude; "
              "investigate pose-convention conversion before FGO.")
    if np.mean(cinv)<.6*np.mean(cerr):
        print("  WARNING: Camera INVERSE agrees much better with OXTS attitude; "
              "investigate pose-convention conversion before FGO.")
    print("\nAudit complete. Original OXTS orientation was used ONLY "
          "after frozen odometry estimation; never adjust method from 0020.")


if __name__=="__main__":
    main()
