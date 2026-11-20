from __future__ import annotations

"""READ-ONLY 0027 odometry geometry audit, BEFORE any additional model/gate.

Compares LiDAR (physical Stage-6 BetweenPose3) + optional Camera fallback
motion with original, uncorrupted OXTS/GPS local displacements at short
horizons 1, 5, 20, 63 (evaluation reference ONLY). This original GPS is
NOT independent GNSS ground truth. No FGO pose or synthetic fault labels
used to construct the motion, and no outputs are written.

Separates:
* norm error, insensitive to absolute heading;
* vector error after a yaw bootstrap fitted on the first 63 DEGRADED
  GPS positions (same bootstrap as failed E1);
* direction disagreement in XY for diagnostic interpretation.

Only 0027 and existing physical LiDAR/Camera factor NPZs are read.
0020 and 0016 remain untouched. Model/network weights not loaded.
"""

from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from experiments.factor_graph import run_stage6_v6_mamba_factor_graph_ablation as s6
from src.reliability.causal_gps_position_innovation import (
    CausalInnovationSettings,causal_position_innovation,
    _yaw_rotation,
)

HORIZONS=(1,5,20,63)
SETTINGS=CausalInnovationSettings()


def segment_errors(
    between,valid,quality,reference,rotations,k,
    *,first=63,last=None,quality_floor=.1,
):
    """Return paired clean-vs-LiDAR length, vector and heading errors.

    Rotation at each start frame is a bootstrap global orientation
    estimated only from pre-63 noisy GPS and relative odometry.
    This function never estimates heading from the future clean
    reference; 'reference' is used strictly for error evaluation.
    """
    T=np.asarray(between,float)
    mask=np.asarray(valid,bool) & (np.asarray(quality,float)>=quality_floor)
    gps=np.asarray(reference,float)
    R=np.asarray(rotations,float)
    n=len(gps)
    limit=n if last is None else min(n,last)
    if (T.shape!=(n-1,4,4) or mask.shape!=(n-1,)
            or R.shape!=(n,3,3) or gps.shape!=(n,3)):
        raise ValueError("Odometry/reference dimensions mismatch")
    if not all(np.isfinite(x).all() for x in (T,gps,R)):
        raise ValueError("Nonfinite diagnostic input")
    if not isinstance(k,int) or k<1 or first<0 or limit-first<=k:
        raise ValueError("Invalid horizon or evaluation range")
    length_errors=[]
    vector_errors=[]
    heading_angles=[]
    gps_distances=[]
    odom_distances=[]
    used=0
    for start in range(first,limit-k):
        if not np.all(mask[start:start+k]):
            continue
        chain=np.eye(4)
        for j in range(start,start+k):
            chain=chain@T[j]
        odom_disp=R[start]@chain[:3,3]
        gps_disp=gps[start+k]-gps[start]
        a=float(np.linalg.norm(odom_disp))
        b=float(np.linalg.norm(gps_disp))
        odom_distances.append(a)
        gps_distances.append(b)
        length_errors.append(abs(a-b))
        vector_errors.append(float(np.linalg.norm(odom_disp-gps_disp)))
        axy=odom_disp[:2]
        bxy=gps_disp[:2]
        if min(np.linalg.norm(axy),np.linalg.norm(bxy))>0.1:
            angle=np.arctan2(
                axy[0]*bxy[1]-axy[1]*bxy[0],
                axy@bxy
            )
            heading_angles.append(abs(float(angle)))
        used+=1
    if used==0:
        raise ValueError("No valid relative odometry pairs for diagnostic")
    return {
        "pairs":used,
        "gps_distance_mean_m":float(np.mean(gps_distances)),
        "odom_distance_mean_m":float(np.mean(odom_distances)),
        "length_mae_m":float(np.mean(length_errors)),
        "length_p95_m":float(np.percentile(length_errors,95)),
        "vector_mae_m":float(np.mean(vector_errors)),
        "vector_p95_m":float(np.percentile(vector_errors,95)),
        "heading_abs_median_deg":(
            float(np.degrees(np.median(heading_angles)))
            if heading_angles else float("nan")
        ),
    }


def bootstrap_world_rotations(m,yaw,settings=SETTINGS):
    n=len(m.gps_local)
    R=np.empty((n,3,3),float)
    R[0]=_yaw_rotation(yaw)
    good_l=np.asarray(m.lidar_valid,bool)&(
        np.asarray(m.lidar_quality,float)>=settings.lidar_quality_min
    )
    good_c=np.asarray(m.camera_valid,bool)&(
        np.asarray(m.camera_quality,float)>=settings.camera_quality_min
    )
    for i in range(n-1):
        step=(m.lidar_between[i] if good_l[i] else
              m.camera_between[i] if good_c[i] else np.eye(4))
        R[i+1]=R[i]@step[:3,:3]
    return R


def print_result(source,k,metrics):
    print(
        f"  {source:<17s} k={k:2d} pairs={metrics['pairs']:4d} "
        f"gps_dist={metrics['gps_distance_mean_m']:.3f}m "
        f"odom_dist={metrics['odom_distance_mean_m']:.3f}m "
        f"|norm_err|mean={metrics['length_mae_m']:.3f}m "
        f"p95={metrics['length_p95_m']:.3f}m "
        f"vector_mean={metrics['vector_mae_m']:.3f}m "
        f"p95={metrics['vector_p95_m']:.3f}m "
        f"heading_med={metrics['heading_abs_median_deg']:.1f}deg",
        flush=True
    )


def main():
    m=s6.load_measurements()
    n=len(m.gps_local)
    if n!=4544:
        raise ValueError(f"Expected frozen KITTI 0027 n=4544, got {n}")
    with np.load(s6.BASE/"degraded_sensor_data.npz",
                 allow_pickle=False) as data:
        clean=np.asarray(data["clean_gps_local"],float)
    if clean.shape!=(n,3) or not np.isfinite(clean).all():
        raise ValueError("Invalid OXTS-derived GPS reference shape")
    e1=causal_position_innovation(
        gps_local=m.gps_local,
        lidar_between=m.lidar_between,lidar_valid=m.lidar_valid,
        lidar_quality=m.lidar_quality,
        camera_between=m.camera_between,camera_valid=m.camera_valid,
        camera_quality=m.camera_quality,settings=SETTINGS,
    )
    world_rot=bootstrap_world_rotations(
        m,e1["bootstrapped_heading_rad"],SETTINGS
    )
    print("="*112)
    print("0027 E1-ROOT-CAUSE: READ-ONLY SHORT-HORIZON ODOMETRY GEOMETRY CHECK")
    print("Reference is original OXTS/GPS-derived local coordinates (NOT independent).")
    print("E1 bootstrapped yaw from first 63 DEGRADED GPS frames; "
          f"yaw={e1['bootstrapped_heading_rad']:+.5f}rad.")
    print("Vector errors incorporate this bootstrapped yaw and propagated "
          "odometry heading; norm errors do not.")
    print("Comparison uses original OXTS/GPS ONLY for post-hoc evaluation.")
    print("No future clean GPS used in prior, 0016/0020 not accessed, no files written.")
    print("="*112)
    for source in ("LiDAR","Camera"):
        T=np.asarray(getattr(m,source.lower()+"_between"),float)
        valid=np.asarray(getattr(m,source.lower()+"_valid"),bool)
        quality=np.asarray(getattr(m,source.lower()+"_quality"),float)
        floor=(SETTINGS.lidar_quality_min if source=="LiDAR"
               else SETTINGS.camera_quality_min)
        for k in HORIZONS:
            try:
                result=segment_errors(
                    T,valid,quality,clean,world_rot,k,
                    first=SETTINGS.warmup,last=n-2,
                    quality_floor=floor
                )
            except ValueError as e:
                print(f"  {source} k={k}: not available: {e}")
                continue
            print_result(source,k,result)
    print("Interpretation: large norm error implies translation magnitude/"
          "scale/odometry problem; small norm but large vector error implies "
          "heading, direction, frame alignment or accumulated orientation problem.")
    print("Original GPS is not independent ground truth; do not claim "
          "real-world drift calibration or robustness from this check.")
    print("E1 global innovation was 62m mean and fault AUC=.6006; "
          "these local checks identify whether geometrical propagation is plausible.")


if __name__=="__main__":
    main()
