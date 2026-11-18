from __future__ import annotations

"""Read-only 0027 signed-heading diagnostic after E1 mean GPS innovation 62m.

For LiDAR and Camera SEPARATELY:
- fit yaw using only frames [0,62] of *degraded* GPS and corresponding
  historical odometry (same causal bootstrap principle as E1);
- propagate rotations using ONLY that odometry stream;
- evaluate signed (not just absolute) direction differences against
  the original OXTS/GPS displacement in 4 chronological blocks, k=1,5,20.
- fit a single *oracle* yaw correction using original OXTS GPS over
  the whole development sequence SOLELY as post-hoc diagnostics.
  It is NEVER an actual prediction or recommended deployable tuning.

No 0020/0016, no model prediction, no factor graph, no file writes.
References are not independent high-precision ground truth.
"""

from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from experiments.factor_graph import run_stage6_v6_mamba_factor_graph_ablation as s6
from src.reliability.causal_gps_position_innovation import _yaw_rotation

WARMUP=63
END_MARGIN=2
HORIZONS=(1,5,20)
MIN_HORIZ_DIST_M=.35
MIN_SUPPORT_FRAC=.5


def historical_yaw(gps,between,valid,quality,*,warmup=WARMUP,floor=.1):
    """Only past degraded GPS and past odometry are available at warmup."""
    gps=np.asarray(gps,float)
    between=np.asarray(between,float)
    mask=np.asarray(valid,bool)&(np.asarray(quality,float)>=floor)
    if (gps.ndim!=2 or gps.shape[1]!=3 or len(gps)<=warmup
        or between.shape!=(len(gps)-1,4,4)
        or mask.shape!=(len(gps)-1,)
        or warmup<4):
        raise ValueError("Invalid history shapes/window")
    positions=np.zeros((warmup,3))
    R=np.eye(3)
    count=0
    for t in range(1,warmup):
        i=t-1
        positions[t]=positions[t-1]
        if mask[i]:
            positions[t]+=R@between[i,:3,3]
            R=R@between[i,:3,:3]
            count+=1
    if count<max(4,int((warmup-1)*MIN_SUPPORT_FRAC)):
        raise ValueError("Insufficient historical odometry")
    x=positions[:,:2]-np.mean(positions[:,:2],axis=0)
    y=gps[:warmup,:2]-np.mean(gps[:warmup,:2],axis=0)
    cross=np.sum(x[:,0]*y[:,1]-x[:,1]*y[:,0])
    dot=np.sum(x*y)
    if np.linalg.norm(x)<1e-6 or np.hypot(cross,dot)<1e-8:
        raise ValueError("Historical yaw not identifiable")
    return float(np.arctan2(cross,dot))


def orientation_stream(between,valid,quality,yaw,*,floor=.1):
    """World-from-body rotations propagated with one sensor only."""
    T=np.asarray(between,float)
    valid=np.asarray(valid,bool)
    quality=np.asarray(quality,float)
    if T.ndim!=3 or T.shape[1:]!=(4,4):
        raise ValueError("Bad between pose shape")
    n=len(T)+1
    if valid.shape!=(n-1,) or quality.shape!=(n-1,):
        raise ValueError("Bad valid/quality shape")
    good=valid&(quality>=floor)
    R=np.empty((n,3,3))
    R[0]=_yaw_rotation(yaw)
    for i in range(n-1):
        R[i+1]=R[i]@T[i,:3,:3] if good[i] else R[i]
    return R,good


def make_heading_pairs(
    between,good,rotations,reference,k,*,first=WARMUP,last=None
):
    T=np.asarray(between,float)
    good=np.asarray(good,bool)
    R=np.asarray(rotations,float)
    ref=np.asarray(reference,float)
    n=len(ref)
    last=(n-END_MARGIN if last is None else min(last,n))
    if (T.shape!=(n-1,4,4) or good.shape!=(n-1,)
        or R.shape!=(n,3,3) or ref.shape!=(n,3)
        or not np.isfinite(ref).all()
        or not np.isfinite(T).all() or not np.isfinite(R).all()
        or k<1 or first<0 or last-first<=k):
        raise ValueError("Invalid heading diagnostic input")
    frames=[]
    a_vec=[]
    b_vec=[]
    for t in range(first,last-k):
        if not np.all(good[t:t+k]):
            continue
        C=np.eye(4)
        for j in range(t,t+k):
            C=C@T[j]
        a=(R[t]@C[:3,3])[:2]
        b=(ref[t+k]-ref[t])[:2]
        if min(float(np.linalg.norm(a)),float(np.linalg.norm(b))) < MIN_HORIZ_DIST_M:
            continue
        frames.append(t)
        a_vec.append(a)
        b_vec.append(b)
    if not frames:
        raise ValueError("No valid heading pairs")
    return np.asarray(frames,int),np.asarray(a_vec,float),np.asarray(b_vec,float)


def angles_from_vectors(a,b):
    a=np.asarray(a,float)
    b=np.asarray(b,float)
    if a.shape!=b.shape or a.ndim!=2 or a.shape[1]!=2:
        raise ValueError("Bad XY pair vectors")
    return np.arctan2(
        a[:,0]*b[:,1]-a[:,1]*b[:,0],
        np.einsum("ij,ij->i",a,b),
    )


def describe_xy_pair(a,b,*,oracle_correction_rad=None):
    """Angles are signed from odometry prediction to original GPS motion."""
    signed=angles_from_vectors(a,b)
    lengths=np.abs(np.linalg.norm(a,axis=1)-np.linalg.norm(b,axis=1))
    vector=np.linalg.norm(a-b,axis=1)
    mean_angle=np.arctan2(np.mean(np.sin(signed)),np.mean(np.cos(signed)))
    stats={
        "n":len(a),
        "signed_circular_mean_deg":float(np.rad2deg(mean_angle)),
        "signed_median_deg":float(np.median(np.rad2deg(signed))),
        "abs_angle_median_deg":float(np.median(np.abs(np.rad2deg(signed)))),
        "norm_mae_m":float(np.mean(lengths)),
        "vector_mae_m":float(np.mean(vector)),
    }
    if oracle_correction_rad is not None:
        ar=a@_yaw_rotation(oracle_correction_rad)[:2,:2].T
        stats["ORACLE_vector_mae_m"]=float(np.mean(np.linalg.norm(ar-b,axis=1)))
    return stats


def best_oracle_yaw(a,b):
    """Post-hoc whole-0027 clean-GPS heading: FOR DIAGNOSTICS ONLY."""
    a=np.asarray(a,float)
    b=np.asarray(b,float)
    if a.shape!=b.shape or a.ndim!=2 or a.shape[1]!=2:
        raise ValueError("Invalid oracle pair shapes")
    cross=float(np.sum(a[:,0]*b[:,1]-a[:,1]*b[:,0]))
    dot=float(np.sum(a*b))
    if np.hypot(cross,dot)<1e-8:
        raise ValueError("Oracle diagnostic yaw not identifiable")
    return float(np.arctan2(cross,dot))


def report(name,k,frames,a,b,*,warmup=WARMUP,last=None):
    last=len(frames)+warmup+END_MARGIN if last is None else last
    oracle=best_oracle_yaw(a,b)
    print(f"\n{name} k={k} n={len(frames)}; "
          f"POST-HOC OXTS ORACLE fixed yaw={np.rad2deg(oracle):+.2f} deg "
          "(DIAGNOSTIC ONLY, NEVER MODEL INPUT)")
    ranges=np.linspace(warmup,last,5,dtype=int)
    subsets=[("ALL",np.ones(len(frames),bool))]
    subsets+= [
        (f"Q{i+1}_{ranges[i]}..{ranges[i+1]-1}",
         (frames>=ranges[i])&(frames<ranges[i+1]))
        for i in range(4)
    ]
    for label,mask in subsets:
        if mask.sum()<10:
            print(f"  {label:<21s}: insufficient")
            continue
        stats=describe_xy_pair(a[mask],b[mask],
                              oracle_correction_rad=oracle)
        print(
            f"  {label:<21s} n={stats['n']:4d} "
            f"angle_signed_mean={stats['signed_circular_mean_deg']:+7.2f}deg "
            f"angle_signed_med={stats['signed_median_deg']:+7.2f}deg "
            f"|angle|_med={stats['abs_angle_median_deg']:6.2f}deg "
            f"norm_MAE={stats['norm_mae_m']:6.3f}m "
            f"vector_MAE={stats['vector_mae_m']:7.3f}m "
            f"oracle_rotated_vector_MAE={stats['ORACLE_vector_mae_m']:7.3f}m",
            flush=True
        )


def main():
    m=s6.load_measurements()
    n=len(m.gps_local)
    if n!=4544:
        raise ValueError(f"Expected 0027 n=4544, got {n}")
    with np.load(s6.BASE/"degraded_sensor_data.npz",
                 allow_pickle=False) as z:
        clean=np.asarray(z["clean_gps_local"],float)
    if clean.shape!=(n,3) or not np.isfinite(clean).all():
        raise ValueError("Invalid OXTS/GPS local reference")
    print("0027 READ-ONLY SIGNED HEADING ROOT-CAUSE AUDIT")
    print("Separate LiDAR and Camera yaw bootstrap and rotation propagation.")
    print("Initial yaw estimated only from first 63 DEGRADED GPS samples.")
    print("Reference CLEAN OXTS/GPS is used ONLY for post-hoc diagnostic errors.")
    print("Oracle rotation uses future reference and is NOT a valid method.")
    print("0020 / 0016 not read; no outputs written, no model/training/FGO.")
    for src in ("lidar","camera"):
        T=np.asarray(getattr(m,f"{src}_between"),float)
        v=np.asarray(getattr(m,f"{src}_valid"),bool)
        q=np.asarray(getattr(m,f"{src}_quality"),float)
        yaw=historical_yaw(m.gps_local,T,v,q)
        R,good=orientation_stream(T,v,q,yaw)
        print(f"\n==== {src.upper()} bootstrap_yaw={np.rad2deg(yaw):+.3f}deg "
              f"valid_pairs={good.sum()}/{n-1} ====")
        for k in HORIZONS:
            try:
                frames,a,b=make_heading_pairs(T,good,R,clean,k)
                report(src.upper(),k,frames,a,b,last=n-END_MARGIN)
            except ValueError as e:
                print(f"{src.upper()} k={k}: unavailable: {e}")
    print("\nDO NOT interpret post-hoc oracle orientation as a deployable calibration.")
    print("Decision: stable same-signed offset favors bootstrap/frame mismatch;"
          " time-changing signed offset favors odometry rotational drift.")


if __name__=="__main__":
    main()
