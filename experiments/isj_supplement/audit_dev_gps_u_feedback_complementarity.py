from __future__ import annotations

"""E2A READ-ONLY: does GPS evidential U ADD synthetic-fault information?

Runs on development KITTI 2011_10_03_drive_0027_sync ONLY.
Frozen 0027 GRU V5 / Mamba V6 outputs (r,U) + the pre-existing
5-step GPS/LiDAR displacement-magnitude mismatch ("raw_D").
Fault label: original synthetic GPS injection >=5m against original
OXTS GPS. This is NOT a real fault label or independent GT.

For each original model-specific chronological split, same exact valid
frame mask is used for every score. Logistic probes are trained ONLY on
TRAIN with fixed L2=1, fixed features and train-only normalization,
then evaluated on VAL and TEST without refitting/selection:
  (1-r), raw_D, U (standalone, no fitting)
  R=(1-r) [train-only linear probe]
  R+D [train-only logistic probe]
  R+D+U [train-only logistic probe]
Logistic probes are *diagnostics*, NOT the manuscript method; their
AUC differences do not prove independent U contribution or location
improvement. Correlation and temporal dependence can mislead.
No optimizer, no training of GRU/Mamba, no writes, no 0016/0020 access.
"""

import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from experiments.factor_graph import run_stage6_v6_mamba_factor_graph_ablation as s6
from experiments.isj_supplement.audit_dev_uncertainty_risk_by_split import (
    split_ranges, rank_auc, finite_corr,
)
from src.reliability.cross_modal_consistency import (
    CrossModalSettings, measurement_consistency,
)

MODELS=(("GRU_V5","v5"),("MAMBA_V6","v6"))
FAULT_THRESHOLD_M=5.0
PROBE_L2=1.0
FEATURE_SETS={
    "R_ONLY":(0,),
    "R_PLUS_D":(0,1),
    "R_PLUS_D_PLUS_U":(0,1,2),
}


def _sigmoid(z):
    z=np.asarray(z,dtype=float)
    return np.exp(-np.logaddexp(0.,-z))


def fit_logit_probe(x,y,l2=PROBE_L2):
    """One fixed, low-capacity L2 logistic probe on TRAIN data only."""
    x=np.asarray(x,float)
    y=np.asarray(y,float).reshape(-1)
    if (x.ndim!=2 or len(x)!=len(y) or len(y)<20 or x.shape[1]<1
        or not np.isfinite(x).all() or not np.isfinite(y).all()
        or np.any((y!=0)&(y!=1)) or len(np.unique(y))!=2):
        raise ValueError("Bad/degenerate TRAIN probe data")
    if not np.isfinite(l2) or l2<=0:
        raise ValueError("l2 must be positive")
    mean=x.mean(axis=0)
    sd=x.std(axis=0)
    sd=np.where(sd>1e-9,sd,1.)
    z=np.column_stack([np.ones(len(x)),(x-mean)/sd])
    b=np.zeros(z.shape[1])
    rate=np.clip(y.mean(),1e-5,1-1e-5)
    b[0]=np.log(rate/(1-rate))
    reg=np.diag(np.r_[0.,np.full(x.shape[1],l2)])
    for _ in range(100):
        p=_sigmoid(z@b)
        grad=z.T@(p-y)+reg@b
        w=np.maximum(p*(1-p),1e-10)
        hess=z.T@(w[:,None]*z)+reg
        delta=np.linalg.solve(hess,grad)
        # Fixed Newton steps, no validation-set fitting or tuning.
        b-=delta
        if np.max(np.abs(delta))<1e-8:
            break
    if not np.isfinite(b).all():
        raise ValueError("Nonfinite logistic coefficients")
    return {"mean":mean,"sd":sd,"coef":b}


def apply_probe(fitted,x):
    x=np.asarray(x,float)
    if (x.ndim!=2 or x.shape[1]!=len(fitted["mean"])
        or not np.isfinite(x).all()):
        raise ValueError("Invalid held-out probe input")
    z=(x-fitted["mean"])/fitted["sd"]
    return _sigmoid(fitted["coef"][0]+z@fitted["coef"][1:])


def format_score(score,y):
    auc=rank_auc(score,y)
    return f"{auc:.4f}" if np.isfinite(auc) else "N/A(one class)"


def diagnostics_for(modelname,source,n,label,raw,valid):
    r,u=s6.load_prediction(source,"gps",n)
    if r.shape!=(n,) or u.shape!=(n,):
        raise ValueError("Prediction length mismatch")
    splits=split_ranges(modelname,n)
    r_risk=1.-r
    # Use log1p for fixed scaling of heavy-tailed physical mismatch.
    # This is monotonic and cannot change the raw_D rank AUC.
    D=np.log1p(np.maximum(raw,0.))
    X=np.column_stack((r_risk,D,u))
    names=("1-r","log1p(raw_D)","U")
    available=np.asarray(valid,bool)&np.isfinite(X).all(axis=1)
    start,end=splits["TRAIN"]
    train=np.arange(start,end)
    train=train[available[train]]
    if len(train)<100:
        raise ValueError(f"{modelname}: too few valid TRAIN frames")
    print(f"\n{'='*100}\n{modelname}: {source} "
          f"split boundaries={splits}\n"
          "Matched mask: model prediction valid AND original "
          "LiDAR/GPS 5-step score valid; 0027 ONLY")
    print("Label: injected GPS error >=5m versus original OXTS GPS")
    fitted={}
    for name,idx in FEATURE_SETS.items():
        fitted[name]=fit_logit_probe(X[train][:,idx],label[train])
        print(f"TRAIN-FIT {name:<18s} coefficients="
              f"{np.array2string(fitted[name]['coef'],precision=4)}")
    for subset,(start,end) in splits.items():
        ids=np.arange(start,end)
        ids=ids[available[ids]]
        if len(ids)<50:
            raise ValueError(f"Too few valid frames: {subset}")
        y=label[ids]
        xx=X[ids]
        print(f"\n  {subset}: n={len(ids)} "
              f"positive={int(y.sum())} "
              f"fault_fraction={np.mean(y):.2%}")
        print("  stand-alone fault detection AUC: "
              +"  ".join(f"{a}={format_score(xx[:,j],y)}"
                         for j,a in enumerate(names)))
        for name,idx in FEATURE_SETS.items():
            probe=apply_probe(fitted[name],xx[:,idx])
            print(f"  {name:<18s} TRAIN-FIT AUROC={format_score(probe,y)}")
        print("  pairwise: "
              f"corr(U,1-r)={finite_corr(xx[:,2],xx[:,0]):+.4f} "
              f"corr(U,logD)={finite_corr(xx[:,2],xx[:,1]):+.4f}")
        # For systematic added-value analysis, stratify the 0027 internal
        # TEST frames by r-risk, without fitting anything to their labels.
        # The printed r-risk quartiles are descriptive, not thresholds
        # for a deployed gating system.
        if subset=="TEST":
            edges=np.quantile(xx[:,0],[0,.25,.5,.75,1.])
            for j in range(4):
                selected=(xx[:,0]>=edges[j]) & (
                    xx[:,0]<=edges[j+1] if j==3
                    else xx[:,0]<edges[j+1]
                )
                if selected.sum()<20:
                    print(f"    r-risk quartile {j+1}: insufficient")
                    continue
                print(f"    r-risk quartile {j+1}: "
                      f"n={selected.sum()} fault={np.mean(y[selected]):.2%} "
                      f"U_AUC={format_score(xx[selected,2],y[selected])} "
                      f"D_AUC={format_score(xx[selected,1],y[selected])}")


def main():
    m=s6.load_measurements()
    n=len(m.gps_local)
    if n!=4544:
        raise ValueError(f"0027 expected n=4544, actual n={n}")
    consistency=measurement_consistency(
        gps_local=m.gps_local, imu_gyro=m.imu_gyro,
        timestamps_seconds=np.asarray(m.timestamps,float),
        lidar_between=m.lidar_between, lidar_valid=m.lidar_valid,
        lidar_quality=m.lidar_quality, settings=CrossModalSettings(),
    )
    # Only read synthetic injection labels after the input-only feedback.
    with np.load(s6.BASE/"degraded_sensor_data.npz",
                 allow_pickle=False) as data:
        clean=np.asarray(data["clean_gps_local"],float)
    if clean.shape!=m.gps_local.shape or not np.isfinite(clean).all():
        raise ValueError("Invalid original OXTS-based GPS reference")
    injected=np.linalg.norm(m.gps_local-clean,axis=1)
    label=injected>=FAULT_THRESHOLD_M
    valid=np.asarray(consistency["gps_valid"],bool)
    raw=np.asarray(consistency["gps_nis"],float)
    print("0027 GPS FEEDBACK + EVIDENTIAL U COMPLEMENTARITY (READ ONLY)")
    print("Fixed model-specific chronological split; VAL/TEST not fit")
    print("Fault labels based on synthetic injection and original OXTS GPS")
    print("raw_D is heuristic (not calibrated NIS); U is not GPS variance")
    print("Logistic probes measure predictability, not localization quality")
    print("No files written; 0016 and 0020 not accessed")
    for name,src in MODELS:
        diagnostics_for(name,src,n,label,raw,valid)
    print("\nDONE. Inspect TEST R+D vs R+D+U, but do not optimize")
    print("the formula or threshold on 0016/0020. This is still 0027.")


if __name__=="__main__":
    main()
