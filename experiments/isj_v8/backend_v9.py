"""V9.2 engineering adapter: arbitrary length H0 and GPS-only Huber C8.

No test release entrypoint. Existing C8 and training lock are immutable.
Huber k=sqrt(chi2_3(0.95)) in block-whitened 3D GPS residual units,
chosen before any V9 cross-sequence scores (not a calibrated test).
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

from experiments.isj_v8.backend_c8_numeric import (
    build_c8_graphs, verify_objective, directional_audit, yaw_aligned_seed,
    poses_to_values,
)
from experiments.isj_v8.solver_health_v8 import solve_v8_health
from experiments.isj_v7.cross_sequence_v7 import _positions
from experiments.isj_v8.run_c8_weight_controls import shifts

ROOT = Path(__file__).resolve().parents[2]
LOCK = ROOT / "results/isj_v9/train_h0_method_lock_v1/method_lock.json"
LOCK_SHA = "eb046a5505a22bee9af6c2a89c6d11225fcb38cab47191a82a652d1cdc8e4e64"
HUBER_K = float(np.sqrt(7.814727903251179))
COLD = 63


def load_lock():
    import hashlib
    data = LOCK.read_bytes()
    if hashlib.sha256(data).hexdigest() != LOCK_SHA:
        raise ValueError("V9.1 method lock SHA mismatch")
    lock = json.loads(data)
    if (lock["status"] != "TRAIN_METHOD_LOCKED_TEST_PROTOCOL_PENDING"
            or lock["test_release"] is not False
            or lock["seeds"] != [43,44,45]
            or len(lock["calibrated_h0_mappings"]) != 9
            or lock["inference"]["cold_start_frames"] != COLD
            or lock["calibration"]["cap_information_ratio"] != 4.):
        raise ValueError("invalid frozen method lock")
    return lock


def map_h0(anchors, reliability, n, fitted, fixed):
    """Fixed first 63 frames; strictly chronological scores thereafter."""
    a, r = np.asarray(anchors), np.asarray(reliability, float)
    fixed = np.asarray(fixed,float)
    centers = np.asarray(fitted["centers"],float)
    bins = np.asarray(fitted["sigma_bins_xyz"],float)
    if (not isinstance(n,int) or n <= COLD
            or not np.array_equal(a,np.arange(COLD,n))
            or r.shape != (n-COLD,) or not np.isfinite(r).all()
            or np.any((r<0)|(r>1)) or fixed.shape != (3,)
            or not np.isfinite(fixed).all() or np.any(fixed<=0)
            or centers.shape != (5,) or not np.isfinite(centers).all()
            or np.any((centers<0)|(centers>1)) or np.any(np.diff(centers)<=0)
            or bins.shape != (5,3) or not np.isfinite(bins).all()
            or np.any(bins<=0) or np.any(np.diff(bins,axis=0)>1e-10)):
        raise ValueError("invalid H0 scores, anchors or mapping")
    result = np.tile(fixed,(n,1))
    result[a.astype(int)] = np.maximum(
        np.column_stack([np.interp(r,centers,bins[:,j]) for j in range(3)]),
        fixed/2.)
    if np.any((fixed/result)**2 > 4.+1e-10):
        raise ValueError("Cap4 violation")
    return result


def build_graphs(gtsam, legacy, gps, sigma, *, huber=False):
    """Apply a Huber kernel to GPS only; preserve all other C8 factors."""
    native, corrected, keepalive, count = build_c8_graphs(gtsam,legacy,gps,sigma)
    if not huber:
        return native,corrected,keepalive,count
    a,b = gtsam.NonlinearFactorGraph(),gtsam.NonlinearFactorGraph()
    seen=set()
    for j in range(native.size()):
        f=native.at(j)
        if isinstance(f,gtsam.GPSFactor):
            k=int(f.keys()[0])
            if k in seen:
                raise ValueError("duplicate GPS key")
            seen.add(k)
            noise=gtsam.noiseModel.Robust.Create(
                gtsam.noiseModel.mEstimator.Huber.Create(HUBER_K),
                gtsam.noiseModel.Diagonal.Sigmas(np.asarray(sigma[k],float)))
            f=gtsam.GPSFactor(k,np.asarray(gps[k],float),noise)
            a.add(f)
            b.add(f)
        else:
            a.add(f)
            b.add(corrected.at(j))
    if seen != set(range(len(gps))) or a.size()!=native.size() or b.size()!=corrected.size():
        raise ValueError("robust GPS factor count drift")
    return a,b,keepalive,count


def solve_uniform(gtsam,native,corrected,seed,n):
    """Unchanged V8.4.4 + V8.4.5 policy; no reference used."""
    verify_objective(native,corrected,seed)
    warm=solve_v8_health(native,seed,n)
    v=warm["values"]
    j0=verify_objective(native,corrected,v)
    audit0=directional_audit(gtsam,native,corrected,v,n)
    if not audit0["passed"]:
        raise RuntimeError("warm derivative check failed")
    middle=solve_v8_health(corrected,v,n)
    v=middle["values"]
    j1=verify_objective(native,corrected,v)
    audit1=directional_audit(gtsam,native,corrected,v,n)
    if not audit1["passed"] or j1>j0+1e-8*max(j0,1.):
        raise RuntimeError("corrected continuation check failed")
    passes=[]
    for i in range(2):
        before=_positions(gtsam,v,n)
        jb=verify_objective(native,corrected,v)
        params=gtsam.LevenbergMarquardtParams()
        params.setMaxIterations(100)
        params.setRelativeErrorTol(1e-12)
        params.setAbsoluteErrorTol(1e-10)
        v=gtsam.LevenbergMarquardtOptimizer(corrected,v,params).optimize()
        ja=verify_objective(native,corrected,v)
        after=_positions(gtsam,v,n)
        if ja>jb+1e-8*max(jb,1.) or not np.isfinite(after).all():
            raise RuntimeError("invalid refinement")
        dt,dr=shifts(before,after)
        passes.append(dict(pass_index=i+1,objective_before=jb,objective_after=ja,
                           relative_gain=(jb-ja)/max(jb,1.),
                           translation_shift_m=dt,rotation_shift_rad=dr))
    audit=directional_audit(gtsam,native,corrected,v,n)
    last=passes[-1]
    stable=bool(abs(last["relative_gain"])<=1e-6 and
                last["translation_shift_m"]<=.01 and last["rotation_shift_rad"]<=.01)
    if not audit["passed"]:
        raise RuntimeError("final derivative check failed")
    return dict(native_status=warm["status"],corrected_status=middle["status"],
                refinement=passes,derivative_audit=audit,last_step_stable=stable,
                stationarity_certified=False,poses=_positions(gtsam,v,n))


def dev_smoke():
    """One 0027 engineering Huber solve, no new position score/reference."""
    import gtsam
    import importlib.metadata
    from experiments.isj_v8.run_c8_numeric_replay import preflight
    from experiments.isj_v8.evaluate_c8_refined_dev import PREVIOUS,verify_sources
    from experiments.isj_v7.run_gps_covariance_pilot import DEGRADED,BEGIN,END,subgraph_measurements
    from src.factor_graph.degraded_measurements import load_degraded_four_sensor_measurements
    from src.factor_graph.four_sensor_graph import run_four_sensor_factor_graph
    from experiments.isj_v8.lock_v9_train_methods import source_check
    if importlib.metadata.version("gtsam")!="4.2":
        raise ValueError("GTSAM version drift")
    lock=load_lock()
    source_check()
    sigmas=preflight()
    fixed=np.asarray(lock["original_oracle_mapping"]["sigma_fixed_xyz"],float)
    sigma=sigmas["axis_fixed"]
    if not np.array_equal(sigma,np.tile(fixed,(680,1))):
        raise ValueError("fixed sigma drift")
    m=subgraph_measurements(load_degraded_four_sensor_measurements(DEGRADED),
                            begin=BEGIN,end=END)
    gps=np.asarray(m.gps_local,float)
    with np.load(PREVIOUS/"poses.npz",allow_pickle=False) as z:
        seed=poses_to_values(gtsam,z["common_aligned_seed"])
    legacy,initial,counts=run_four_sensor_factor_graph(
        m,"NO_GRAPH_WRITES",mode="fixed",gps_sigmas_override=sigma,build_graph_only=True)
    if counts!={"gps":680,"imu":679,"lidar":679,"camera":670} or legacy.size()!=2709:
        raise ValueError("DEV graph drift")
    native,corrected,keepalive,count=build_graphs(gtsam,legacy,gps,sigma,huber=True)
    if count!=2028 or len(keepalive)!=2029:
        raise ValueError("relative factors drift")
    print("HUBER_BLOCK_K",HUBER_K,flush=True)
    res=solve_uniform(gtsam,native,corrected,seed,680)
    res.pop("poses")
    print("DEV_HUBER_NUMERICAL_RESULT",json.dumps(res,allow_nan=False),flush=True)
    verify_sources()
    if not res["last_step_stable"]:
        raise RuntimeError("DEV Huber numerical smoke not stable")
    print("V9_2_DEV_HUBER_SMOKE_OK NO_DEV_SCORE NO_TEST_RELEASE NO_WRITES")


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dev-smoke",required=True,action="store_true")
    p.parse_args()
    dev_smoke()
