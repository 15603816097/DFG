from __future__ import annotations
import os, sys, csv
from pathlib import Path
from types import SimpleNamespace
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

from src.factor_graph.degraded_measurements import load_degraded_four_sensor_measurements
from src.factor_graph.sensorwise_reliability_graph import (
    ReliabilitySource,_timestamp_seconds,_pose3_from_matrix,_pose3_to_matrix,
    _imu_between_pose,_initial_trajectory,_gps_sigma,_camera_sigmas
)
from src.factor_graph.sensor_factor_config import FourSensorFactorConfig

BASE=ROOT/"results/degraded_four_sensor_measurements"
PHYS=ROOT/"results/lidar_physical_oracle_benchmark/physical_lidar_factor_data.npz"
V4=ROOT/"results/sensor_specific_reliability_v4"
LIDAR_MAP=ROOT/"results/lidar_covariance_calibration_v1/mapping/lidar_dynamic_mapping.npz"
GT=ROOT/"results/ground_truth/trajectory.txt"
OUT=ROOT/"results/stage3_confidence_aware_mapping"

# Frozen nominal anchors from 0027 calibration.
LIDAR_T0=.15
LIDAR_R0=.012857142857142857
IMU_R0=.03
CAM_T0=.45
CAM_R0=.04

def load_pred(sensor,n):
    f=V4/f"{sensor}_predictive_prior_target_aligned.txt"
    x=np.loadtxt(f,dtype=float).reshape(-1)
    if len(x)<n: raise ValueError(f"{f}: {len(x)} < {n}")
    return np.clip(x[:n],0,1)

def load_measurements():
    b=load_degraded_four_sensor_measurements(BASE)
    m=SimpleNamespace(**vars(b))
    d=np.load(PHYS,allow_pickle=False)
    m.lidar_between=np.asarray(d["body_between"],float)
    m.lidar_quality=np.asarray(d["quality"],float)
    m.lidar_valid=np.asarray(d["converged"],bool)
    return m

def evaluate(traj):
    gt=np.loadtxt(GT,dtype=float)
    est=np.asarray(traj,float)
    # Keep the same convention used by the existing experiments: first 3 columns are xyz
    if gt.ndim!=2 or est.ndim!=2: raise ValueError("trajectory must be 2-D")
    if gt.shape[1]>3: gt=gt[:,-3:]
    if est.shape[1]>3: est=est[:,-3:]
    n=min(len(gt),len(est)); e=est[:n,:3]-gt[:n,:3]
    d3=np.linalg.norm(e,axis=1); d2=np.linalg.norm(e[:,:2],axis=1)
    return dict(ATE3D=float(np.sqrt(np.mean(d3*d3))),
                ATE2D=float(np.sqrt(np.mean(d2*d2))),
                Mean3D=float(np.mean(d3)),Max3D=float(np.max(d3)),
                ZRMSE=float(np.sqrt(np.mean(e[:,2]**2))))

def confidence_from_reliability(r, strength):
    # Conservative confidence-aware gate:
    # extreme reliability predictions get more authority;
    # ambiguous r≈0.5 automatically falls back toward nominal covariance.
    confidence=np.clip(2.0*np.abs(r-0.5),0.0,1.0)
    risk=1.0-r
    return np.clip(strength*confidence*risk*risk,0.0,1.0)

def run_case(name, measurements, strengths):
    import gtsam
    n=len(measurements.gps_local)
    gps=load_pred("gps",n); imu=load_pred("imu",n); cam=load_pred("camera",n)
    lm=np.load(LIDAR_MAP,allow_pickle=False)
    lidar_risk=np.asarray(lm["risk_score_pair"],float).reshape(-1)
    if len(lidar_risk)!=n-1: raise ValueError("LiDAR risk length mismatch")

    cfg=FourSensorFactorConfig()
    poses0=_initial_trajectory(gtsam,measurements)
    times=_timestamp_seconds(measurements.timestamps)
    graph=gtsam.NonlinearFactorGraph(); initial=gtsam.Values()
    prior=gtsam.noiseModel.Diagonal.Sigmas(np.array(
        [cfg.prior_rotation_sigma]*3+[cfg.prior_translation_sigma]*3,float))
    graph.add(gtsam.PriorFactorPose3(0,gtsam.Pose3(),prior))
    for i,p in enumerate(poses0): initial.insert(i,p)

    stats={k:[] for k in ["gps","imu","camera_t","camera_r","lidar_t","lidar_r"]}
    for i in range(n):
        # GPS: interpolate between nominal and existing predictive sigma.
        pred_sigma=_gps_sigma(cfg,"predictive",gps[i])
        a=confidence_from_reliability(gps[i],strengths["gps"])
        gs=cfg.gps_fixed_sigma+a*(pred_sigma-cfg.gps_fixed_sigma)
        stats["gps"].append(gs)
        noise=gtsam.noiseModel.Diagonal.Sigmas(np.array([1e6]*3+[gs]*3,float))
        gp=gtsam.Pose3(gtsam.Rot3(),gtsam.Point3(*map(float,measurements.gps_local[i])))
        graph.add(gtsam.PriorFactorPose3(i,gp,noise))
        if i>=n-1: continue

        dt=float(times[i+1]-times[i]) if len(times)==n else .1
        if not np.isfinite(dt) or dt<=0 or dt>1: dt=.1

        # IMU: reliability controls authority, but mapping is anchored at nominal.
        # low reliability -> larger rotation sigma; high reliability -> slightly smaller.
        ir=imu[i+1]
        ic=np.clip(strengths["imu"]*2*abs(ir-.5),0,1)
        target=cfg.imu_rot_sigma_min+(1-ir)*(cfg.imu_rot_sigma_max-cfg.imu_rot_sigma_min)
        isig=IMU_R0+ic*(target-IMU_R0)
        stats["imu"].append(isig)
        inoise=gtsam.noiseModel.Diagonal.Sigmas(
            np.array([isig]*3+[cfg.imu_translation_sigma]*3,float))
        graph.add(gtsam.BetweenFactorPose3(
            i,i+1,_imu_between_pose(gtsam,measurements.imu_gyro[i],dt),inoise))

        if bool(measurements.lidar_valid[i]):
            # Dedicated LiDAR risk; confidence = distance from ambiguous risk 0.5.
            lr=np.clip(lidar_risk[i],0,1)
            lc=np.clip(strengths["lidar"]*2*abs(lr-.5),0,1)
            scale=1.0+lc*(lr*lr)  # max 2x, automatically near nominal when uncertain
            st=LIDAR_T0*scale; sr=LIDAR_R0*scale
            stats["lidar_t"].append(st); stats["lidar_r"].append(sr)
            lnoise=gtsam.noiseModel.Diagonal.Sigmas(np.array([sr]*3+[st]*3,float))
            graph.add(gtsam.BetweenFactorPose3(
                i,i+1,_pose3_from_matrix(gtsam,measurements.lidar_between[i]),lnoise))

        if bool(measurements.camera_valid[i]):
            cr=cam[i+1]
            cc=np.clip(strengths["camera"]*2*abs(cr-.5),0,1)
            pr,pt=_camera_sigmas(cfg,"predictive",cr)
            sr=CAM_R0+cc*(pr-CAM_R0); st=CAM_T0+cc*(pt-CAM_T0)
            stats["camera_t"].append(st); stats["camera_r"].append(sr)
            cnoise=gtsam.noiseModel.Diagonal.Sigmas(np.array([sr]*3+[st]*3,float))
            graph.add(gtsam.BetweenFactorPose3(
                i,i+1,_pose3_from_matrix(gtsam,measurements.camera_between[i]),cnoise))

    print(name,"factors:",graph.size())
    opt=gtsam.LevenbergMarquardtParams(); opt.setMaxIterations(100); opt.setRelativeErrorTol(1e-7)
    result=gtsam.LevenbergMarquardtOptimizer(graph,initial,opt).optimize()
    poses=np.stack([_pose3_to_matrix(result.atPose3(i)) for i in range(n)])
    traj=poses[:,:3,3]
    od=OUT/name; od.mkdir(parents=True,exist_ok=True)
    np.save(od/"poses.npy",poses); np.savetxt(od/"trajectory.txt",traj,fmt="%.9f")
    for k,v in stats.items():
        a=np.asarray(v,float)
        if len(a): print(f"{k:10s} sigma min/max/mean {a.min():.6f}/{a.max():.6f}/{a.mean():.6f}")
    return evaluate(traj)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    m=load_measurements()
    # This is a diagnostic strength sweep, NOT a new network training.
    # It tests whether confidence-aware fallback can make all four sensors dynamic safely.
    cases=[
        ("C0_all_fixed_authority",dict(gps=0.,imu=0.,camera=0.,lidar=0.)),
        ("C1_conf025",dict(gps=.25,imu=.25,camera=.25,lidar=.25)),
        ("C2_conf050",dict(gps=.50,imu=.50,camera=.50,lidar=.50)),
        ("C3_conf075",dict(gps=.75,imu=.75,camera=.75,lidar=.75)),
        ("C4_conf100",dict(gps=1.,imu=1.,camera=1.,lidar=1.)),
    ]
    rows=[]
    for name,s in cases:
        print("\n"+"="*100+"\n"+name+"\n"+"="*100)
        met=run_case(name,m,s); rows.append(dict(case=name,**s,**met))
        print(name,met)
    with open(OUT/"confidence_aware_sweep.csv","w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    rows.sort(key=lambda x:x["ATE3D"])
    print("\nFINAL RANKING")
    for i,r in enumerate(rows,1): print(f'{i:02d}. {r["case"]}: ATE3D={r["ATE3D"]:.6f}, ATE2D={r["ATE2D"]:.6f}, ZRMSE={r["ZRMSE"]:.6f}')
    print("CSV:",OUT/"confidence_aware_sweep.csv")

if __name__=="__main__":
    main()
