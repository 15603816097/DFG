from __future__ import annotations
import argparse,csv,json,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from experiments.cross_sequence.stage7_config import *
from src.factor_graph.sensorwise_reliability_graph import _pose3_from_matrix,_pose3_to_matrix,_imu_between_pose,_initial_trajectory,_gps_sigma,_camera_sigmas
from src.factor_graph.sensor_factor_config import FourSensorFactorConfig

def load_measurements(seq):
    od=sequence_out(seq);d=np.load(od/"sensor_data.npz",allow_pickle=False)
    l=np.load(od/"lidar_factor_data.npz",allow_pickle=False);c=np.load(od/"camera_factor_data.npz",allow_pickle=False)
    def pick(z,*names):
        for n in names:
            if n in z.files:return np.asarray(z[n])
        raise KeyError((z.files,names))
    return SimpleNamespace(
      gps_local=np.asarray(d["degraded_gps_local"],float),
      imu_acc=np.asarray(d["degraded_imu_acc"],float),
      imu_gyro=np.asarray(d["degraded_imu_gyro"],float),
      timestamps=np.asarray(d["timestamps_seconds"],float),
      lidar_between=pick(l,"between_measurement","body_between","between"),
      lidar_valid=pick(l,"valid","converged").astype(bool),
      lidar_quality=pick(l,"quality").astype(float),
      camera_between=pick(c,"between_measurement","body_between","between"),
      camera_valid=pick(c,"valid","converged","success").astype(bool),
      camera_quality=pick(c,"quality").astype(float))

def pred(seq,source,sensor,n):
    base=sequence_out(seq)/"predictions"/source
    r=np.loadtxt(base/f"{sensor}_reliability.txt").reshape(-1);u=np.loadtxt(base/f"{sensor}_uncertainty.txt").reshape(-1)
    if len(r)!=n or len(u)!=n:raise ValueError(f"{seq}/{source}/{sensor} length mismatch")
    return np.clip(r,0,1),np.clip(u,0,1)

def confidence(u):return float(np.clip(1-float(u),0,1))
def gps_sigma(cfg,r,u):
    p=_gps_sigma(cfg,"predictive",float(r));return GPS_FIXED+confidence(u)*(p-GPS_FIXED)
def imu_sigma(cfg,r,u):
    p=cfg.imu_rot_sigma_min+(1-float(r))*(cfg.imu_rot_sigma_max-cfg.imu_rot_sigma_min)
    return IMU_R0+confidence(u)*(p-IMU_R0)
def cam_sigma(cfg,r,u):
    pr,pt=_camera_sigmas(cfg,"predictive",float(r));c=confidence(u)
    return CAM_R0+c*(pr-CAM_R0),CAM_T0+c*(pt-CAM_T0)

def evaluate(seq,traj):
    gt=np.loadtxt(sequence_out(seq)/"ground_truth.txt",dtype=float)[:,:3];est=np.asarray(traj,float)[:,:3]
    n=min(len(gt),len(est));e=est[:n]-gt[:n];d3=np.linalg.norm(e,axis=1);d2=np.linalg.norm(e[:,:2],axis=1)
    return {"ATE3D":float(np.sqrt(np.mean(d3*d3))),"ATE2D":float(np.sqrt(np.mean(d2*d2))),
            "Mean3D":float(np.mean(d3)),"Max3D":float(np.max(d3)),"ZRMSE":float(np.sqrt(np.mean(e[:,2]**2)))}

def run_case(seq,name,gps_src=None,imu_src=None,cam_src=None):
    import gtsam
    m=load_measurements(seq);n=len(m.gps_local);cfg=FourSensorFactorConfig()
    cache={}
    for src in set(x for x in (gps_src,imu_src,cam_src) if x):
        for s in ("gps","imu","camera"):cache[(src,s)]=pred(seq,src,s,n)
    poses0=_initial_trajectory(gtsam,m);times=np.asarray(m.timestamps,float)
    graph=gtsam.NonlinearFactorGraph();initial=gtsam.Values()
    graph.add(gtsam.PriorFactorPose3(0,gtsam.Pose3(),gtsam.noiseModel.Diagonal.Sigmas(np.array([cfg.prior_rotation_sigma]*3+[cfg.prior_translation_sigma]*3))))
    for i,p in enumerate(poses0):initial.insert(i,p)
    for i in range(n):
        gs=GPS_FIXED if gps_src is None else gps_sigma(cfg,*[x[i] for x in cache[(gps_src,"gps")]])
        gp=gtsam.Pose3(gtsam.Rot3(),gtsam.Point3(*map(float,m.gps_local[i,:3])))
        graph.add(gtsam.PriorFactorPose3(i,gp,gtsam.noiseModel.Diagonal.Sigmas(np.array([1e6]*3+[gs]*3))))
        if i==n-1:continue
        dt=float(times[i+1]-times[i]);dt=dt if np.isfinite(dt) and 0<dt<=1 else .1
        ir=IMU_R0 if imu_src is None else imu_sigma(cfg,*[x[i+1] for x in cache[(imu_src,"imu")]])
        graph.add(gtsam.BetweenFactorPose3(i,i+1,_imu_between_pose(gtsam,m.imu_gyro[i],dt),
                  gtsam.noiseModel.Diagonal.Sigmas(np.array([ir]*3+[cfg.imu_translation_sigma]*3))))
        if bool(m.lidar_valid[i]):
            graph.add(gtsam.BetweenFactorPose3(i,i+1,_pose3_from_matrix(gtsam,m.lidar_between[i]),
                      gtsam.noiseModel.Diagonal.Sigmas(np.array([LIDAR_R0]*3+[LIDAR_T0]*3))))
        if bool(m.camera_valid[i]):
            cr,ct=(CAM_R0,CAM_T0) if cam_src is None else cam_sigma(cfg,*[x[i+1] for x in cache[(cam_src,"camera")]])
            graph.add(gtsam.BetweenFactorPose3(i,i+1,_pose3_from_matrix(gtsam,m.camera_between[i]),
                      gtsam.noiseModel.Diagonal.Sigmas(np.array([cr]*3+[ct]*3))))
    params=gtsam.LevenbergMarquardtParams();params.setMaxIterations(100);params.setRelativeErrorTol(1e-7)
    res=gtsam.LevenbergMarquardtOptimizer(graph,initial,params).optimize()
    poses=np.stack([_pose3_to_matrix(res.atPose3(i)) for i in range(n)]);traj=poses[:,:3,3]
    od=sequence_out(seq)/"factor_graph"/name;od.mkdir(parents=True,exist_ok=True)
    np.save(od/"poses.npy",poses);np.savetxt(od/"trajectory.txt",traj,fmt="%.9f")
    met=evaluate(seq,traj);(od/"metrics.json").write_text(json.dumps(met,indent=2));print(seq,name,met);return met

CASES=[
 ("S0_fixed",None,None,None),
 ("S1_v5_gru","v5","v5","v5"),
 ("S2_v6_all_mamba","v6","v6","v6"),
 ("S3_d5_hybrid","v6","v5","v6"),
]

def main():
    p=argparse.ArgumentParser();p.add_argument("--sequence",required=True,choices=TEST_SEQUENCES);a=p.parse_args()
    rows=[]
    for name,g,i,c in CASES:
        met=run_case(a.sequence,name,g,i,c);rows.append({"case":name,**met})
    fixed=rows[0]["ATE3D"]
    for r in rows:r["vs_fixed_pct"]=(fixed-r["ATE3D"])/fixed*100
    out=sequence_out(a.sequence)/"stage7_results.csv"
    with out.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print("Saved",out)
if __name__=="__main__":main()
