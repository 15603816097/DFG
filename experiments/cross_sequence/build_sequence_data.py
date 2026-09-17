from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.cross_sequence.stage7_config import (
    DATE_DIR, DEV_PLAN_DIR, OUT_ROOT, TEST_SEQUENCES, sequence_path, sequence_out
)
from src.loader.gps_loader import GPSLoader
from src.loader.imu_loader import IMULoader
from src.loader.lidar_loader import LidarLoader
from src.loader.camera_loader import CameraLoader
from src.factor_graph.four_sensor_graph import gps_geodetic_to_local
from src.multisensor_reliability.config import MultiSensorReliabilityConfig
from src.multisensor_reliability.degradation.balanced_plans import load_balanced_plan
from src.multisensor_reliability.degradation.apply import corrupt_imu, corrupt_camera
from src.multisensor_reliability.features import gps_features, imu_features, camera_features
from src.degradation.lidar_physical_degradation import BenchmarkConfig, build_schedule, apply_degradation
from src.preprocessing.calibration import KITTICalibration
from src.odometry.body_frame_converter import BodyFrameConverter
from src.odometry.lidar_odometry import LidarOdometry
from src.odometry.stereo_visual_odometry import StereoVisualOdometry
from src.odometry.relative_factor_data import make_factor_data

def _lidar(x):
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, dict):
        for k in ("points","point_cloud","lidar","data"):
            if k in x:
                return np.asarray(x[k])
    raise KeyError("Cannot extract LiDAR points")

def _camera(x):
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, dict):
        for k in ("image","rgb","data"):
            if k in x:
                return np.asarray(x[k])
    raise KeyError("Cannot extract camera image")

def _ts_seconds(ts):
    if not ts:
        return np.array([], float)
    if hasattr(ts[0], "timestamp"):
        return np.asarray([float(t.timestamp()) for t in ts], float)
    return np.asarray(ts, float)

def _resample_plan_vector(x, n):
    x = np.asarray(x)
    if len(x) == n:
        return x.copy()
    if len(x) < 1:
        raise ValueError("empty development degradation plan")
    # Frozen normalized-time mapping. No target-sequence tuning.
    idx = np.rint(np.linspace(0, len(x)-1, n)).astype(int)
    return x[idx]

def normalized_plan(dev_plan, n):
    out = {"n_frames": int(n), "seed": int(dev_plan["seed"]), "severity": {}, "mode": {}}
    for s in ("gps","imu","lidar","camera"):
        out["severity"][s] = _resample_plan_vector(dev_plan["severity"][s], n)
        out["mode"][s] = _resample_plan_vector(dev_plan["mode"][s], n)
    return out

def corrupt_local_gps(p, severity, mode, frame_id, seed):
    p=np.asarray(p,float).copy(); s=float(np.clip(severity,0,1))
    if s<=0: return p
    rng=np.random.default_rng(int(seed)+int(frame_id)*1009+31)
    if int(mode)==2:
        d=rng.normal(size=3); d/=max(np.linalg.norm(d),1e-12); d[2]*=.25
        p += d*(10+18*s)
    else:
        noise=rng.normal(0,.30+4*s,3); noise[2]*=.5
        bias=np.array([6*s*np.sin(frame_id/120),4*s*np.cos(frame_id/155),.8*s*np.sin(frame_id/190)])
        p += noise+bias
    return p

def lidar_quality(r):
    ok=bool(r.converged and np.isfinite(r.rmse) and r.correspondences>=80)
    if not ok: return False,0.0
    q=.50*np.clip(r.fitness,0,1)+.25*np.exp(-r.rmse/.75)+.25*np.clip(r.correspondences/6000,0,1)
    return True,float(np.clip(q,0,1))

def camera_quality(r):
    ok=bool(r.success and np.isfinite(r.reprojection_error) and r.inliers>=20 and r.matches>=30)
    if not ok: return False,0.0
    ratio=r.inliers/max(r.matches,1)
    q=.50*np.clip(ratio,0,1)+.20*np.clip(r.inliers/1200,0,1)+.30*np.exp(-r.reprojection_error/2)
    return True,float(np.clip(q,0,1))

def build_one(sequence_name: str):
    if sequence_name not in TEST_SEQUENCES:
        raise ValueError(f"Stage-7 target must be one of {TEST_SEQUENCES}; got {sequence_name}")
    seq=sequence_path(sequence_name)
    out=sequence_out(sequence_name)
    out.mkdir(parents=True,exist_ok=True)

    gl,il,ll,cl=GPSLoader(seq),IMULoader(seq),LidarLoader(seq),CameraLoader(seq,camera_id="image_02")
    left=sorted((seq/"image_02"/"data").glob("*.png"))
    right=sorted((seq/"image_03"/"data").glob("*.png"))
    n=min(len(gl),len(il),len(ll),len(cl),len(left),len(right))
    if n < 100:
        raise RuntimeError(f"{sequence_name}: suspicious frame count {n}")

    dev=load_balanced_plan(DEV_PLAN_DIR)
    plan=normalized_plan(dev,n)
    seed=int(plan["seed"])
    cfg=MultiSensorReliabilityConfig()

    lla=np.asarray([gl[i]["position"] for i in range(n)],np.float64)
    clean=gps_geodetic_to_local(lla)
    dgps=np.zeros_like(clean)
    dacc=np.zeros((n,3),float); dgyro=np.zeros((n,3),float); timestamps=[]
    cam_feat=[]
    for i in range(n):
        dgps[i]=corrupt_local_gps(clean[i],plan["severity"]["gps"][i],plan["mode"]["gps"][i],i,seed)
        a,g=corrupt_imu(np.asarray(il[i]["acceleration"],float),np.asarray(il[i]["angular_velocity"],float),
                        plan["severity"]["imu"][i],plan["mode"]["imu"][i],i,seed=seed)
        dacc[i]=a; dgyro[i]=g; timestamps.append(il[i]["timestamp"])
        cim=corrupt_camera(_camera(cl[i]),plan["severity"]["camera"][i],plan["mode"]["camera"][i],i,seed=seed)
        cam_feat.append(camera_features(cim))
        if i%500==0: print(sequence_name,"features",i,"/",n)

    feat={
        "gps":gps_features(dgps,dt=cfg.dt,window=cfg.gps_window).astype(np.float32),
        "imu":imu_features(dacc,dgyro,window=cfg.imu_window).astype(np.float32),
        "camera":np.asarray(cam_feat,np.float32),
    }
    np.savez_compressed(out/"features.npz",gps_features=feat["gps"],imu_features=feat["imu"],camera_features=feat["camera"])
    np.savez_compressed(out/"sensor_data.npz",
        clean_gps_local=clean,degraded_gps_local=dgps,degraded_imu_acc=dacc,degraded_imu_gyro=dgyro,
        timestamps_seconds=_ts_seconds(timestamps),
        gps_severity=plan["severity"]["gps"],imu_severity=plan["severity"]["imu"],
        camera_severity=plan["severity"]["camera"],lidar_severity=plan["severity"]["lidar"])
    # OXTS/GPS local position is frozen evaluation reference; no post-hoc alignment.
    np.savetxt(out/"ground_truth.txt",clean,fmt="%.9f")

    calib=KITTICalibration(DATE_DIR); conv=BodyFrameConverter(calib)
    le=LidarOdometry(max_iterations=20,max_correspondence_distance=1.5,min_correspondences=80,max_points=6000)
    pcfg=BenchmarkConfig(); sched=build_schedule(n,pcfg)
    lc=[];lb=[];lv=[];lq=[];prev=np.eye(4)
    print(sequence_name,": recomputing PHYSICAL LiDAR odometry")
    for i in range(n-1):
        s=apply_degradation(_lidar(ll[i]),int(sched["type_id"][i]),int(sched["level_id"][i]),i,pcfg)
        t=apply_degradation(_lidar(ll[i+1]),int(sched["type_id"][i+1]),int(sched["level_id"][i+1]),i+1,pcfg)
        r=le.estimate(s,t,initial_transform=prev if i>0 else np.eye(4),source_frame_id=i,target_frame_id=i+1)
        ok,q=lidar_quality(r); m=conv.lidar_to_body(r.transform,valid=ok,quality=q)
        lc.append(m.coordinate_transform);lb.append(m.between_measurement);lv.append(ok);lq.append(q)
        prev=r.transform if ok else np.eye(4)
        if i%250==0: print(sequence_name,"LiDAR",i,"/",n-1,"ok",ok,"q",q)
    make_factor_data(np.asarray(lc),np.asarray(lb),np.asarray(lv,bool),np.asarray(lq,float)).save(out/"lidar_factor_data.npz")

    import cv2
    ce=StereoVisualOdometry(str(DATE_DIR/"calib_cam_to_cam.txt"),num_features=2500,min_matches=30,min_inliers=20,
                            ratio_test=.75,disparity_num=128,disparity_block_size=5,max_depth=80)
    cc=[];cb=[];cv=[];cq=[]
    print(sequence_name,": recomputing degraded stereo VO")
    for i in range(n-1):
        a=cv2.imread(str(left[i]),cv2.IMREAD_COLOR); b=cv2.imread(str(right[i]),cv2.IMREAD_COLOR); c=cv2.imread(str(left[i+1]),cv2.IMREAD_COLOR)
        if a is None or b is None or c is None:
            m=conv.camera_to_body(np.eye(4),valid=False,quality=0)
            cc.append(m.coordinate_transform);cb.append(m.between_measurement);cv.append(False);cq.append(0.);continue
        st,mt=plan["severity"]["camera"][i],plan["mode"]["camera"][i]
        sn,mn=plan["severity"]["camera"][i+1],plan["mode"]["camera"][i+1]
        a=corrupt_camera(a,st,mt,i,seed=seed); b=corrupt_camera(b,st,mt,i+100000,seed=seed); c=corrupt_camera(c,sn,mn,i+1,seed=seed)
        r=ce.estimate(a,b,c);ok,q=camera_quality(r);m=conv.camera_to_body(r.transform,valid=ok,quality=q)
        cc.append(m.coordinate_transform);cb.append(m.between_measurement);cv.append(ok);cq.append(q)
        if i%250==0: print(sequence_name,"Camera",i,"/",n-1,"ok",ok,"q",q)
    make_factor_data(np.asarray(cc),np.asarray(cb),np.asarray(cv,bool),np.asarray(cq,float)).save(out/"camera_factor_data.npz")

    meta={"sequence":sequence_name,"frames":n,"dev_sequence":"2011_10_03_drive_0027_sync",
          "plan":"0027 balanced plan resampled in normalized time; no target tuning",
          "lidar":"physical degradation schedule generated with frozen BenchmarkConfig",
          "camera":"same balanced corruption family as development",
          "gt":"OXTS/GPS local metric trajectory; no post-hoc alignment"}
    (out/"metadata.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
    print("Saved:",out)

def main():
    p=argparse.ArgumentParser();p.add_argument("--sequence",choices=TEST_SEQUENCES);p.add_argument("--all",action="store_true")
    a=p.parse_args()
    seqs=TEST_SEQUENCES if a.all else (a.sequence,)
    if seqs==(None,): raise SystemExit("use --sequence ... or --all")
    for s in seqs: build_one(s)
if __name__=="__main__": main()
