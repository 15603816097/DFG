from __future__ import annotations
import os,sys
from pathlib import Path
import numpy as np
ROOT=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path: sys.path.insert(0,ROOT)

from src.loader.gps_loader import GPSLoader
from src.loader.imu_loader import IMULoader
from src.loader.lidar_loader import LidarLoader
from src.factor_graph.four_sensor_graph import gps_geodetic_to_local
from src.multisensor_reliability.degradation.balanced_plans import load_balanced_plan
from src.multisensor_reliability.degradation.apply import corrupt_imu,corrupt_lidar,corrupt_camera
from src.preprocessing.calibration import KITTICalibration
from src.odometry.body_frame_converter import BodyFrameConverter
from src.odometry.lidar_odometry import LidarOdometry
from src.odometry.stereo_visual_odometry import StereoVisualOdometry
from src.odometry.relative_factor_data import make_factor_data

SEQUENCE=os.path.join(ROOT,"dataset","kitti","2011_10_03","2011_10_03_drive_0027_sync")
DATE_DIR=os.path.join(ROOT,"dataset","kitti","2011_10_03")
PLAN_DIR=os.path.join(ROOT,"results","multisensor_reliability_v2")
OUT=os.path.join(ROOT,"results","degraded_four_sensor_measurements")
LEFT=os.path.join(SEQUENCE,"image_02","data")
RIGHT=os.path.join(SEQUENCE,"image_03","data")
CALIB=os.path.join(DATE_DIR,"calib_cam_to_cam.txt")

def extract_lidar(item):
    if isinstance(item,np.ndarray): return item
    if isinstance(item,dict):
        for k in ("points","point_cloud","lidar","data"):
            if k in item:return np.asarray(item[k])
    raise KeyError("Cannot extract LiDAR points")

def ts_seconds(ts):
    if not ts:return np.array([],float)
    if hasattr(ts[0],"timestamp"):return np.asarray([float(t.timestamp()) for t in ts],float)
    return np.asarray(ts,float)

def corrupt_local_gps(p,severity,mode,frame_id,seed):
    p=np.asarray(p,float).copy();s=float(np.clip(severity,0,1))
    if s<=0:return p
    rng=np.random.default_rng(int(seed)+int(frame_id)*1009+31)
    if int(mode)==2:
        d=rng.normal(size=3);d/=max(np.linalg.norm(d),1e-12);d[2]*=.25;p+=d*(10+18*s)
    else:
        noise=rng.normal(0,.30+4*s,3);noise[2]*=.5
        bias=np.array([6*s*np.sin(frame_id/120),4*s*np.cos(frame_id/155),.8*s*np.sin(frame_id/190)])
        p+=noise+bias
    return p

def lidar_quality(r):
    ok=bool(r.converged and np.isfinite(r.rmse) and r.correspondences>=80)
    if not ok:return False,0.0
    q=.50*np.clip(r.fitness,0,1)+.25*np.exp(-r.rmse/.75)+.25*np.clip(r.correspondences/6000,0,1)
    return True,float(np.clip(q,0,1))

def camera_quality(r):
    ok=bool(r.success and np.isfinite(r.reprojection_error) and r.inliers>=20 and r.matches>=30)
    if not ok:return False,0.0
    ratio=r.inliers/max(r.matches,1);q=.50*np.clip(ratio,0,1)+.20*np.clip(r.inliers/1200,0,1)+.30*np.exp(-r.reprojection_error/2)
    return True,float(np.clip(q,0,1))

def main():
    import cv2
    os.makedirs(OUT,exist_ok=True)
    plan=load_balanced_plan(PLAN_DIR);seed=int(plan["seed"])
    gl=GPSLoader(SEQUENCE);il=IMULoader(SEQUENCE);ll=LidarLoader(SEQUENCE)
    lf=sorted(Path(LEFT).glob("*.png"));rf=sorted(Path(RIGHT).glob("*.png"))
    n=min(len(gl),len(il),len(ll),len(lf),len(rf),int(plan["n_frames"]))
    print("="*104);print("BUILD DEGRADED FOUR-SENSOR MEASUREMENT CHAIN");print("Frames:",n)

    lla=np.asarray([gl[i]["position"] for i in range(n)],float)
    clean=gps_geodetic_to_local(lla);dgps=np.zeros_like(clean);dacc=np.zeros((n,3));dgyro=np.zeros((n,3));timestamps=[]
    for i in range(n):
        dgps[i]=corrupt_local_gps(clean[i],plan["severity"]["gps"][i],plan["mode"]["gps"][i],i,seed)
        a,g=corrupt_imu(il[i]["acceleration"],il[i]["angular_velocity"],plan["severity"]["imu"][i],plan["mode"]["imu"][i],i,seed=seed)
        dacc[i]=a;dgyro[i]=g;timestamps.append(il[i]["timestamp"])
    np.savez_compressed(os.path.join(OUT,"degraded_sensor_data.npz"),clean_gps_local=clean,degraded_gps_local=dgps,degraded_imu_acc=dacc,degraded_imu_gyro=dgyro,timestamps_seconds=ts_seconds(timestamps),gps_severity=plan["severity"]["gps"][:n],imu_severity=plan["severity"]["imu"][:n])
    print("GPS + IMU degraded measurements saved.")

    calib=KITTICalibration(DATE_DIR);conv=BodyFrameConverter(calib)
    le=LidarOdometry(max_iterations=20,max_correspondence_distance=1.5,min_correspondences=80,max_points=6000)
    ce=StereoVisualOdometry(CALIB,num_features=2500,min_matches=30,min_inliers=20,ratio_test=.75,disparity_num=128,disparity_block_size=5,max_depth=80)

    lc=[];lb=[];lv=[];lq=[];prev=np.eye(4)
    print("Recomputing degraded LiDAR odometry...")
    for i in range(n-1):
        s=corrupt_lidar(extract_lidar(ll[i]),plan["severity"]["lidar"][i],plan["mode"]["lidar"][i],i,seed=seed)
        t=corrupt_lidar(extract_lidar(ll[i+1]),plan["severity"]["lidar"][i+1],plan["mode"]["lidar"][i+1],i+1,seed=seed)
        r=le.estimate(s,t,initial_transform=prev if i>0 else np.eye(4),source_frame_id=i,target_frame_id=i+1)
        ok,q=lidar_quality(r);m=conv.lidar_to_body(r.transform,valid=ok,quality=q)
        lc.append(m.coordinate_transform);lb.append(m.between_measurement);lv.append(ok);lq.append(q);prev=r.transform if ok else np.eye(4)
        if i%100==0:print(f"LiDAR {i:04d}->{i+1:04d} sev={plan['severity']['lidar'][i]:.2f} fitness={r.fitness:.3f} rmse={r.rmse:.3f} q={q:.3f} ok={ok}")
    make_factor_data(np.asarray(lc),np.asarray(lb),np.asarray(lv,bool),np.asarray(lq,float)).save(os.path.join(OUT,"degraded_lidar_factor_data.npz"))

    cc=[];cb=[];cv=[];cq=[]
    print("Recomputing degraded stereo visual odometry...")
    for i in range(n-1):
        a=cv2.imread(str(lf[i]),cv2.IMREAD_COLOR);b=cv2.imread(str(rf[i]),cv2.IMREAD_COLOR);c=cv2.imread(str(lf[i+1]),cv2.IMREAD_COLOR)
        if a is None or b is None or c is None:
            m=conv.camera_to_body(np.eye(4),valid=False,quality=0);cc.append(m.coordinate_transform);cb.append(m.between_measurement);cv.append(False);cq.append(0.);continue
        st=plan["severity"]["camera"][i];mt=plan["mode"]["camera"][i];sn=plan["severity"]["camera"][i+1];mn=plan["mode"]["camera"][i+1]
        a=corrupt_camera(a,st,mt,i,seed=seed);b=corrupt_camera(b,st,mt,i+100000,seed=seed);c=corrupt_camera(c,sn,mn,i+1,seed=seed)
        r=ce.estimate(a,b,c);ok,q=camera_quality(r);m=conv.camera_to_body(r.transform,valid=ok,quality=q)
        cc.append(m.coordinate_transform);cb.append(m.between_measurement);cv.append(ok);cq.append(q)
        if i%100==0:print(f"Camera {i:04d}->{i+1:04d} sev={st:.2f} inliers={r.inliers} matches={r.matches} reproj={r.reprojection_error:.3f} q={q:.3f} ok={ok}")
    make_factor_data(np.asarray(cc),np.asarray(cb),np.asarray(cv,bool),np.asarray(cq,float)).save(os.path.join(OUT,"degraded_camera_factor_data.npz"))
    np.savez_compressed(os.path.join(OUT,"degradation_metadata.npz"),gps_severity=plan["severity"]["gps"][:n],imu_severity=plan["severity"]["imu"][:n],lidar_severity=plan["severity"]["lidar"][:n],camera_severity=plan["severity"]["camera"][:n],lidar_valid=np.asarray(lv,bool),camera_valid=np.asarray(cv,bool),lidar_quality=np.asarray(lq,float),camera_quality=np.asarray(cq,float))
    print("="*104);print("DEGRADED MEASUREMENT BUILD FINISHED");print("LiDAR valid:",int(np.sum(lv)),"/",n-1);print("Camera valid:",int(np.sum(cv)),"/",n-1);print("Saved:",OUT)

if __name__=="__main__":main()
