import os,sys,numpy as np
ROOT=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:sys.path.insert(0,ROOT)
from src.loader.gps_loader import GPSLoader
from src.loader.imu_loader import IMULoader
from src.loader.lidar_loader import LidarLoader
from src.loader.camera_loader import CameraLoader
from src.multisensor_reliability.config import MultiSensorReliabilityConfig
from src.multisensor_reliability.degradation.plans import load_plan
from src.multisensor_reliability.degradation.apply import corrupt_gps,corrupt_imu,corrupt_lidar,corrupt_camera
from src.multisensor_reliability.labels import build_future_labels
from src.multisensor_reliability.features import gps_features,imu_features,lidar_features,camera_features

DATASET=os.path.join(ROOT,"dataset","kitti","2011_10_03","2011_10_03_drive_0027_sync")
PLAN=os.path.join(ROOT,"results","multisensor_reliability")
OUT=os.path.join(PLAN,"multisensor_reliability_data.npz")

def ext_lidar(item):
    if isinstance(item,np.ndarray):return item
    if isinstance(item,dict):
        for k in ("points","point_cloud","lidar","data"):
            if k in item:return np.asarray(item[k])
    raise KeyError("Cannot extract LiDAR points; check LidarLoader output.")

def ext_camera(item):
    if isinstance(item,np.ndarray):return item
    if isinstance(item,dict):
        for k in ("image","rgb","data"):
            if k in item:return np.asarray(item[k])
    raise KeyError("Cannot extract camera image; check CameraLoader output.")

def main():
    cfg=MultiSensorReliabilityConfig();plan=load_plan(PLAN)
    gl=GPSLoader(DATASET);il=IMULoader(DATASET);ll=LidarLoader(DATASET);cl=CameraLoader(DATASET,camera_id="image_02")
    n=min(len(gl),len(il),len(ll),len(cl),plan["n_frames"]);seed=plan["seed"]
    gps=[];acc=[];gyro=[];lf=[];cf=[]
    print("="*88);print("BUILD MULTI-SENSOR RELIABILITY DATA");print("Frames:",n)
    for i in range(n):
        gi,ii,li,ci=gl[i],il[i],ll[i],cl[i]
        gp=np.asarray(gi["position"],float);a=np.asarray(ii["acceleration"],float);g=np.asarray(ii["angular_velocity"],float)
        lp=ext_lidar(li);im=ext_camera(ci)
        gp=corrupt_gps(gp,plan["severity"]["gps"][i],plan["mode"]["gps"][i],i,seed)
        a,g=corrupt_imu(a,g,plan["severity"]["imu"][i],plan["mode"]["imu"][i],i,seed)
        lp=corrupt_lidar(lp,plan["severity"]["lidar"][i],plan["mode"]["lidar"][i],i,seed)
        im=corrupt_camera(im,plan["severity"]["camera"][i],plan["mode"]["camera"][i],i,seed)
        gps.append(gp);acc.append(a);gyro.append(g);lf.append(lidar_features(lp));cf.append(camera_features(im))
        if i%500==0:print("Processed:",i)
    gps=np.asarray(gps,float);acc=np.asarray(acc,float);gyro=np.asarray(gyro,float)
    gf=gps_features(gps,cfg.dt,cfg.gps_window);imf=imu_features(acc,gyro,cfg.imu_window);lf=np.asarray(lf,np.float32);cf=np.asarray(cf,np.float32)
    current,future=build_future_labels({s:plan["severity"][s][:n] for s in ("gps","imu","lidar","camera")},cfg.horizon,cfg.reliability_gamma)
    np.savez_compressed(OUT,gps_features=gf,imu_features=imf,lidar_features=lf,camera_features=cf,
        gps_current_label=current["gps"],imu_current_label=current["imu"],lidar_current_label=current["lidar"],camera_current_label=current["camera"],
        gps_future_label=future["gps"],imu_future_label=future["imu"],lidar_future_label=future["lidar"],camera_future_label=future["camera"],
        gps_severity=plan["severity"]["gps"][:n],imu_severity=plan["severity"]["imu"][:n],lidar_severity=plan["severity"]["lidar"][:n],camera_severity=plan["severity"]["camera"][:n],
        horizon=np.asarray([cfg.horizon],np.int64))
    print("GPS features:",gf.shape);print("IMU features:",imf.shape);print("LiDAR features:",lf.shape);print("Camera features:",cf.shape);print("Saved:",OUT)
if __name__=="__main__":main()
