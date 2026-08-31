from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

from src.loader.gps_loader import GPSLoader
from src.loader.imu_loader import IMULoader
from src.loader.lidar_loader import LidarLoader
from src.loader.camera_loader import CameraLoader
from src.factor_graph.four_sensor_graph import gps_geodetic_to_local
from src.multisensor_reliability.config import MultiSensorReliabilityConfig
from src.multisensor_reliability.degradation.balanced_plans import load_balanced_plan
from src.multisensor_reliability.degradation.apply import corrupt_gps,corrupt_imu,corrupt_camera
from src.multisensor_reliability.features import gps_features,imu_features,lidar_features,camera_features
from src.degradation.lidar_physical_degradation import BenchmarkConfig,build_schedule,apply_degradation

SENSORS=("gps","imu","lidar","camera")

def _lidar(x):
    if isinstance(x,np.ndarray): return x
    for k in ("points","point_cloud","lidar","data"):
        if isinstance(x,dict) and k in x:return np.asarray(x[k])
    raise KeyError("Cannot extract LiDAR points")

def _camera(x):
    if isinstance(x,np.ndarray): return x
    for k in ("image","rgb","data"):
        if isinstance(x,dict) and k in x:return np.asarray(x[k])
    raise KeyError("Cannot extract camera image")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--sequence",default=str(ROOT/"dataset/kitti/2011_10_03/2011_10_03_drive_0027_sync"))
    p.add_argument("--plan-dir",default=str(ROOT/"results/multisensor_reliability_v2"))
    p.add_argument("--old-training-data",default=str(ROOT/"results/predictive_factor_reliability_v1/factor_reliability_training_data.npz"))
    p.add_argument("--output-dir",default=str(ROOT/"results/predictive_factor_reliability_v2_corrected"))
    a=p.parse_args()
    seq,plan_dir,old_path,out=Path(a.sequence),Path(a.plan_dir),Path(a.old_training_data),Path(a.output_dir)
    out.mkdir(parents=True,exist_ok=True)

    cfg=MultiSensorReliabilityConfig(); plan=load_balanced_plan(plan_dir)
    gl,il,ll,cl=GPSLoader(seq),IMULoader(seq),LidarLoader(seq),CameraLoader(seq,camera_id="image_02")
    n=min(len(gl),len(il),len(ll),len(cl),int(plan["n_frames"]))
    gps_lla=np.asarray([gl[i]["position"] for i in range(n)],np.float64)

    # FIX 1: meter coordinates BEFORE corruption/features.
    gps_local=gps_geodetic_to_local(gps_lla)

    gps_rows=[]; acc_rows=[]; gyro_rows=[]; lidar_rows=[]; camera_rows=[]
    seed=int(plan["seed"])
    pcfg=BenchmarkConfig(); sched=build_schedule(n,pcfg)

    print("="*96);print("CORRECTED SHARED FACTOR FEATURES V2");print("="*96)
    print("Frames:",n)
    for i in range(n):
        im=il[i]
        g=corrupt_gps(gps_local[i],plan["severity"]["gps"][i],plan["mode"]["gps"][i],i,seed=seed)
        acc,gyro=corrupt_imu(np.asarray(im["acceleration"],float),np.asarray(im["angular_velocity"],float),
                            plan["severity"]["imu"][i],plan["mode"]["imu"][i],i,seed=seed)
        # FIX 2: shared-model LiDAR feature uses physical degradation stream.
        lp=apply_degradation(_lidar(ll[i]),int(sched["type_id"][i]),int(sched["level_id"][i]),i,pcfg)
        cam=corrupt_camera(_camera(cl[i]),plan["severity"]["camera"][i],plan["mode"]["camera"][i],i,seed=seed)
        gps_rows.append(g);acc_rows.append(acc);gyro_rows.append(gyro)
        lidar_rows.append(lidar_features(lp));camera_rows.append(camera_features(cam))
        if i%500==0:print("Processed:",i)

    feat={
      "gps":gps_features(np.asarray(gps_rows),dt=cfg.dt,window=cfg.gps_window),
      "imu":imu_features(np.asarray(acc_rows),np.asarray(gyro_rows),window=cfg.imu_window),
      "lidar":np.asarray(lidar_rows,np.float32),
      "camera":np.asarray(camera_rows,np.float32),
    }
    old=np.load(old_path,allow_pickle=False)
    m=min([n]+[len(np.asarray(old[f"{s}_current_factor_reliability"]).reshape(-1)) for s in SENSORS])
    arr={"horizon":np.asarray(old["horizon"]).copy()}
    for s in SENSORS:
        arr[f"{s}_features"]=np.asarray(feat[s][:m],np.float32)
        arr[f"{s}_current_factor_reliability"]=np.asarray(old[f"{s}_current_factor_reliability"][:m],np.float32)
        arr[f"{s}_future_factor_reliability"]=np.asarray(old[f"{s}_future_factor_reliability"][:m],np.float32)
    np.savez_compressed(out/"factor_reliability_training_data.npz",**arr)
    np.savez_compressed(out/"corrected_feature_debug.npz",gps_lla=gps_lla[:m],gps_local_clean=gps_local[:m],
                        gps_local_corrupted=np.asarray(gps_rows[:m]),lidar_type_id=sched["type_id"][:m],
                        lidar_level_id=sched["level_id"][:m],lidar_features=feat["lidar"][:m])
    meta={"version":"v2_corrected","frames":m,
          "gps_fix":"geodetic -> local meters before corruption/features",
          "lidar_fix":"shared input uses physical LiDAR degradation",
          "labels":"validated V1 factor-error labels retained; old features not reused",
          "note":"final E8 does not consume the shared LiDAR head; dedicated LiDAR predictor supplies LiDAR covariance"}
    (out/"dataflow_metadata.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
    for s in SENSORS:print(s,arr[f"{s}_features"].shape)
    print("Saved:",out)

if __name__=="__main__":main()
