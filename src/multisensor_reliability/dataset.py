import numpy as np
from src.multisensor_reliability.degradation.apply import corrupt_gps,corrupt_imu,corrupt_lidar,corrupt_camera
from src.multisensor_reliability.labels import severity_to_reliability

def _lidar(item):
    if isinstance(item,np.ndarray):return item
    if isinstance(item,dict):
        for k in ("points","point_cloud","lidar","data"):
            if k in item:return np.asarray(item[k])
    raise KeyError("Cannot find LiDAR points in loader output")

def _camera(item):
    if isinstance(item,np.ndarray):return item
    if isinstance(item,dict):
        for k in ("image","rgb","data"):
            if k in item:return np.asarray(item[k])
    raise KeyError("Cannot find camera image in loader output")

class MultiSensorReliabilityDataset:
    def __init__(self,gps_loader,imu_loader,lidar_loader,camera_loader,plan,reliability_gamma=1.5):
        self.gps_loader=gps_loader;self.imu_loader=imu_loader;self.lidar_loader=lidar_loader;self.camera_loader=camera_loader
        self.plan=plan;self.reliability_gamma=float(reliability_gamma)
        self.length=min(len(gps_loader),len(imu_loader),len(lidar_loader),len(camera_loader),int(plan["n_frames"]))
    def __len__(self):return self.length
    def __getitem__(self,index):
        index=int(index)
        if index<0:index+=self.length
        if index<0 or index>=self.length:raise IndexError(index)
        gi=self.gps_loader[index];ii=self.imu_loader[index];li=self.lidar_loader[index];ci=self.camera_loader[index]
        gp=np.asarray(gi["position"],float);acc=np.asarray(ii["acceleration"],float);gyro=np.asarray(ii["angular_velocity"],float)
        lp=_lidar(li);img=_camera(ci);seed=int(self.plan["seed"])
        sev={s:float(self.plan["severity"][s][index]) for s in ("gps","imu","lidar","camera")}
        mode={s:int(self.plan["mode"][s][index]) for s in ("gps","imu","lidar","camera")}
        cgp=corrupt_gps(gp,sev["gps"],mode["gps"],index,seed)
        cacc,cgyro=corrupt_imu(acc,gyro,sev["imu"],mode["imu"],index,seed)
        cl=corrupt_lidar(lp,sev["lidar"],mode["lidar"],index,seed)
        cc=corrupt_camera(img,sev["camera"],mode["camera"],index,seed)
        rel={s:float(severity_to_reliability([sev[s]],self.reliability_gamma)[0]) for s in sev}
        return {"frame_id":index,"clean":{"gps":gp,"acceleration":acc,"angular_velocity":gyro,"lidar":lp,"camera":img},"corrupted":{"gps":cgp,"acceleration":cacc,"angular_velocity":cgyro,"lidar":cl,"camera":cc},"severity":sev,"mode":mode,"reliability":rel}
