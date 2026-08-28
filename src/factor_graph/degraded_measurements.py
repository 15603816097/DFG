from pathlib import Path
import numpy as np
from src.factor_graph.four_sensor_graph import FourSensorMeasurements

def load_degraded_four_sensor_measurements(degraded_dir):
    d=Path(degraded_dir)
    s=np.load(d/"degraded_sensor_data.npz",allow_pickle=False)
    l=np.load(d/"degraded_lidar_factor_data.npz",allow_pickle=False)
    c=np.load(d/"degraded_camera_factor_data.npz",allow_pickle=False)
    gps=np.asarray(s["degraded_gps_local"],float)
    gyro=np.asarray(s["degraded_imu_gyro"],float)
    ts=np.asarray(s["timestamps_seconds"],float).tolist()
    n=min(len(gps),len(gyro),len(ts),len(l["between_measurements"])+1,len(c["between_measurements"])+1)
    return FourSensorMeasurements(
        gps_local=gps[:n], imu_gyro=gyro[:n], timestamps=ts[:n],
        lidar_between=np.asarray(l["between_measurements"][:n-1],float),
        lidar_valid=np.asarray(l["valid"][:n-1],bool),
        lidar_quality=np.asarray(l["quality"][:n-1],float),
        camera_between=np.asarray(c["between_measurements"][:n-1],float),
        camera_valid=np.asarray(c["valid"][:n-1],bool),
        camera_quality=np.asarray(c["quality"][:n-1],float),
    )

def load_clean_reference(degraded_dir):
    s=np.load(Path(degraded_dir)/"degraded_sensor_data.npz",allow_pickle=False)
    return np.asarray(s["clean_gps_local"],float)
