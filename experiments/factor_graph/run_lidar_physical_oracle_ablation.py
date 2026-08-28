from __future__ import annotations
import os,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
ROOT=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path: sys.path.insert(0,ROOT)
from src.factor_graph.degraded_measurements import load_degraded_four_sensor_measurements
from src.factor_graph.sensorwise_reliability_graph import ReliabilitySource, run_sensorwise_reliability_graph

BASE=os.path.join(ROOT,"results/degraded_four_sensor_measurements")
PRED=os.path.join(ROOT,"results/predictive_factor_reliability_v1")
BENCH=os.path.join(ROOT,"results/lidar_physical_oracle_benchmark")
FACTOR=os.path.join(BENCH,"physical_lidar_factor_data.npz")
ORACLE=os.path.join(BENCH,"lidar_gt_oracle_reliability.npz")
OUT=os.path.join(ROOT,"results/lidar_physical_oracle_fg")

def physical_measurements():
    base=load_degraded_four_sensor_measurements(BASE)
    values=dict(vars(base)); data=np.load(FACTOR,allow_pickle=False); m=SimpleNamespace(**values)
    m.lidar_between=np.asarray(data["body_between"],float)
    m.lidar_quality=np.asarray(data["quality"],float)
    m.lidar_valid=np.asarray(data["converged"],bool)
    return m

def sources(lidar_mode):
    return {"gps":ReliabilitySource("predictive"),"imu":ReliabilitySource("predictive"),
            "lidar":ReliabilitySource(lidar_mode),"camera":ReliabilitySource("predictive")}

def main():
    print("="*120); print("PHYSICAL LIDAR FIXED VS GT-ORACLE DYNAMIC"); print("="*120)
    if not Path(FACTOR).exists(): raise FileNotFoundError("Run physical benchmark builder first")
    m=physical_measurements()
    print("Frames:",len(m.gps_local),"LiDAR valid:",int(m.lidar_valid.sum()),"/",len(m.lidar_valid))
    for name,mode in [("physical_lidar_fixed","fixed"),("physical_lidar_oracle","oracle")]:
        print("\n"+"#"*120); print(name,"LiDAR mode:",mode); print("#"*120)
        run_sensorwise_reliability_graph(m,os.path.join(OUT,name),sources(mode),
            predictive_dir=PRED,oracle_path=(ORACLE if mode=="oracle" else None))
if __name__=="__main__": main()
