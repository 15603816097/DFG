from __future__ import annotations
import os
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
GT=ROOT/"results/ground_truth/trajectory.txt"
METHODS=[
("Current Best (old LiDAR fixed)",ROOT/"results/sensorwise_predictive_ablation/gps_imu_camera/trajectory.txt"),
("Physical LiDAR Fixed",ROOT/"results/lidar_physical_oracle_fg/physical_lidar_fixed/trajectory.txt"),
("Physical LiDAR GT-Oracle Dynamic",ROOT/"results/lidar_physical_oracle_fg/physical_lidar_oracle/trajectory.txt")]

def metrics(gt,x):
    n=min(len(gt),len(x)); e=x[:n,:3]-gt[:n,:3]; e3=np.linalg.norm(e,axis=1); e2=np.linalg.norm(e[:,:2],axis=1)
    return dict(ATE3D=float(np.sqrt(np.mean(e3**2))),ATE2D=float(np.sqrt(np.mean(e2**2))),
                Mean3D=float(e3.mean()),Max3D=float(e3.max()),ZRMSE=float(np.sqrt(np.mean(e[:,2]**2))))
def main():
    gt=np.loadtxt(GT); R={}
    print("="*128); print("PHYSICAL LIDAR ORACLE DYNAMIC COVARIANCE EVALUATION"); print("="*128)
    for name,p in METHODS:
        if not p.exists(): print(f"{name:42s}: NOT RUN"); continue
        m=metrics(gt,np.loadtxt(p)); R[name]=m
        print(f"{name:42s} ATE3D={m['ATE3D']:.6f} ATE2D={m['ATE2D']:.6f} Mean3D={m['Mean3D']:.6f} Max3D={m['Max3D']:.6f} ZRMSE={m['ZRMSE']:.6f}")
    f=R.get("Physical LiDAR Fixed"); o=R.get("Physical LiDAR GT-Oracle Dynamic")
    if f and o:
        imp=(f["ATE3D"]-o["ATE3D"])/f["ATE3D"]*100
        print(f"\nGT-Oracle improvement vs Physical LiDAR Fixed: {imp:+.2f}%")
        print("DECISION:", "LiDAR dynamic covariance has potential." if imp>1 else "Keeping LiDAR fixed is justified for this benchmark.")
if __name__=="__main__": main()
