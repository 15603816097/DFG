import os
from pathlib import Path
import numpy as np
ROOT=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D=os.path.join(ROOT,"results","degraded_four_sensor_measurements")
S=np.load(os.path.join(D,"degraded_sensor_data.npz"),allow_pickle=False)
ref=np.asarray(S["clean_gps_local"],float)
METHODS=[
("Clean Fixed Four-Sensor",os.path.join(ROOT,"results","four_sensor_fixed_fg","trajectory.txt")),
("Degraded Fixed Four-Sensor",os.path.join(ROOT,"results","degraded_four_sensor_fixed_fg","trajectory.txt")),
("Degraded Predictive Four-Sensor",os.path.join(ROOT,"results","degraded_four_sensor_predictive_fg","trajectory.txt")),
]
def metric(r,t):
    n=min(len(r),len(t));e=t[:n]-r[:n];e3=np.linalg.norm(e,axis=1);e2=np.linalg.norm(e[:,:2],axis=1)
    return dict(ATE3D=float(np.sqrt(np.mean(e3**2))),ATE2D=float(np.sqrt(np.mean(e2**2))),Mean3D=float(np.mean(e3)),Max3D=float(np.max(e3)),ZRMSE=float(np.sqrt(np.mean(e[:,2]**2))))
print("="*112);print("DEGRADED FOUR-SENSOR FACTOR GRAPH EVALUATION");print("="*112)
res={}
for name,path in METHODS:
    if not Path(path).exists(): print(f"{name:36s}: NOT RUN");continue
    x=metric(ref,np.loadtxt(path,float));res[name]=x
    print(f"{name:36s} ATE3D={x['ATE3D']:.6f} ATE2D={x['ATE2D']:.6f} Mean3D={x['Mean3D']:.6f} Max3D={x['Max3D']:.6f} ZRMSE={x['ZRMSE']:.6f}")
if "Degraded Fixed Four-Sensor" in res and "Degraded Predictive Four-Sensor" in res:
    f=res["Degraded Fixed Four-Sensor"]["ATE3D"];p=res["Degraded Predictive Four-Sensor"]["ATE3D"]
    print(f"\nPredictive improvement vs degraded fixed: {(f-p)/max(f,1e-12)*100:.2f}%")
