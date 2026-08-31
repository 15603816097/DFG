from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
V5=ROOT/"results/sensor_specific_reliability_v5_uncertainty"
OUT=ROOT/"results/sensor_specific_reliability_v5_stage3_adapter";OUT.mkdir(parents=True,exist_ok=True)
for s in ("gps","imu","camera"):
    r=np.loadtxt(V5/f"{s}_reliability_target_aligned.txt")
    u=np.loadtxt(V5/f"{s}_uncertainty_target_aligned.txt")
    np.savetxt(OUT/f"{s}_reliability.txt",r,fmt="%.9f")
    np.savetxt(OUT/f"{s}_confidence.txt",1-np.clip(u,0,1),fmt="%.9f")
    print(s,"r",r.min(),r.max(),r.mean(),"u",u.min(),u.max(),u.mean())
print("PASS")
