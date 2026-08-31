from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
base=ROOT/"results/predictive_factor_reliability_v2_corrected"
d=np.load(base/"factor_reliability_training_data.npz",allow_pickle=False)
g=np.load(base/"corrected_feature_debug.npz",allow_pickle=False)
dims={"gps":21,"imu":20,"lidar":19,"camera":16}
n=None
for s,k in dims.items():
    x=d[f"{s}_features"];print(s,x.shape)
    assert x.ndim==2 and x.shape[1]==k and np.all(np.isfinite(x))
    n=len(x) if n is None else n
    assert len(x)==n
lla=g["gps_lla"];local=g["gps_local_clean"];cor=g["gps_local_corrupted"]
assert np.linalg.norm(local[0])<1e-6
assert np.ptp(local[:,0])>10 or np.ptp(local[:,1])>10
delta=np.linalg.norm(cor-local,axis=1)
print("GPS local span [m]:",np.ptp(local,axis=0))
print("GPS corruption [m] min/max/mean:",delta.min(),delta.max(),delta.mean())
tid=g["lidar_type_id"];lid=g["lidar_level_id"]
print("Physical LiDAR degraded frames:",int(np.sum(tid>0)))
print("type IDs:",np.unique(tid),"level IDs:",np.unique(lid))
assert np.any(tid>0)
print("PASS")
