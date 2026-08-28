from __future__ import annotations
import csv, json, os, sys
from pathlib import Path
import numpy as np

ROOT=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path: sys.path.insert(0,ROOT)

from src.degradation.lidar_physical_degradation import *
from src.reliability.lidar_gt_factor_oracle import build_oracle
from src.loader.lidar_loader import LidarLoader
from src.odometry.lidar_odometry import LidarOdometry
from src.preprocessing.calibration import KITTICalibration

DATASET=Path(ROOT)/"dataset/kitti/2011_10_03/2011_10_03_drive_0027_sync"
CALIB=Path(ROOT)/"dataset/kitti/2011_10_03"
OUT=Path(ROOT)/"results/lidar_physical_oracle_benchmark"

def points(item):
    if isinstance(item,np.ndarray): return item
    for k in ("points","point_cloud","lidar","data"):
        if k in item: return np.asarray(item[k])
    raise KeyError("LiDAR points not found")

def qscore(result):
    if not result.converged or not np.isfinite(result.rmse): return 0.
    return float(np.clip(.55*result.fitness+.30*np.exp(-result.rmse/1.5)+.15*np.clip(result.correspondences/5000.,0,1),0,1))

def main():
    print("="*120); print("LIDAR PHYSICAL DEGRADATION + GT FACTOR ERROR BENCHMARK"); print("="*120)
    cfg=BenchmarkConfig(); loader=LidarLoader(DATASET); n=len(loader); schedule=build_schedule(n,cfg)
    calib=KITTICalibration(CALIB); world=load_oxts_world_poses(DATASET); gt=gt_between(world)
    if len(world)!=n: raise RuntimeError(f"OXTS/LiDAR mismatch: {len(world)} vs {n}")
    estimator=LidarOdometry(max_iterations=20,max_correspondence_distance=1.5,min_correspondences=80,max_points=6000)
    OUT.mkdir(parents=True,exist_ok=True)
    arr={k:[] for k in ("raw_transform","body_between","fitness","rmse","corr","converged","quality","te","re","type","level","block")}
    src=apply_degradation(points(loader[0]),int(schedule["type_id"][0]),int(schedule["level_id"][0]),0,cfg)
    prev=np.eye(4)
    for i in range(n-1):
        tgt=apply_degradation(points(loader[i+1]),int(schedule["type_id"][i+1]),int(schedule["level_id"][i+1]),i+1,cfg)
        result=estimator.estimate(src,tgt,initial_transform=(prev if i>0 else np.eye(4)),source_frame_id=i,target_frame_id=i+1)
        prev=result.transform if result.converged else np.eye(4)
        b=lidar_transform_to_body_between(result.transform,calib.T_imu_to_velo,calib.T_velo_to_imu)
        te,re=factor_error(b,gt[i])
        arr["raw_transform"].append(result.transform); arr["body_between"].append(b)
        arr["fitness"].append(result.fitness); arr["rmse"].append(result.rmse); arr["corr"].append(result.correspondences)
        arr["converged"].append(int(result.converged)); arr["quality"].append(qscore(result)); arr["te"].append(te); arr["re"].append(re)
        arr["type"].append(int(schedule["type_id"][i+1])); arr["level"].append(int(schedule["level_id"][i+1])); arr["block"].append(int(schedule["block_id"][i+1]))
        if i%100==0: print(f"{i:04d}->{i+1:04d} type={arr['type'][-1]} level={arr['level'][-1]} fit={result.fitness:.3f} rmse={result.rmse:.3f} tErr={te:.3f} rErr={re:.3f}")
        src=tgt
    A={k:np.asarray(v) for k,v in arr.items()}
    np.savez_compressed(OUT/"physical_lidar_factor_data.npz",**A,gt_between=gt,**schedule)
    clean=A["level"]==0; oracle=build_oracle(A["te"],A["re"],clean)
    rel_frame=np.ones(n); rel_frame[1:]=oracle["lidar_reliability"]
    np.savez_compressed(OUT/"lidar_gt_oracle_reliability.npz",lidar_reliability=rel_frame,**oracle,translation_error=A["te"],rotation_error_deg=A["re"])
    names={0:"clean",1:"sparse",2:"range_noise",3:"occlusion",4:"ghost_outlier"}; levels={0:"clean",1:"mild",2:"moderate",3:"severe"}
    rows=[]
    for tid in range(5):
        for lid in range(4):
            if (tid==0)!=(lid==0): continue
            m=(A["type"]==tid)&(A["level"]==lid)
            if not m.any(): continue
            rows.append(dict(type=names[tid],level=levels[lid],count=int(m.sum()),
                translation_error_mean=float(A["te"][m].mean()),translation_error_p90=float(np.percentile(A["te"][m],90)),
                rotation_error_deg_mean=float(A["re"][m].mean()),rotation_error_deg_p90=float(np.percentile(A["re"][m],90)),
                fitness_mean=float(A["fitness"][m].mean()),rmse_mean=float(A["rmse"][m].mean()),
                oracle_reliability_mean=float(oracle["lidar_reliability"][m].mean())))
    with (OUT/"degradation_factor_error_summary.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    (OUT/"metadata.json").write_text(json.dumps(dict(frames=n,config=cfg.__dict__,types=list(TYPES),levels=list(LEVELS)),indent=2))
    print("Saved:",OUT)

if __name__=="__main__": main()
