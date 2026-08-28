import os,sys
ROOT=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:sys.path.insert(0,ROOT)
from src.loader.imu_loader import IMULoader
from src.multisensor_reliability.config import MultiSensorReliabilityConfig
from src.multisensor_reliability.degradation.plans import create_multisensor_degradation_plan,save_plan

DATASET=os.path.join(ROOT,"dataset","kitti","2011_10_03","2011_10_03_drive_0027_sync")
OUT=os.path.join(ROOT,"results","multisensor_reliability")

def main():
    cfg=MultiSensorReliabilityConfig();n=len(IMULoader(DATASET));plan=create_multisensor_degradation_plan(n,cfg.random_seed);save_plan(plan,OUT)
    print("="*80);print("MULTI-SENSOR DEGRADATION PLAN");print("Frames:",n,"Horizon:",cfg.horizon)
    for s in ("gps","imu","lidar","camera"):
        x=plan["severity"][s];print(f"{s:8s} min={x.min():.3f} max={x.max():.3f} mean={x.mean():.3f}")
    print("Saved:",OUT)
if __name__=="__main__":main()
