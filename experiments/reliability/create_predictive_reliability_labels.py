import os,sys,numpy as np
ROOT=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path: sys.path.insert(0,ROOT)
from src.reliability.label_generator import reliability_from_error,future_window_min
GT=os.path.join(ROOT,"results","ground_truth","trajectory.txt")
GPS=os.path.join(ROOT,"results","progressive_gps_degradation","gps_corrupted.txt")
OUT=os.path.join(ROOT,"results","predictive_reliability_labels")
H=5
def main():
    gt=np.loadtxt(GT);gps=np.loadtxt(GPS)
    e=np.linalg.norm(gps-gt,axis=1);cur=reliability_from_error(e,8.0);future=future_window_min(cur,H)
    os.makedirs(OUT,exist_ok=True)
    np.savetxt(os.path.join(OUT,"gps_error.txt"),e,fmt="%.8f")
    np.savetxt(os.path.join(OUT,"current_reliability.txt"),cur,fmt="%.8f")
    np.savetxt(os.path.join(OUT,f"future_window_reliability_h{H}.txt"),future,fmt="%.8f")
    print("Current min/max/mean:",cur.min(),cur.max(),cur.mean())
    print("Future min/max/mean:",future.min(),future.max(),future.mean());print("Saved:",OUT)
if __name__=="__main__":main()
