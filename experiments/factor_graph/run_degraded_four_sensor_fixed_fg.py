import os,sys
ROOT=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path: sys.path.insert(0,ROOT)
from src.factor_graph.degraded_measurements import load_degraded_four_sensor_measurements
from src.factor_graph.four_sensor_graph import run_four_sensor_factor_graph
D=os.path.join(ROOT,"results","degraded_four_sensor_measurements")
O=os.path.join(ROOT,"results","degraded_four_sensor_fixed_fg")
m=load_degraded_four_sensor_measurements(D)
print("="*100);print("DEGRADED FOUR-SENSOR FIXED-COVARIANCE FACTOR GRAPH");print("Frames:",len(m.gps_local))
t,_,c=run_four_sensor_factor_graph(m,O,mode="fixed",use_gps=True,use_imu=True,use_lidar=True,use_camera=True)
print("Trajectory:",t.shape);print("Factor counts:",c);print("Final position:",t[-1]);print("Saved:",O)
