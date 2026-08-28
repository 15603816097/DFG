#!/usr/bin/env python3
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; R=ROOT/"results"; O=R/"lidar_covariance_calibration_v1"
O.mkdir(parents=True,exist_ok=True)
SIGMAS=[.10,.12,.15,.18,.20,.25,.30,.35,.40,.50]
(O/"fixed_sigma_candidates.txt").write_text("\n".join(map(str,SIGMAS))+"\n")
print("LiDAR fixed sigma candidates:",SIGMAS)
print("IMPORTANT: run each candidate with the same GPS/IMU/Camera settings and physical LiDAR factors.")
print("After evaluation write the best translation sigma only to:")
print(O/"best_fixed_sigma.txt")
