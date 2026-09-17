from __future__ import annotations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATE_DIR = ROOT / "dataset" / "kitti" / "2011_10_03"
DEV_SEQUENCE = "2011_10_03_drive_0027_sync"
TEST_SEQUENCES = (
    "2011_10_03_drive_0034_sync",
    "2011_10_03_drive_0042_sync",
    "2011_10_03_drive_0047_sync",
)

DEV_PLAN_DIR = ROOT / "results" / "multisensor_reliability_v2"
V5_DIR = ROOT / "results" / "sensor_specific_reliability_v5_uncertainty"
V6_DIR = ROOT / "results" / "v6_mamba_uncertainty"
OUT_ROOT = ROOT / "results" / "cross_sequence"

CAMERA_ID = "image_02"
WINDOW = 64

# Frozen Stage-6 covariance anchors. Do not tune on 0034/0042/0047.
GPS_FIXED = 5.0
IMU_R0 = 0.03
CAM_T0 = 0.45
CAM_R0 = 0.04
LIDAR_T0 = 0.15
LIDAR_R0 = 0.012857142857142857

# Stage-7 deliberately freezes LiDAR covariance.
# Reason: Stage-6 LiDAR risk_score_pair is 0027-specific and must not be reused.
LIDAR_MODE = "fixed"

def sequence_path(name: str) -> Path:
    return DATE_DIR / name

def sequence_out(name: str) -> Path:
    return OUT_ROOT / name
