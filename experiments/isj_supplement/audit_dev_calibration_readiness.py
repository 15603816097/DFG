from __future__ import annotations

"""READ-ONLY audit for developing evidence-guided FGO on 0027, not 0016.

No imports of GTSAM/Mamba, model inference, retraining, or filesystem writes.
Checks original 2011_10_03_drive_0027_sync development artifacts, existing
frozen V5/V6 predictions, and whether the V3 training source is traceable.
This is a *readiness audit*; existence does not establish data provenance,
independent reference truth, or train/validation split integrity.
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PATHS = {
    "V3 same-source training NPZ":
        ROOT / "results/predictive_factor_reliability_v3_same_source"
        / "factor_reliability_training_data.npz",
    "V3 source metadata":
        ROOT / "results/predictive_factor_reliability_v3_same_source"
        / "same_source_metadata.json",
    "0027 degraded measurement NPZ":
        ROOT / "results/degraded_four_sensor_measurements"
        / "degraded_sensor_data.npz",
    "0027 degraded camera factor NPZ":
        ROOT / "results/degraded_four_sensor_measurements"
        / "degraded_camera_factor_data.npz",
    "0027 physical LiDAR factor NPZ":
        ROOT / "results/lidar_physical_oracle_benchmark"
        / "physical_lidar_factor_data.npz",
    "0027 fixed-FGO poses":
        ROOT / "results/four_sensor_fixed_fg" / "poses.npy",
    "0027 trajectory evaluation reference":
        ROOT / "results/ground_truth" / "trajectory.txt",
}

for sensor in ("gps", "imu", "camera"):
    PATHS[f"V5 frozen {sensor} checkpoint"] = (
        ROOT / "results/sensor_specific_reliability_v5_uncertainty"
        / sensor / "model.pt"
    )
    PATHS[f"V6 frozen {sensor} checkpoint"] = (
        ROOT / "results/v6_mamba_uncertainty" / sensor / "best.pt"
    )
    for model, directory in (
        ("V5", "sensor_specific_reliability_v5_uncertainty"),
        ("V6", "v6_mamba_uncertainty"),
    ):
        for signal in ("reliability", "uncertainty"):
            PATHS[f"0027 {model} {sensor} {signal}"] = (
                ROOT / "results" / directory
                / f"{sensor}_{signal}_target_aligned.txt"
            )


def audit() -> None:
    print("=" * 104)
    print("ISJ DEVELOPMENT-ONLY READINESS AUDIT (NO FILES WRITTEN)")
    print("Scope: development 2011_10_03_drive_0027_sync, not explored 0016")
    print("=" * 104)
    present = 0
    for label, path in PATHS.items():
        status = "READY" if path.is_file() else "MISSING"
        present += int(path.is_file())
        print(f"{status:<7} {label:<39} {path.relative_to(ROOT)}")
    print(f"\nPRESENT {present}/{len(PATHS)} artifacts")

    v3 = PATHS["V3 same-source training NPZ"]
    if v3.is_file():
        with np.load(v3, allow_pickle=False) as z:
            lens = {}
            for sensor in ("gps", "imu", "camera"):
                x = f"{sensor}_features"
                y = f"{sensor}_current_factor_reliability"
                if x not in z.files or y not in z.files:
                    print(f"WARNING: V3 missing {x} or {y}")
                    continue
                if len(z[x]) != len(z[y]):
                    print(f"WARNING: V3 array length mismatch: {sensor}")
                lens[sensor] = len(z[y])
            print("\nV3 FRAME COUNTS", lens)
            if lens and len(set(lens.values())) != 1:
                print("WARNING: training sequence lengths mismatch")

    metadata_file = PATHS["V3 source metadata"]
    if metadata_file.is_file():
        meta = json.loads(metadata_file.read_text(encoding="utf-8"))
        print("\nV3 PROVENANCE FIELDS")
        for key in ("version", "frames", "horizon", "feature_source",
                    "gps_imu_measurement_source", "reference_poses"):
            print(f"{key}: {meta.get(key)}")

    print("\nIMPORTANT PREPUBLICATION CHECKS")
    print("1. V6 checkpoint saving code does NOT include training NPZ path.")
    print("   Preserve original training commands/logs to establish lineage;")
    print("   file existence/metadata above cannot prove V6 used V3 labels.")
    print("2. V3 'future' array stores t+3, while current V5/V6 target")
    print("   construction treats its first column as the first future head.")
    print("   Do not claim correctly aligned multi-horizon forecasting.")
    print("3. No re-training, FGO solve, GT use in weight generation,")
    print("   or historical-result modification was performed by this audit.")
    print("=" * 104)


if __name__ == "__main__":
    audit()
