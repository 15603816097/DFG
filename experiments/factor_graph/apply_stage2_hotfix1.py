from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
target = ROOT / "experiments/factor_graph/run_stage2_v4_factor_graph_ablation.py"

text = target.read_text(encoding="utf-8")
old = "    n = len(measurements.gps_position)\n"
new = "    n = len(measurements.gps_local)\n"

if old not in text:
    if new in text:
        print("Hotfix already applied:", target)
    else:
        raise RuntimeError(
            "Expected Stage-2 line not found. Do not patch blindly. "
            "Please send the current runner file."
        )
else:
    target.write_text(text.replace(old, new, 1), encoding="utf-8")
    print("Patched:", target)
    print("gps_position -> gps_local")

# Fast API sanity check before the expensive graph run.
from src.factor_graph.degraded_measurements import load_degraded_four_sensor_measurements
m = load_degraded_four_sensor_measurements(ROOT / "results/degraded_four_sensor_measurements")
assert hasattr(m, "gps_local")
assert hasattr(m, "imu_gyro")
assert hasattr(m, "timestamps")
assert hasattr(m, "lidar_between")
assert hasattr(m, "camera_between")
print("FourSensorMeasurements API sanity PASS")
print("Frames:", len(m.gps_local))
