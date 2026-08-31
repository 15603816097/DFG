from pathlib import Path
import inspect

ROOT = Path(__file__).resolve().parents[2]
target = ROOT / "experiments/factor_graph/run_stage2_v4_factor_graph_ablation.py"
text = target.read_text(encoding="utf-8")

old = '''    trajectory, poses, counts = run_sensorwise_reliability_graph(
        measurements=measurements,
        output_dir=case_dir,
        gps_source=source(gps_on),
        imu_source=source(imu_on),
        lidar_source=ReliabilitySource(mode="fixed"),
        camera_source=source(camera_on),
        predictive_dir=pred_dir,
        config=fixed_config(),
    )
'''

new = '''    sources = {
        "gps": source(gps_on),
        "imu": source(imu_on),
        "lidar": ReliabilitySource(mode="fixed"),
        "camera": source(camera_on),
    }

    trajectory, poses, counts = run_sensorwise_reliability_graph(
        measurements=measurements,
        output_dir=case_dir,
        sources=sources,
        predictive_dir=pred_dir,
        config=fixed_config(),
    )
'''

if old in text:
    target.write_text(text.replace(old, new, 1), encoding="utf-8")
    print("Patched sensorwise graph call -> sources dict API")
elif new in text:
    print("Hotfix 2 already applied.")
else:
    raise RuntimeError("Expected Stage-2 call not found; refusing blind patch.")

from src.factor_graph.sensorwise_reliability_graph import (
    run_sensorwise_reliability_graph,
    ReliabilitySource,
)
sig = inspect.signature(run_sensorwise_reliability_graph)
print("Installed signature:", sig)
assert "sources" in sig.parameters
for wrong in ("gps_source", "imu_source", "lidar_source", "camera_source"):
    assert wrong not in sig.parameters
probe = {s: ReliabilitySource("fixed") for s in ("gps","imu","lidar","camera")}
assert len(probe) == 4
print("Sensorwise API sanity PASS")
print("No model retraining is required.")
