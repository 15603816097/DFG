from pathlib import Path
import inspect
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
target = ROOT / "experiments/factor_graph/run_stage2_v4_factor_graph_ablation.py"
text = target.read_text(encoding="utf-8")

old = '        dyn.PRED = pred_dir\n        dyn.MAPPING = mapping\n        dyn.OUT = case_dir\n\n        trajectory, poses, counts = dyn.run_dynamic_graph(measurements)\n'
new = '        dyn.PRED = str(pred_dir)\n        dyn.MAPPING = str(mapping)\n        dyn.OUT = str(case_dir)\n\n        mapping_data = np.load(mapping, allow_pickle=False)\n        sigma_t_pair = np.asarray(\n            mapping_data["translation_sigma_pair"],\n            dtype=np.float64,\n        ).reshape(-1)\n        sigma_r_pair = np.asarray(\n            mapping_data["rotation_sigma_pair"],\n            dtype=np.float64,\n        ).reshape(-1)\n\n        expected_pairs = len(measurements.gps_local) - 1\n        if len(sigma_t_pair) != expected_pairs:\n            raise ValueError(\n                f"F8 sigma_t length {len(sigma_t_pair)} != {expected_pairs}"\n            )\n        if len(sigma_r_pair) != expected_pairs:\n            raise ValueError(\n                f"F8 sigma_r length {len(sigma_r_pair)} != {expected_pairs}"\n            )\n\n        trajectory, poses, counts = dyn.run_dynamic_graph(\n            measurements,\n            sigma_t_pair,\n            sigma_r_pair,\n            str(case_dir),\n        )\n'

if old in text:
    target.write_text(text.replace(old, new, 1), encoding="utf-8")
    print("Patched F8 direct dynamic LiDAR call.")
elif new in text:
    print("Hotfix 3 already applied.")
else:
    raise RuntimeError(
        "Expected F8 dynamic call block not found. Refusing blind replacement."
    )

import experiments.factor_graph.run_lidar_calibrated_dynamic_fg as dyn
sig = inspect.signature(dyn.run_dynamic_graph)
print("Installed run_dynamic_graph signature:", sig)

required = ["measurements", "sigma_t_pair", "sigma_r_pair", "output_dir"]
if list(sig.parameters.keys())[:4] != required:
    raise RuntimeError(
        "Unexpected run_dynamic_graph API: "
        f"{list(sig.parameters.keys())}"
    )

mapping_path = (
    ROOT / "results/stage2_v4_factor_graph_ablation"
    / "F8/lidar_lambda1_mapping.npz"
)

if mapping_path.exists():
    data = np.load(mapping_path, allow_pickle=False)
    st = np.asarray(data["translation_sigma_pair"], float).reshape(-1)
    sr = np.asarray(data["rotation_sigma_pair"], float).reshape(-1)
    print("Existing F8 mapping pairs:", len(st))
    print("sigma_t min/max/mean:", float(st.min()), float(st.max()), float(st.mean()))
    print("sigma_r min/max/mean:", float(sr.min()), float(sr.max()), float(sr.mean()))
    assert len(st) == 4543
    assert len(sr) == 4543
    assert np.all(np.isfinite(st)) and np.all(st > 0)
    assert np.all(np.isfinite(sr)) and np.all(sr > 0)
    print("F8 covariance sanity PASS")
else:
    print("F8 mapping will be regenerated automatically on rerun.")

print("No retraining required.")
