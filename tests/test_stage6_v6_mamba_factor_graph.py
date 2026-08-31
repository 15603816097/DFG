from pathlib import Path
import ast
import importlib.util
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments" / "factor_graph" / "run_stage6_v6_mamba_factor_graph_ablation.py"


def load_module():
    spec = importlib.util.spec_from_file_location("stage6_fg", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_stage6_script_syntax():
    ast.parse(SCRIPT.read_text(encoding="utf-8"))


def test_stage6_case_design():
    m = load_module()
    cases = dict(m.build_cases())

    assert len(cases) == 7

    assert cases["G0_all_fixed"] == {
        "gps_mode": "fixed",
        "imu_mode": "fixed",
        "camera_mode": "fixed",
        "lidar_mode": "fixed",
    }

    assert cases["G1_v5_gru_all_uncertainty"]["gps_mode"] == "v5_uncertainty"
    assert cases["G1_v5_gru_all_uncertainty"]["imu_mode"] == "v5_uncertainty"
    assert cases["G1_v5_gru_all_uncertainty"]["camera_mode"] == "v5_uncertainty"
    assert cases["G1_v5_gru_all_uncertainty"]["lidar_mode"] == "uncertainty"

    assert cases["G2_v6_mamba_all_uncertainty"]["gps_mode"] == "v6_uncertainty"
    assert cases["G2_v6_mamba_all_uncertainty"]["imu_mode"] == "v6_uncertainty"
    assert cases["G2_v6_mamba_all_uncertainty"]["camera_mode"] == "v6_uncertainty"
    assert cases["G2_v6_mamba_all_uncertainty"]["lidar_mode"] == "uncertainty"

    hybrid = cases["G3_hybrid_gps_mamba_imu_gru_camera_mamba"]
    assert hybrid["gps_mode"] == "v6_uncertainty"
    assert hybrid["imu_mode"] == "v5_uncertainty"
    assert hybrid["camera_mode"] == "v6_uncertainty"
    assert hybrid["lidar_mode"] == "uncertainty"


def test_uncertainty_confidence_bounds():
    m = load_module()
    assert m.uncertainty_confidence(-1.0) == 1.0
    assert m.uncertainty_confidence(0.0) == 1.0
    assert m.uncertainty_confidence(0.25) == 0.75
    assert m.uncertainty_confidence(1.0) == 0.0
    assert m.uncertainty_confidence(2.0) == 0.0


def test_lidar_mapping_identity_at_zero_confidence():
    m = load_module()
    r, t = m.lidar_dynamic_sigmas(0.9, 0.0)
    assert np.isclose(r, m.LIDAR_R0)
    assert np.isclose(t, m.LIDAR_T0)
