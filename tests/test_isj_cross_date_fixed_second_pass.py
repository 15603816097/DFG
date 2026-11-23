from __future__ import annotations

import json

import pytest

from experiments.isj_supplement.run_cross_date_fixed_second_pass import (
    FIXED_NAME, FIXED_SPEC, load_saved_ate, residual_sigma_stats,
)


def test_fixed_second_pass_is_named_separately_and_has_original_fixed_graph_type():
    assert FIXED_SPEC == {
        "name": "ISJ_CTRL_CROSSDATE_FIXED_SECOND_PASS",
        "type": "fixed",
    }
    assert FIXED_NAME not in (
        "FINAL_S0_fixed", "ISJ_RESADAPT_GPS",
        "ISJ_RESADAPT_GPS_IMU_CAMERA",
    )
    assert json.loads(json.dumps(FIXED_SPEC)) == FIXED_SPEC


def test_read_saved_ate_handles_bare_adaptive_and_continued_metrics(tmp_path):
    p = tmp_path / "bare.json"
    p.write_text(json.dumps({"ATE3D": 2.0}), encoding="utf-8")
    assert load_saved_ate(p) == pytest.approx(2.0)
    p.write_text(json.dumps({"metrics": {"ATE3D": 1.9}}), encoding="utf-8")
    assert load_saved_ate(p) == pytest.approx(1.9)
    p.write_text(json.dumps({"final_metrics": {"ATE3D": 1.8}}), encoding="utf-8")
    assert load_saved_ate(p) == pytest.approx(1.8)


def test_invalid_ate_rejected(tmp_path):
    p = tmp_path / "bad.json"
    with pytest.raises(FileNotFoundError):
        load_saved_ate(p)
    p.write_text(json.dumps({"ATE3D": float("nan")}))
    with pytest.raises(ValueError, match="Invalid ATE3D"):
        load_saved_ate(p)
    p.write_text(json.dumps({"ATE3D": -1}))
    with pytest.raises(ValueError, match="Invalid ATE3D"):
        load_saved_ate(p)


def test_residual_sigma_stats_read_and_validate(tmp_path):
    p = tmp_path / "metrics.json"
    payload = {
        "sigma_statistics": {
            "gps": {
                "mean_sigma_multiplier": 1.1,
                "max_sigma_multiplier": 2.0,
                "inflated_fraction": 0.1,
            },
            "imu": {
                "mean_sigma_multiplier": 1.0,
                "max_sigma_multiplier": 1.0,
                "inflated_fraction": 0.0,
            },
            "camera": {
                "mean_sigma_multiplier": 1.0,
                "max_sigma_multiplier": 1.0,
                "inflated_fraction": 0.0,
            },
        }
    }
    p.write_text(json.dumps(payload))
    result = residual_sigma_stats(p)
    assert result["gps"]["inflated_fraction"] == pytest.approx(.1)
    assert result["imu"]["mean_sigma_multiplier"] == pytest.approx(1)
    assert result["camera"]["max_sigma_multiplier"] == pytest.approx(1)
    payload["sigma_statistics"]["imu"]["inflated_fraction"] = float("nan")
    p.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Invalid imu/inflated_fraction"):
        residual_sigma_stats(p)
