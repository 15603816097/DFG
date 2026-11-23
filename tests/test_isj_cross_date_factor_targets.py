from __future__ import annotations

import math

import numpy as np
import pytest

from experiments.isj_supplement.audit_cross_date_factor_targets import quality_scores
from src.reliability.oracle_factor_reliability import (
    OracleReliabilityConfig, error_to_reliability,
)


def test_zero_prediction_error_and_mean_baseline():
    y = np.array([0.9, 0.7, 0.3, 0.1], dtype=float)
    result = quality_scores(y.copy(), y)
    assert result["n"] == 4
    assert result["MAE"] == pytest.approx(0)
    assert result["RMSE"] == pytest.approx(0)
    assert result["Pearson"] == pytest.approx(1.0)
    assert result["Spearman"] == pytest.approx(1.0)
    assert result["mean_target_baseline_RMSE"] == pytest.approx(y.std())


def test_reference_target_formula_matches_known_imu_scale():
    cfg = OracleReliabilityConfig()
    assert cfg.imu_rotation_scale_deg == 2.0
    e = np.array([0.0, 1.0, 2.0, 4.0])
    expected = np.clip(np.exp(-((e/2.0)**2)), 0.01, 1.0)
    got = error_to_reliability(
        e, cfg.imu_rotation_scale_deg,
        cfg.minimum_reliability, cfg.maximum_reliability
    )
    assert np.allclose(got, expected)


def test_constant_target_does_not_fabricate_correlation():
    y = np.ones(10)
    r = np.linspace(.5,.9,10)
    stats = quality_scores(r,y)
    assert math.isnan(stats["Pearson"])
    assert math.isnan(stats["Spearman"])
    assert stats["mean_target_baseline_RMSE"] == 0


def test_invalid_arrays_fail_before_scoring():
    with pytest.raises(ValueError):
        quality_scores(np.ones(3),np.ones(4))
    with pytest.raises(ValueError):
        quality_scores(np.array([0.5,math.nan,0.8]),np.ones(3))
