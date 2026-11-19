from __future__ import annotations

import numpy as np
import pytest

from experiments.isj_v7 import run_gps_covariance_pilot as pilot
from src.factor_graph.four_sensor_graph import FourSensorMeasurements
from src.factor_graph.sensor_factor_config import FourSensorFactorConfig


def _measurements(n: int = 4544):
    gps = np.zeros((n, 3), dtype=float)
    gps[:, 0] = 10 + np.arange(n) * 0.11
    gps[:, 1] = 4 + np.arange(n) * .03
    eye = np.tile(np.eye(4, dtype=float), (n - 1, 1, 1))
    return FourSensorMeasurements(
        gps_local=gps, imu_gyro=np.zeros((n, 3), float),
        timestamps=(np.arange(n) * .1).tolist(),
        lidar_between=eye.copy(), lidar_valid=np.ones(n-1, bool),
        lidar_quality=np.ones(n-1, float),
        camera_between=eye.copy(), camera_valid=np.ones(n-1, bool),
        camera_quality=np.ones(n-1, float),
    )


def test_frozen_base_and_bounded_reliability_policy():
    base = pilot.build_gps_sigmas(None)
    assert base.shape == (680,)
    assert np.all(base == 5.0)
    r = np.linspace(0.0, 1.0, 680)
    sig = pilot.build_gps_sigmas(r)
    assert sig[0] == pytest.approx(5.0 * np.sqrt(5.0))
    assert sig[-1] == pytest.approx(5.0)
    assert np.all(np.diff(sig) <= 0)
    assert np.all((sig >= 5.0) & (sig <= 5.0*np.sqrt(5.0)))


def test_uncertainty_is_not_direct_sigma_and_has_bounded_effect():
    r = np.ones(680) * .5
    u = np.ones(680) * .8
    reliability_only = pilot.build_gps_sigmas(r)
    with_uncertainty = pilot.build_gps_sigmas(r, u)
    assert np.all(with_uncertainty > reliability_only)
    assert np.allclose(
        with_uncertainty, 5.0 * np.sqrt(1 + 4 * .5 + .5 * .8))
    assert float(with_uncertainty.max()) <= 5.0*np.sqrt(5.5)


def test_invalid_sigma_inputs_are_rejected():
    with pytest.raises(ValueError, match="reliability"):
        pilot.build_gps_sigmas(np.ones(681))
    with pytest.raises(ValueError, match="reliability"):
        pilot.build_gps_sigmas(np.full(680, -0.01))
    with pytest.raises(ValueError, match="evidence"):
        pilot.build_gps_sigmas(np.full(680, 0.5), np.full(680, 1.2))
    with pytest.raises(ValueError, match="evidence"):
        pilot.build_gps_sigmas(None, np.ones(680))
    with pytest.raises(ValueError, match="base sigma"):
        pilot.build_gps_sigmas(None, base_sigma=0.0)


def test_subgraph_is_only_0027_validation_and_rebases_without_clean_reference():
    m = _measurements()
    segment = pilot.subgraph_measurements(m)
    assert segment.gps_local.shape == (680, 3)
    assert np.allclose(segment.gps_local[0], 0.0)
    assert segment.gps_local[-1, 0] == pytest.approx(.11*679)
    assert len(segment.lidar_valid) == 679
    assert len(segment.camera_valid) == 679
    assert len(segment.timestamps) == 680
    assert segment.timestamps[0] == pytest.approx(318.0)
    assert np.array_equal(
        segment.lidar_between[0], m.lidar_between[3180])
    assert not hasattr(segment, "clean_gps_local")
    assert len(m.gps_local) == 4544


def test_changing_future_internal_test_does_not_affect_subgraph():
    m = _measurements()
    expected = pilot.subgraph_measurements(m)
    m.gps_local[3860:] += 10000.
    m.imu_gyro[3860:] += 7.
    m.lidar_between[3860:] *= 3.
    m.camera_between[3860:] *= 9.
    observed = pilot.subgraph_measurements(m)
    assert np.array_equal(expected.gps_local, observed.gps_local)
    assert np.array_equal(
        expected.lidar_between, observed.lidar_between)
    assert np.array_equal(
        expected.camera_between, observed.camera_between)


def test_val_bounds_cannot_expand_into_internal_test():
    m = _measurements()
    with pytest.raises(ValueError, match="VAL boundary"):
        pilot.subgraph_measurements(m, end=3863)
    with pytest.raises(ValueError, match="VAL boundary"):
        pilot.subgraph_measurements(m, begin=3179)


def test_oxTS_diagnostic_is_relative_and_not_independent():
    truth = np.arange(680, dtype=float)
    ref = np.stack([truth, truth*.1, truth*.01], axis=1)
    observed = ref - ref[0]
    out = pilot.reference_diagnostics(observed, ref)
    assert out["independent_ATE"] is False
    assert out["independent_RPE"] is False
    assert out["origin_aligned_oxTS_relative_3d_rmse_m"] == pytest.approx(0)
    assert out["relative_displacement_10_frame_rmse_m"] == pytest.approx(0)
    with pytest.raises(ValueError, match="invalid"):
        pilot.reference_diagnostics(observed, np.ones((681, 3)))


def test_only_predeclared_pilot_cases_seeds_allowed_before_file_reads():
    with pytest.raises(ValueError, match="predeclared"):
        pilot.prepare_case("oracle", 43)
    with pytest.raises(ValueError, match="predeclared"):
        pilot.prepare_case("mamba_r", 42)


def test_preflight_does_not_write_or_optimize(tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr(
        pilot, "prepare_case",
        lambda case, seed: (
            np.ones(680)*5.0, {"case": case, "seed": seed,
                               "development_only": True}),
    )
    monkeypatch.setattr(pilot, "OUTPUT", tmp_path / "no_new_folders")
    monkeypatch.setattr(
        pilot, "run_four_sensor_factor_graph",
        lambda *a, **kw: called.append("optimized"),
    )
    out = pilot.run("fixed", 43, preflight_only=True)
    assert out["development_only"] is True
    assert called == []
    assert not (tmp_path / "no_new_folders").exists()


def test_legacy_factor_graph_override_rejects_malformed_sigmas(tmp_path):
    gtsam = pytest.importorskip("gtsam")
    from src.factor_graph.four_sensor_graph import (
        run_four_sensor_factor_graph,
    )
    m = _measurements(n=2)
    with pytest.raises(ValueError, match="finite positive"):
        run_four_sensor_factor_graph(
            m, tmp_path / "not_written", mode="fixed",
            gps_sigmas_override=np.asarray([5.0, np.nan]))
    with pytest.raises(ValueError, match="requires fixed mode"):
        run_four_sensor_factor_graph(
            m, tmp_path / "not_written", mode="predictive",
            gps_sigmas_override=np.asarray([5.0, 5.0]))
    assert not (tmp_path / "not_written").exists()



def test_fgo_interval_matches_exactly_680_h0_validation_anchors():
    # Raw VAL has 682 rows, but h0 has only 680 valid model anchors.
    # The last two raw rows are future targets for h1/h2, never FGO inputs.
    anchors = np.arange(3180, 3860, dtype=np.int64)
    assert pilot.BEGIN == 3180
    assert pilot.END == 3860
    assert pilot.END - pilot.BEGIN == len(anchors) == 680
    m = _measurements()
    sub = pilot.subgraph_measurements(m)
    assert sub.gps_local.shape[0] == len(anchors)
    assert np.array_equal(
        sub.gps_local[:, 0],
        m.gps_local[anchors, 0] - m.gps_local[3180, 0],
    )
    assert sub.gps_local[-1, 0] == pytest.approx(
        m.gps_local[3859, 0] - m.gps_local[3180, 0])



def test_mamba_mean_sigma_keeps_mean_but_removes_timing():
    source = np.linspace(5.1, 10.2, 680)
    outcome = pilot.mamba_sigma_control(source, "mamba_mean_sigma", 43)
    assert outcome.shape == source.shape
    assert np.all(outcome == np.mean(source))
    assert np.mean(outcome) == pytest.approx(np.mean(source))
    assert not np.shares_memory(source, outcome)


def test_mamba_permutation_preserves_values_not_time_alignment():
    source = np.linspace(5.1, 10.2, 680)
    one = pilot.mamba_sigma_control(source, "mamba_perm_sigma", 43)
    two = pilot.mamba_sigma_control(source, "mamba_perm_sigma", 43)
    assert np.array_equal(one, two)
    assert np.array_equal(np.sort(one), np.sort(source))
    assert not np.array_equal(one, source)
    assert not np.shares_memory(source, one)
    assert np.array_equal(source, np.linspace(5.1, 10.2, 680))


def test_mamba_sigma_controls_fail_on_missing_or_invalid_source():
    with pytest.raises(ValueError, match="invalid Mamba"):
        pilot.mamba_sigma_control(np.ones(682), "mamba_mean_sigma", 43)
    with pytest.raises(ValueError, match="invalid Mamba"):
        pilot.mamba_sigma_control(np.full(680, np.nan), "mamba_perm_sigma", 43)
    with pytest.raises(ValueError, match="frozen dev seeds"):
        pilot.mamba_sigma_control(np.ones(680), "mamba_perm_sigma", 42)
    with pytest.raises(ValueError, match="unknown mean"):
        pilot.mamba_sigma_control(np.ones(680), "unapproved", 43)


def test_posthoc_cases_are_explicit_and_preflight_only(monkeypatch, tmp_path):
    for name in ("no_gps", "mamba_mean_sigma", "mamba_perm_sigma"):
        assert name in pilot.CASES
    assert pilot.get_development_run({}, "no_gps", 43) is None
    for name in ("no_gps", "mamba_mean_sigma", "mamba_perm_sigma"):
        assert not (tmp_path / name).exists()
