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



def _synthetic_train_axis_data():
    n = len(pilot.TRAIN_ANCHORS)
    groups = np.arange(n) % 5
    levels = np.asarray([.1, .3, .5, .7, .9])
    reliability = levels[groups]
    # Squared error is constructed from group severity and axis energy;
    # a completely synthetic fixture, not KITTI/VAL/test values.
    scales = np.asarray([12., 7., 4., 2., .5])
    errors = np.column_stack((
        scales[groups] * .8,
        scales[groups] * .6,
        scales[groups] * .15,
    ))
    return errors, reliability


def test_train_axis_fit_has_five_monotone_bins_and_correct_fixed_rms():
    error, r = _synthetic_train_axis_data()
    fit = pilot.fit_train_axis_noise(error, r)
    assert fit["train_anchors"] == [63, 3177]
    assert sum(fit["counts"]) == 3115
    assert fit["sigma_bins_xyz"].shape == (5, 3)
    assert np.allclose(fit["centers"], [.1, .3, .5, .7, .9])
    assert np.all(np.diff(fit["sigma_bins_xyz"], axis=0) <= 0)
    assert np.allclose(
        fit["sigma_fixed_xyz"],
        np.sqrt(np.mean(error**2, axis=0)),
    )
    assert fit["sigma_fixed_xyz"][0] > fit["sigma_fixed_xyz"][2]


def test_axis_prediction_map_responds_to_reliability_and_is_axis_specific():
    error, r = _synthetic_train_axis_data()
    fitted = pilot.fit_train_axis_noise(error, r)
    prediction = np.linspace(.1, .9, 680)
    sigma = pilot.axis_sigma_from_prediction(prediction, fitted)
    assert sigma.shape == (680, 3)
    assert np.all(np.isfinite(sigma))
    assert np.all(np.diff(sigma, axis=0) <= 1e-10)
    assert np.all(sigma[:, 0] > sigma[:, 2])
    assert np.allclose(
        sigma[0], fitted["sigma_bins_xyz"][0],
    )
    assert np.allclose(
        sigma[-1], fitted["sigma_bins_xyz"][-1],
    )


def test_train_axis_mapping_rejects_invalid_labels_or_shapes():
    error, r = _synthetic_train_axis_data()
    with pytest.raises(ValueError, match="3115 clean train"):
        pilot.fit_train_axis_noise(error[:-1], r)
    with pytest.raises(ValueError, match="3115 clean train"):
        pilot.fit_train_axis_noise(error, np.full(3115, np.nan))
    fit = pilot.fit_train_axis_noise(error, r)
    with pytest.raises(ValueError, match="axis prediction"):
        pilot.axis_sigma_from_prediction(np.ones(682), fit)
    with pytest.raises(ValueError, match="axis prediction"):
        pilot.axis_sigma_from_prediction(np.full(680, -1.), fit)
    # All TRAIN anchors in only one bin is not enough for bin fit.
    with pytest.raises(ValueError, match="too few train"):
        pilot.fit_train_axis_noise(error, np.full(3115, .9))


def test_axis_train_fit_is_unaffected_by_val_and_test_source_rows(
        tmp_path, monkeypatch):
    n = 4544
    error, reliability = _synthetic_train_axis_data()
    clean = np.zeros((n, 3), dtype=np.float64)
    degraded = clean.copy()
    degraded[pilot.TRAIN_ANCHORS] = error
    label = np.zeros(n, dtype=np.float64)
    label[pilot.TRAIN_ANCHORS] = reliability

    sensors = tmp_path / "degraded_four_sensor_measurements"
    sensors.mkdir()
    oracle = tmp_path / "source" / "gps_oracle.npz"
    oracle.parent.mkdir()
    sensor_path = sensors / "degraded_sensor_data.npz"

    def write_archives(tail):
        degraded[3180:] = tail
        clean[3180:] = -tail
        label[3180:] = .99 if tail != 0 else .01
        np.savez(sensor_path, degraded_gps_local=degraded,
                 clean_gps_local=clean)
        np.savez(oracle, gps_current_factor_reliability=label)
        policy = {"raw_sources": {
            "degraded_sensor": {"sha256": pilot.sha256(sensor_path)},
            "v3": {"sha256": pilot.sha256(oracle),
                   "path": "source/gps_oracle.npz"},
        }}
        return policy

    monkeypatch.setattr(pilot, "ROOT", tmp_path)
    monkeypatch.setattr(pilot, "DEGRADED", sensors)
    first = pilot.load_train_axis_noise(write_archives(0))
    second = pilot.load_train_axis_noise(write_archives(1e6))
    assert np.array_equal(first["sigma_fixed_xyz"],
                          second["sigma_fixed_xyz"])
    assert np.array_equal(first["sigma_bins_xyz"],
                          second["sigma_bins_xyz"])
    assert first["counts"] == second["counts"]


def test_xyz_fgo_sigma_is_accepted_but_malformed_matrices_rejected(tmp_path):
    pytest.importorskip("gtsam")
    from src.factor_graph.four_sensor_graph import (
        run_four_sensor_factor_graph,
    )
    m = _measurements(n=2)
    matrix = np.asarray([[5., 6., 7.], [8., 9., 10.]])
    graph, initial, factor_counts = run_four_sensor_factor_graph(
        m, tmp_path / "never_write", mode="fixed",
        gps_sigmas_override=matrix, build_graph_only=True,
    )
    assert graph.size() == 6
    assert factor_counts["gps"] == 2
    assert np.isfinite(graph.error(initial))
    assert not (tmp_path / "never_write").exists()
    with pytest.raises(ValueError, match="finite positive"):
        run_four_sensor_factor_graph(
            m, tmp_path / "never_write", mode="fixed",
            gps_sigmas_override=np.ones((2, 2)),
            build_graph_only=True,
        )
    with pytest.raises(ValueError, match="finite positive"):
        run_four_sensor_factor_graph(
            m, tmp_path / "never_write", mode="fixed",
            gps_sigmas_override=np.asarray([[1, -2, 3], [1, 2, 3]]),
            build_graph_only=True,
        )


def test_xyz_and_repeated_isotropic_sigma_have_identical_graph_error(tmp_path):
    pytest.importorskip("gtsam")
    from src.factor_graph.four_sensor_graph import (
        run_four_sensor_factor_graph,
    )
    m = _measurements(n=2)
    scalar = np.asarray([5., 8.])
    xyztied = np.repeat(scalar[:, None], 3, axis=1)
    g1, v1, _ = run_four_sensor_factor_graph(
        m, tmp_path / "unused", mode="fixed",
        gps_sigmas_override=scalar, build_graph_only=True,
    )
    g2, v2, _ = run_four_sensor_factor_graph(
        m, tmp_path / "unused", mode="fixed",
        gps_sigmas_override=xyztied, build_graph_only=True,
    )
    assert g1.error(v1) == pytest.approx(g2.error(v2), abs=1e-10)


def test_axis_fixed_needs_no_network_output():
    assert pilot.get_development_run({}, "axis_fixed", 43) is None
    for name in ("axis_fixed", "axis_gru96_r",
                 "axis_gru106_r", "axis_mamba_r"):
        assert name in pilot.CASES



def test_cap4_limits_precision_to_four_times_axis_fixed():
    sigma = np.tile(
        np.array([0.50, 0.48, 0.31], dtype=np.float64),
        (pilot.END - pilot.BEGIN, 1),
    )
    axis_fixed = np.asarray([4.1457, 3.7861, 1.2329])
    capped = pilot.cap_axis_information(sigma, axis_fixed)
    assert capped.shape == sigma.shape
    assert np.allclose(capped[0], axis_fixed / 2.0)
    assert np.max((axis_fixed[None, :] / capped)**2) <= 4.0 + 1e-10
    assert np.array_equal(sigma[0], [.50, .48, .31])
    assert not np.shares_memory(sigma, capped)


def test_cap4_preserves_outlier_downweighting_without_shrinking_sigmas():
    axis_fixed = np.asarray([4.0, 3.0, 1.0])
    sigma = np.tile(np.asarray([10., 6., 2.0]), (680, 1))
    sigma[0] = [.1, .2, .05]
    actual = pilot.cap_axis_information(sigma, axis_fixed)
    assert np.allclose(actual[0], [2., 1.5, .5])
    assert np.array_equal(actual[1:], sigma[1:])
    assert np.all(actual >= sigma)
    assert np.all(actual > 0)


def test_cap4_rejects_invalid_inputs_or_precision_ratio_below_one():
    sigma = np.ones((680, 3))
    fixed = np.array([4., 3., 1.])
    with pytest.raises(ValueError, match="precision cap"):
        pilot.cap_axis_information(np.ones((680,)), fixed)
    with pytest.raises(ValueError, match="precision cap"):
        pilot.cap_axis_information(sigma, np.ones((4,)))
    with pytest.raises(ValueError, match="precision cap"):
        pilot.cap_axis_information(np.full((680, 3), np.nan), fixed)
    with pytest.raises(ValueError, match="precision cap"):
        pilot.cap_axis_information(sigma, fixed, max_information_ratio=.99)


def test_cap4_variants_are_explicit_posthoc_controls():
    assert pilot.AXIS_MAX_PRECISION_RATIO == 4.0
    assert (
        "axis_gru96_r_cap4", "axis_gru106_r_cap4", "axis_mamba_r_cap4"
    ) == tuple(c for c in pilot.CASES if c.endswith("_cap4"))
    for case in ("axis_gru96_r_cap4",
                 "axis_gru106_r_cap4", "axis_mamba_r_cap4"):
        assert case.startswith("axis_")
        assert case.removeprefix("axis_").removesuffix("_cap4") in (
            "gru96_r", "gru106_r", "mamba_r")


def _frozen_sequence_fixture(n=96):
    gps = np.column_stack((
        np.arange(n, dtype=float) * .11 + 10.,
        np.sin(np.arange(n) * .13),
        np.arange(n, dtype=float) * .005,
    ))
    err, r = _synthetic_train_axis_data()
    mapping = pilot.fit_train_axis_noise(err, r)
    return gps, mapping


def _frozen_gru_provenance():
    return {
        "case": "gru96_r", "window": 64,
        "prediction_offsets": [0, 1, 2],
        "model_is_development_only": True,
        "input_clip": 20.0,
    }


def test_frozen_sequence_fixed_and_dynamic_cold_start():
    import torch
    from experiments.isj_v7.model_dev_gps import ReliabilityForecastV7
    gps, fit = _frozen_sequence_fixture(n=96)
    original = gps.copy()
    model = ReliabilityForecastV7(
        backend="gru", input_dim=21, hidden=8, layers=1,
        horizon=3, dropout=0.,
    )
    zeros = np.zeros(21, dtype=np.float32)
    ones = np.ones(21, dtype=np.float32)
    fixed, fixed_meta = pilot.frozen_v7_sequence_gps_sigmas(
        gps, case="axis_fixed", train_axis_fit=fit,
    )
    dynamic, meta = pilot.frozen_v7_sequence_gps_sigmas(
        gps, case="axis_gru96_r_cap4", train_axis_fit=fit,
        model=model, train_median=zeros, train_iqr=ones,
        model_provenance=_frozen_gru_provenance(),
    )
    assert fixed.shape == dynamic.shape == (96, 3)
    assert np.array_equal(fixed, np.tile(fit["sigma_fixed_xyz"], (96, 1)))
    assert np.array_equal(dynamic[:63], fixed[:63])
    assert meta["cold_start_frames"] == 63
    assert fixed_meta["head_used_for_fgo"] == "none"
    assert meta["head_used_for_fgo"] == "h0"
    assert meta["future_heads_used_for_fgo"] is False
    assert np.all(dynamic[63:] >= fit["sigma_fixed_xyz"][None, :] / 2.)
    assert np.all(dynamic > 0)
    assert np.array_equal(gps, original)


def test_frozen_sequence_mapping_matches_original_v7_interpolation():
    from experiments.isj_v7.model_dev_gps import ReliabilityForecastV7
    from experiments.isj_v7.train_gps_dev import (
        predict_frozen_gps_features_causally,
    )
    from src.multisensor_reliability.features import gps_features
    from src.multisensor_reliability.config import MultiSensorReliabilityConfig
    gps, fit = _frozen_sequence_fixture(n=84)
    model = ReliabilityForecastV7(
        backend="gru", input_dim=21, hidden=8, layers=1,
        horizon=3, dropout=0.,
    )
    center = np.zeros(21, dtype=np.float32)
    scale = np.ones(21, dtype=np.float32)
    sigmas, _ = pilot.frozen_v7_sequence_gps_sigmas(
        gps, case="axis_gru96_r_cap4", train_axis_fit=fit,
        model=model, train_median=center, train_iqr=scale,
        model_provenance=_frozen_gru_provenance(),
    )
    cfg = MultiSensorReliabilityConfig()
    features = gps_features(gps, dt=cfg.dt, window=cfg.gps_window)
    anchors, predicted, _ = predict_frozen_gps_features_causally(
        model, features, center, scale, clip=20.,
    )
    raw = np.column_stack([
        np.interp(predicted[:, 0], fit["centers"],
                  fit["sigma_bins_xyz"][:, j])
        for j in range(3)
    ])
    expected = np.maximum(raw, fit["sigma_fixed_xyz"][None, :] / 2.)
    assert np.array_equal(anchors, np.arange(63, 84))
    assert np.allclose(sigmas[anchors], expected, atol=1e-10)


def test_frozen_sequence_uses_no_future_gps_and_is_batch_invariant():
    import torch
    from experiments.isj_v7.model_dev_gps import ReliabilityForecastV7
    torch.manual_seed(242)
    gps, fit = _frozen_sequence_fixture(n=95)
    model = ReliabilityForecastV7(
        backend="gru", input_dim=21, hidden=8, layers=1,
        horizon=3, dropout=0.,
    )
    med = np.zeros(21, dtype=np.float32)
    iqr = np.ones(21, dtype=np.float32)
    kwargs = dict(
        case="axis_gru96_r_cap4", train_axis_fit=fit, model=model,
        train_median=med, train_iqr=iqr,
        model_provenance=_frozen_gru_provenance(),
    )
    first, _ = pilot.frozen_v7_sequence_gps_sigmas(gps, **kwargs)
    future_altered = gps.copy()
    future_altered[81:] += np.array([50., -100., 20.])
    second, _ = pilot.frozen_v7_sequence_gps_sigmas(
        future_altered, **kwargs)
    assert np.array_equal(first[:81], second[:81])


def test_frozen_sequence_rejects_mismatched_model_and_training_mapping():
    from experiments.isj_v7.model_dev_gps import ReliabilityForecastV7
    gps, fit = _frozen_sequence_fixture(n=71)
    model = ReliabilityForecastV7(
        backend="gru", input_dim=21, hidden=8, layers=1,
        horizon=3, dropout=0.,
    )
    args = dict(
        model=model, train_median=np.zeros(21),
        train_iqr=np.ones(21), model_provenance=_frozen_gru_provenance())
    with pytest.raises(ValueError, match="cross-sequence case"):
        pilot.frozen_v7_sequence_gps_sigmas(
            gps, case="axis_mamba_r", train_axis_fit=fit, **args)
    with pytest.raises(ValueError, match="mismatched frozen model"):
        pilot.frozen_v7_sequence_gps_sigmas(
            gps, case="axis_mamba_r_cap4", train_axis_fit=fit, **args)
    with pytest.raises(ValueError, match="must not depend"):
        pilot.frozen_v7_sequence_gps_sigmas(
            gps, case="axis_fixed", train_axis_fit=fit, **args)
    bad_fit = dict(fit, sigma_bins_xyz=np.full((5, 3), np.nan))
    with pytest.raises(ValueError, match="training-only XYZ"):
        pilot.frozen_v7_sequence_gps_sigmas(
            gps, case="axis_fixed", train_axis_fit=bad_fit)
    with pytest.raises(ValueError, match="GPS stream dimensions"):
        pilot.frozen_v7_sequence_gps_sigmas(
            gps[:63], case="axis_fixed", train_axis_fit=fit)


def test_frozen_sequence_rebase_is_copy_only_and_strict_shape():
    m = _measurements(n=70)
    original = m.gps_local.copy()
    sub = pilot.rebase_frozen_v7_measurements(m)
    assert np.array_equal(m.gps_local, original)
    assert np.array_equal(sub.gps_local[0], [0., 0., 0.])
    assert np.allclose(sub.gps_local, original - original[0])
    assert np.array_equal(sub.lidar_between, m.lidar_between)
    assert not np.shares_memory(sub.gps_local, m.gps_local)
    assert not np.shares_memory(sub.lidar_between, m.lidar_between)
    m_short = _measurements(n=63)
    with pytest.raises(ValueError, match="sensor arrays"):
        pilot.rebase_frozen_v7_measurements(m_short)
    m_bad = _measurements(n=70)
    m_bad.camera_between = m_bad.camera_between[:-1]
    with pytest.raises(ValueError, match="between-factor"):
        pilot.rebase_frozen_v7_measurements(m_bad)


def test_frozen_sequence_sigma_consumed_by_same_four_sensor_graph(tmp_path):
    pytest.importorskip("gtsam")
    from src.factor_graph.four_sensor_graph import (
        run_four_sensor_factor_graph,
    )
    gps, fit = _frozen_sequence_fixture(n=70)
    m = _measurements(n=70)
    m.gps_local = gps
    sub = pilot.rebase_frozen_v7_measurements(m)
    sigma, _ = pilot.frozen_v7_sequence_gps_sigmas(
        gps, case="axis_fixed", train_axis_fit=fit,
    )
    graph, initial, counts = run_four_sensor_factor_graph(
        sub, tmp_path / "NEVER_WRITE", mode="fixed",
        gps_sigmas_override=sigma, build_graph_only=True,
    )
    assert counts["gps"] == 70
    assert counts["imu"] == 69
    assert counts["lidar"] == 69
    assert counts["camera"] == 69
    assert graph.size() == 1 + 70 + 3 * 69
    assert np.isfinite(graph.error(initial))
    assert not (tmp_path / "NEVER_WRITE").exists()
