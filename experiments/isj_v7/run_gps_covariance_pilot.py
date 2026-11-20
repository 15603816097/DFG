from __future__ import annotations

"""V7 GPS-only 0027 VALIDATION interval four-sensor covariance pilot.

Explicitly a retrospective *development* diagnostic, not causal iSAM2
and not independent-drive evidence. Do not read 0027 internal TEST frames
3862..4543, or held-out drives 0016/0020. Never use synthetic label or
clean OXTS reference for GPS covariance selection. Existing dataset,
training predictions and historical factor graph outputs are immutable.

The other three sensor covariances stay fixed. A single predeclared,
bounded GPS sigma rule tests whether reliable GPS predictions can
actually affect offline factor-graph state estimation.
"""

import argparse
import json
import hashlib
from pathlib import Path

import numpy as np

from experiments.isj_v7.build_dev_gps_candidate import DEFAULT_DEST
from experiments.isj_v7.compare_dev_gps import read_candidate, read_run, sha256
from experiments.isj_v7.train_gps_dev import (
    CAPACITY_OUTPUT, DEFAULT_OUTPUT, predict_frozen_gps_features_causally,
)
from src.multisensor_reliability.config import MultiSensorReliabilityConfig
from src.multisensor_reliability.features import gps_features
from experiments.isj_v7.supervision_gate import read_policy, validate_policy
from src.factor_graph.degraded_measurements import (
    load_degraded_four_sensor_measurements,
)
from src.factor_graph.four_sensor_graph import (
    FourSensorMeasurements, run_four_sensor_factor_graph,
)
from src.factor_graph.sensor_factor_config import FourSensorFactorConfig

ROOT = Path(__file__).resolve().parents[2]
DEGRADED = ROOT / "results" / "degraded_four_sensor_measurements"
OUTPUT = ROOT / "results" / "isj_v7" / "gps_four_sensor_dev_pilot_v1"
BEGIN = 3180
# Validation has 682 raw frames [3180,3862), but 680 h0 anchors
# [3180,3860). The two tail frames [3860,3862) are *labels only*
# for h1/h2 on the final anchors, NOT additional FGO measurements.
END = 3860
EXPECTED_N = 4544
CASES = ("fixed", "no_gps", "gru96_r", "gru106_r", "mamba_r", "mamba_r_u",
         "mamba_mean_sigma", "mamba_perm_sigma",
         "axis_fixed", "axis_gru96_r", "axis_gru106_r", "axis_mamba_r",
         "axis_gru96_r_cap4", "axis_gru106_r_cap4", "axis_mamba_r_cap4")
BASE_SIGMA = FourSensorFactorConfig().gps_fixed_sigma
R_GAIN = 4.0
EVIDENCE_GAIN = 0.5


def build_gps_sigmas(
    reliability: np.ndarray | None,
    evidence: np.ndarray | None = None,
    *, base_sigma: float = BASE_SIGMA,
) -> np.ndarray:
    """Frozen heuristic; NO calibration, oracle labels or GT tuning.

    sigma_gps(t) = 5m * sqrt(1 + 4(1-r_t) + 0.5 U_t)
    'reliability-only' omits U. U is only a bounded evidence proxy,
    NOT calibrated sensor observation variance.
    """
    if not np.isfinite(base_sigma) or base_sigma <= 0:
        raise ValueError("invalid base sigma")
    if reliability is None:
        if evidence is not None:
            raise ValueError("evidence without reliability")
        return np.full(END - BEGIN, base_sigma, dtype=np.float64)
    r = np.asarray(reliability, dtype=np.float64)
    if (r.shape != (END - BEGIN,) or not np.isfinite(r).all()
            or np.any((r < 0) | (r > 1))):
        raise ValueError("invalid reliability: require 680 values in [0,1]")
    e = np.zeros_like(r)
    if evidence is not None:
        e = np.asarray(evidence, dtype=np.float64)
        if (e.shape != r.shape or not np.isfinite(e).all()
                or np.any((e < 0) | (e > 1))):
            raise ValueError("invalid evidence U: require 680 values in [0,1]")
    return base_sigma * np.sqrt(1.0 + R_GAIN * (1.0 - r)
                                + EVIDENCE_GAIN * e)



# All axis-noise fit statistics come from *training* anchors only.
# [3180,3862) raw validation and [3862,4544) test are NEVER calibration
# inputs. Bins were chosen before examining axis-variant FGO outcomes.
TRAIN_ANCHORS = np.arange(63, 3178, dtype=np.int64)
AXIS_BIN_EDGES = np.linspace(0.0, 1.0, 6, dtype=np.float64)
AXIS_BIN_CENTERS = (AXIS_BIN_EDGES[1:] + AXIS_BIN_EDGES[:-1]) / 2.0
AXIS_STD_FLOOR_M = 0.10


def fit_train_axis_noise(
    errors_xyz: np.ndarray, oracle_reliability: np.ndarray,
) -> dict:
    """Train-only empirical per-axis RMS, grouped by *oracle* reliability.

    This is an exploratory synthetic-label conditional noise mapping,
    NOT probabilistically calibrated to out-of-sample predicted scores.
    In particular, the oracle target was defined from these very errors.
    No validation/truth information is accepted by this function.
    """
    error = np.asarray(errors_xyz, dtype=np.float64)
    r = np.asarray(oracle_reliability, dtype=np.float64)
    if (error.shape != (len(TRAIN_ANCHORS), 3)
            or r.shape != (len(TRAIN_ANCHORS),)
            or not np.isfinite(error).all() or not np.isfinite(r).all()
            or np.any((r < 0) | (r > 1))):
        raise ValueError("axis mapping requires 3115 clean train anchors")
    fixed = np.sqrt(np.mean(error**2, axis=0))
    bin_xyz = []
    counts = []
    for i, (lo, hi) in enumerate(zip(AXIS_BIN_EDGES[:-1],
                                     AXIS_BIN_EDGES[1:])):
        m = (r >= lo) & (r < hi if i < 4 else r <= hi)
        n = int(m.sum())
        if n < 25:
            raise ValueError("too few train labels for axis noise bin")
        counts.append(n)
        bin_xyz.append(np.sqrt(np.mean(error[m]**2, axis=0)))
    bins = np.asarray(bin_xyz, dtype=np.float64)
    # Conservative monotone enforcement when a noisy finite bin
    # contradicts the expected synthetic oracle severity ordering.
    bins = np.minimum.accumulate(bins, axis=0)
    bins = np.maximum(bins, AXIS_STD_FLOOR_M)
    fixed = np.maximum(fixed, AXIS_STD_FLOOR_M)
    return {
        "sigma_fixed_xyz": fixed,
        "sigma_bins_xyz": bins,
        "centers": AXIS_BIN_CENTERS.copy(),
        "counts": counts,
        "train_anchors": [int(TRAIN_ANCHORS[0]),
                          int(TRAIN_ANCHORS[-1])],
    }


def load_train_axis_noise(policy: dict) -> dict:
    """Check frozen source hashes, then slice *only* the 3115 TRAIN rows."""
    gps_path = DEGRADED / "degraded_sensor_data.npz"
    oracle_meta = policy["raw_sources"]["v3"]
    oracle_path = ROOT / oracle_meta["path"]
    if (sha256(gps_path) !=
            policy["raw_sources"]["degraded_sensor"]["sha256"]
            or sha256(oracle_path) != oracle_meta["sha256"]):
        raise ValueError("GPS training calibration source hash mismatch")
    with np.load(gps_path, allow_pickle=False) as z:
        degraded = np.asarray(z["degraded_gps_local"][:3180],
                              dtype=np.float64)
        clean = np.asarray(z["clean_gps_local"][:3180],
                           dtype=np.float64)
    with np.load(oracle_path, allow_pickle=False) as z:
        label = np.asarray(
            z["gps_current_factor_reliability"][:3180], dtype=np.float64)
    if (degraded.shape != (3180, 3) or clean.shape != (3180, 3)
            or label.shape != (3180,)):
        raise ValueError("train-only axis source has wrong shape")
    return fit_train_axis_noise(
        degraded[TRAIN_ANCHORS] - clean[TRAIN_ANCHORS],
        label[TRAIN_ANCHORS],
    )


def axis_sigma_from_prediction(r: np.ndarray, fitted: dict) -> np.ndarray:
    """Smooth, monotone train-binned XYZ RMS indexed by predicted r(t).

    The prediction distribution is not calibrated to the oracle; this
    mapping is intentionally hypothesis-testing, not a validated
    physical observation likelihood.
    """
    prediction = np.asarray(r, dtype=np.float64)
    bins = np.asarray(fitted["sigma_bins_xyz"], dtype=np.float64)
    centers = np.asarray(fitted["centers"], dtype=np.float64)
    if (prediction.shape != (END-BEGIN,) or bins.shape != (5, 3)
            or centers.shape != (5,) or not np.isfinite(prediction).all()
            or np.any((prediction < 0) | (prediction > 1))
            or not np.isfinite(bins).all() or np.any(bins <= 0)):
        raise ValueError("axis prediction/calibration dimensions mismatch")
    return np.column_stack([
        np.interp(prediction, centers, bins[:, axis])
        for axis in range(3)
    ])


# Fixed post-hoc stress-test value: per-axis precision cannot exceed
# four times the training-only fixed-axis precision. This is NOT
# validation-selected, calibrated, or a confirmed optimal cap.
AXIS_MAX_PRECISION_RATIO = 4.0


def cap_axis_information(
    sigmas_xyz: np.ndarray, train_fixed_xyz: np.ndarray,
    *, max_information_ratio: float = AXIS_MAX_PRECISION_RATIO,
) -> np.ndarray:
    """Cap GPS information relative to *TRAIN-only* fixed-axis noise.

    P(t,a)=1/sigma(t,a)^2 <= 4 / sigma_train_fixed(a)^2.
    Only the lower sigma bound is changed; severe GPS downweighting
    remains intact. Never reads synthetic labels or OXTS reference here.
    """
    sigma = np.asarray(sigmas_xyz, dtype=np.float64)
    fixed = np.asarray(train_fixed_xyz, dtype=np.float64)
    if (sigma.shape != (END - BEGIN, 3) or fixed.shape != (3,)
            or not np.isfinite(sigma).all()
            or not np.isfinite(fixed).all()
            or np.any(sigma <= 0) or np.any(fixed <= 0)
            or not np.isfinite(max_information_ratio)
            or max_information_ratio < 1):
        raise ValueError("invalid GPS precision cap input")
    floor_xyz = fixed / np.sqrt(max_information_ratio)
    result = np.maximum(sigma, floor_xyz[None, :])
    if (np.any((fixed[None, :] / result)**2 >
               max_information_ratio * (1 + 1e-12))):
        raise AssertionError("GPS precision exceeds fixed-axis cap")
    return result


def mamba_sigma_control(
    raw_sigmas: np.ndarray, case: str, seed: int,
) -> np.ndarray:
    """Post-hoc diagnostic controls isolating mean versus temporal variation.

    Fixed, deterministic permutation seed is independent of any GT metric.
    Returns a fresh array and never edits the original Mamba predictions.
    """
    a = np.asarray(raw_sigmas, dtype=np.float64)
    if (a.shape != (END - BEGIN,) or not np.isfinite(a).all()
            or np.any(a <= 0)):
        raise ValueError("invalid Mamba GPS sigma sequence")
    if seed not in (43, 44, 45):
        raise ValueError("sigma control is restricted to frozen dev seeds")
    if case == "mamba_mean_sigma":
        return np.full_like(a, float(a.mean()))
    if case == "mamba_perm_sigma":
        return np.random.default_rng(20261120 + seed).permutation(a)
    raise ValueError("unknown mean/permutation control")



# Closed-test adapter. It does NOT authorize reading KITTI 0016/0020;
# its inputs are supplied by a future separately audited data builder.
# In-memory code only, with the original 0027 protocol left unchanged.
CROSS_SEQUENCE_CASES = {
    "axis_fixed": None,
    "axis_gru96_r_cap4": "gru96_r",
    "axis_gru106_r_cap4": "gru106_r",
    "axis_mamba_r_cap4": "mamba_r",
}
CROSS_SEQUENCE_HISTORY = 64


def rebase_frozen_v7_measurements(measurements: FourSensorMeasurements):
    """Make a per-drive origin-relative, copy-only four-sensor graph input.

    Do not read the reference trajectory or alter raw sensor arrays.
    LiDAR/stereo factor construction must use the correct calibration
    and identical generation protocol *before* calling this helper.
    """
    m = measurements
    gps = np.asarray(m.gps_local, dtype=np.float64)
    gyro = np.asarray(m.imu_gyro, dtype=np.float64)
    if (gps.ndim != 2 or gps.shape[1] != 3
            or len(gps) < CROSS_SEQUENCE_HISTORY
            or gyro.shape != gps.shape or not np.isfinite(gps).all()
            or not np.isfinite(gyro).all()
            or len(m.timestamps) != len(gps)):
        raise ValueError("invalid frozen V7 cross-sequence sensor arrays")
    n = len(gps)
    arrays = {
        name: np.asarray(getattr(m, name))
        for name in ("lidar_between", "camera_between",
                     "lidar_valid", "camera_valid",
                     "lidar_quality", "camera_quality")
    }
    if any(arrays[name].shape != (n - 1, 4, 4)
           or not np.isfinite(arrays[name]).all()
           for name in ("lidar_between", "camera_between")):
        raise ValueError("invalid frozen V7 between-factor shapes")
    if any(arrays[name].shape != (n - 1,)
           or not np.isfinite(arrays[name]).all()
           for name in ("lidar_quality", "camera_quality")):
        raise ValueError("invalid frozen V7 factor quality shapes")
    if any(arrays[name].shape != (n - 1,)
           for name in ("lidar_valid", "camera_valid")):
        raise ValueError("invalid frozen V7 factor validity shapes")
    return FourSensorMeasurements(
        gps_local=gps.copy() - gps[0].copy(),
        imu_gyro=gyro.copy(),
        timestamps=list(m.timestamps),
        lidar_between=arrays["lidar_between"].astype(np.float64).copy(),
        lidar_valid=arrays["lidar_valid"].astype(bool).copy(),
        lidar_quality=arrays["lidar_quality"].astype(np.float64).copy(),
        camera_between=arrays["camera_between"].astype(np.float64).copy(),
        camera_valid=arrays["camera_valid"].astype(bool).copy(),
        camera_quality=arrays["camera_quality"].astype(np.float64).copy(),
    )


def frozen_v7_sequence_gps_sigmas(
    degraded_gps_local: np.ndarray, *, case: str,
    train_axis_fit: dict,
    model=None,
    train_median: np.ndarray | None = None,
    train_iqr: np.ndarray | None = None,
    model_provenance: dict | None = None,
) -> tuple[np.ndarray, dict]:
    """Build arbitrary-length, model-causal XYZ sigmas with frozen Cap4.

    The only sensor input is the same degraded GPS realization that
    must later enter the FGO. No clean reference, GT, oracle labels,
    future measurements or validation-derived calibration are accepted.
    No checkpoint is trained/reloaded; all parameters are supplied.
    At t<63, all cases use the *same* fixed train-axis sigmas.
    Predictions at h1/h2 are NOT used to weigh GPS at t.
    """
    if case not in CROSS_SEQUENCE_CASES:
        raise ValueError("unsupported frozen V7 cross-sequence case")
    gps = np.asarray(degraded_gps_local, dtype=np.float64)
    if (gps.ndim != 2 or gps.shape[1] != 3
            or len(gps) < CROSS_SEQUENCE_HISTORY
            or not np.isfinite(gps).all()):
        raise ValueError("invalid frozen V7 GPS stream dimensions")
    fixed = np.asarray(train_axis_fit["sigma_fixed_xyz"], dtype=np.float64)
    bins = np.asarray(train_axis_fit["sigma_bins_xyz"], dtype=np.float64)
    centers = np.asarray(train_axis_fit["centers"], dtype=np.float64)
    if (fixed.shape != (3,) or bins.shape != (5, 3)
            or centers.shape != (5,)
            or not np.allclose(centers, AXIS_BIN_CENTERS, atol=1e-12)
            or not np.isfinite(fixed).all()
            or not np.isfinite(bins).all()
            or np.any(fixed <= 0) or np.any(bins <= 0)
            or np.any(np.diff(bins, axis=0) > 1e-10)):
        raise ValueError("invalid training-only XYZ sigma mapping")
    n = len(gps)
    sigma = np.repeat(fixed[None, :], n, axis=0)
    required_model = CROSS_SEQUENCE_CASES[case]
    if required_model is None:
        if any(x is not None for x in (
                model, train_median, train_iqr, model_provenance)):
            raise ValueError("axis_fixed must not depend on any model")
    else:
        info = model_provenance
        if (model is None or train_median is None or train_iqr is None
                or not isinstance(info, dict)
                or info.get("case") != required_model
                or info.get("window") != 64
                or info.get("prediction_offsets") != [0, 1, 2]
                or info.get("model_is_development_only") is not True):
            raise ValueError("invalid or mismatched frozen model provenance")
        cfg = MultiSensorReliabilityConfig()
        features = gps_features(gps, dt=cfg.dt, window=cfg.gps_window)
        anchors, pred, _ = predict_frozen_gps_features_causally(
            model, features, train_median, train_iqr,
            window=64, clip=float(info["input_clip"]),
        )
        if not np.array_equal(
                anchors, np.arange(CROSS_SEQUENCE_HISTORY - 1, n)):
            raise AssertionError("V7 GPS prediction anchor misalignment")
        raw = np.column_stack([
            np.interp(pred[:, 0], centers, bins[:, axis])
            for axis in range(3)
        ])
        # Frozen maximum 4x GPS information relative to the
        # train-only fixed XYZ baseline; no validation tuning.
        sigma[anchors] = np.maximum(
            raw, fixed[None, :] / np.sqrt(AXIS_MAX_PRECISION_RATIO),
        )
    if (sigma.shape != (n, 3) or not np.isfinite(sigma).all()
            or np.any(sigma <= 0)):
        raise AssertionError("V7 GPS sigma stream became invalid")
    metadata = {
        "case": case, "frames": n, "first_model_frame": 63,
        "cold_start_frames": 63,
        "cold_start_sigma_source": "train_axis_fixed",
        "model_used": required_model,
        "head_used_for_fgo": "h0" if required_model else "none",
        "future_heads_used_for_fgo": False,
        "gps_precision_cap_ratio": (
            AXIS_MAX_PRECISION_RATIO if required_model else None
        ),
        "no_evaluation_reference_used": True,
        "development_model_only": True,
    }
    return sigma, metadata


def subgraph_measurements(m, *, begin=BEGIN, end=END):
    """Slice *only* the development VAL span, never the internal TEST.

    Rebase corrupted GPS to its initial corrupted value. Relative pose
    factors, IMU, timestamps and quality masks retain their original
    realization. Clean OXTS is intentionally unavailable here.
    """
    if (begin != BEGIN or end != END
            or np.asarray(m.gps_local).shape != (EXPECTED_N, 3)
            or np.asarray(m.imu_gyro).shape != (EXPECTED_N, 3)
            or len(m.timestamps) != EXPECTED_N):
        raise ValueError("unexpected measurements or 0027 VAL boundary")
    length = end - begin
    for key in ("lidar_between", "camera_between"):
        if np.asarray(getattr(m, key)).shape != (EXPECTED_N-1, 4, 4):
            raise ValueError(f"unexpected {key} shape")
    for key in ("lidar_valid", "camera_valid", "lidar_quality", "camera_quality"):
        if len(getattr(m, key)) != EXPECTED_N-1:
            raise ValueError(f"unexpected {key} length")
    src = np.asarray(m.gps_local, np.float64)[begin:end]
    if not np.isfinite(src).all():
        raise ValueError("invalid degraded GPS")
    if not np.isfinite(np.asarray(m.imu_gyro)[begin:end]).all():
        raise ValueError("invalid degraded IMU")
    gps = src.copy() - src[0].copy()
    sub = FourSensorMeasurements(
        gps_local=gps,
        imu_gyro=np.asarray(m.imu_gyro[begin:end], float).copy(),
        timestamps=list(m.timestamps[begin:end]),
        lidar_between=np.asarray(m.lidar_between[begin:end-1], float).copy(),
        lidar_valid=np.asarray(m.lidar_valid[begin:end-1], bool).copy(),
        lidar_quality=np.asarray(m.lidar_quality[begin:end-1], float).copy(),
        camera_between=np.asarray(m.camera_between[begin:end-1], float).copy(),
        camera_valid=np.asarray(m.camera_valid[begin:end-1], bool).copy(),
        camera_quality=np.asarray(m.camera_quality[begin:end-1], float).copy(),
    )
    if (len(sub.gps_local) != length
            or len(sub.lidar_valid) != length - 1):
        raise ValueError("subgraph length changed")
    return sub


def get_development_run(candidate: dict, name: str, seed: int) -> dict | None:
    base = name.removeprefix("axis_").removesuffix("_cap4")
    if base in ("fixed", "no_gps"):
        return None
    if base == "gru106_r":
        run = read_run(CAPACITY_OUTPUT, "gru", seed, candidate)
        if (run["metrics"].get("run_family") != "capacity_gru_h106"
                or run["metrics"]["config"]["hidden"] != 106
                or run["metrics"]["parameter_count"] != 139290):
            raise ValueError("incorrect GRU106 capacity-control provenance")
    else:
        backend = "gru" if base == "gru96_r" else "mamba"
        run = read_run(DEFAULT_OUTPUT, backend, seed, candidate)
        expected_params = 114630 if backend == "gru" else 139590
        if (run["metrics"]["config"]["hidden"] != 96
                or run["metrics"]["parameter_count"] != expected_params
                or run["metrics"].get("run_family", "baseline") != "baseline"):
            raise ValueError("incorrect baseline provenance")
    if run["metrics"]["config"]["layers"] != 2:
        raise ValueError("unexpected architecture")
    return run


def prepare_case(case: str, seed: int) -> tuple[np.ndarray, dict]:
    """Read-only, source-checked preflight; NO FGO or writes."""
    if case not in CASES or seed not in (43, 44, 45):
        raise ValueError("pilot limited to predeclared cases and seeds 43/44/45")
    policy = read_policy()
    validate_policy(policy)
    source = DEGRADED / "degraded_sensor_data.npz"
    if not source.is_file() or sha256(source) != (
            policy["raw_sources"]["degraded_sensor"]["sha256"]):
        raise ValueError("degraded GPS factor source differs from frozen V7")
    candidate = read_candidate()
    run = get_development_run(candidate, case, seed)
    axis_fit = load_train_axis_noise(policy) if case.startswith("axis_") else None
    capped = case.endswith("_cap4")
    raw_sigma_before_cap = None
    if run is None:
        if axis_fit is not None:
            sigma = np.tile(axis_fit["sigma_fixed_xyz"], (END-BEGIN, 1))
        else:
            sigma = build_gps_sigmas(None)
        prediction_sha = None
    else:
        arrays = run["arrays"]
        if (not np.array_equal(arrays["anchors"], candidate["anchors"])
                or not np.array_equal(candidate["anchors"],
                                      np.arange(BEGIN, END, dtype=np.int64))):
            raise ValueError("wrong timestamp or label source")
        if axis_fit is not None:
            sigma = axis_sigma_from_prediction(
                arrays["reliability"][:, 0], axis_fit,
            )
            if capped:
                raw_sigma_before_cap = sigma.copy()
                sigma = cap_axis_information(
                    sigma, axis_fit["sigma_fixed_xyz"],
                )
        else:
            sigma = build_gps_sigmas(
                arrays["reliability"][:, 0],
                (arrays["evidence_uncertainty"][:, 0]
                 if case == "mamba_r_u" else None),
            )
            if case in ("mamba_mean_sigma", "mamba_perm_sigma"):
                sigma = mamba_sigma_control(sigma, case, seed)
        base_case = case.removeprefix("axis_").removesuffix("_cap4")
        backend = "gru" if base_case.startswith("gru") else "mamba"
        path = (CAPACITY_OUTPUT if base_case == "gru106_r"
                else DEFAULT_OUTPUT) / backend / f"seed_{seed}" / "val_predictions.npz"
        prediction_sha = sha256(path)
    metadata = {
        "schema": "isj_v7_0027_dev_gps_fgo_pilot_v1",
        "development_only": True,
        "not_independent_test": True,
        "not_online_incremental_fgo": True,
        "reference_is_not_independent": True,
        "not_calibrated_covariance": True,
        "train_oracle_conditioning_only": case.startswith("axis_"),
        "not_calibrated_to_network_predictions": True,
        "post_hoc_4x_precision_cap": bool(capped),
        "precision_cap_basis": (
            "train_axis_fixed_sigma_xyz; 4x axis-specific inverse variance"
            if capped else None
        ),
        "gps_sigma_mapping": (
            "TRAIN-only fixed per-axis RMS (XYZ)" if case == "axis_fixed"
            else "TRAIN binned XYZ RMS from predicted r(t), "
                 "then cap each axis GPS information at 4x "
                 "the train-fixed-axis precision"
            if capped
            else "TRAIN oracle reliability bins -> predicted r(t) -> per-axis RMS"
            if case.startswith("axis_")
            else "none: GPS factors disabled" if case == "no_gps"
            else "5m fixed" if case == "fixed"
            else "mean over 680 Mamba-R sigmas" if case == "mamba_mean_sigma"
            else "deterministically permuted Mamba-R sigma sequence"
            if case == "mamba_perm_sigma"
            else "5m * sqrt(1 + 4*(1-r) + 0.5*U) "
                 "(U=0 for reliability-only)"
        ),
        "control_class": (
            "post_hoc_diagnostic_after_initial_FGO_result"
            if (case in ("no_gps", "mamba_mean_sigma", "mamba_perm_sigma")
                or case.startswith("axis_"))
            else "initial_exploratory_pilot"
        ),
        "gps_factors_enabled": case != "no_gps",
        "mamba_permutation_seed": (
            20261120 + seed if case == "mamba_perm_sigma" else None
        ),
        "sigma_vector_used_for_gps_factors": case != "no_gps",
        "case": case,
        "seed": seed,
        "interval_half_open": [BEGIN, END],
        "frames": END - BEGIN,
        "gating": (
            "GPS factors removed; IMU/LiDAR/Camera covariances fixed"
            if case == "no_gps"
            else "GPS sigma only; IMU/LiDAR/Camera covariances fixed"
        ),
        "used_head": (
            "none" if case in ("fixed", "no_gps", "axis_fixed")
            else "h0: GPS reliability predicted from observations <= t"
        ),
        "prediction_archive_sha256": prediction_sha,
        "gps_measurement_archive_sha256": sha256(source),
        "dataset_archive_sha256": candidate["archive_sha256"],
        "lidar_factor_archive_sha256": sha256(
            DEGRADED / "degraded_lidar_factor_data.npz"),
        "camera_factor_archive_sha256": sha256(
            DEGRADED / "degraded_camera_factor_data.npz"),
        "sigma_min": float(sigma.min()) if case != "no_gps" else None,
        "sigma_max": float(sigma.max()) if case != "no_gps" else None,
        "sigma_mean": float(sigma.mean()) if case != "no_gps" else None,
        "sigma_changed_fraction": (
            float(np.mean(np.abs(sigma - BASE_SIGMA) > 1e-6))
            if case != "no_gps" else None
        ),
        "sigma_mean_xyz": (np.mean(sigma, axis=0).tolist()
                           if sigma.ndim == 2 else None),
        "cap4_change_fraction_of_axes": (
            float(np.mean(sigma > raw_sigma_before_cap + 1e-12))
            if capped else None
        ),
        "max_precision_ratio_against_train_axis_fixed": (
            np.max((axis_fit["sigma_fixed_xyz"][None, :] / sigma)**2,
                   axis=0).tolist()
            if axis_fit is not None else None
        ),
        "train_axis_fit": (
            {
                "train_anchor_range_inclusive": axis_fit["train_anchors"],
                "bin_centers": axis_fit["centers"].tolist(),
                "bin_counts": axis_fit["counts"],
                "train_fixed_sigma_xyz": axis_fit["sigma_fixed_xyz"].tolist(),
                "train_binned_sigma_xyz": axis_fit["sigma_bins_xyz"].tolist(),
                "minimum_sigma_m": AXIS_STD_FLOOR_M,
                "synthetic_oracle_conditioned": True,
                "network_prediction_calibrated": False,
                "relative_gps_origin_error_correlation_modeled": False,
            } if axis_fit is not None else None
        ),
        "notes": [
            "Historical 0027 VAL is already used to select networks and early stop; metrics are biased developmental diagnostics.",
            "The FGO uses all sensor observations inside the 680-frame VAL batch (offline future context); NOT an online claim.",
            "Synthetic GPS target is read only for file-provenance checks, never for covariance or factor construction.",
            "Saved clean OXTS-derived GPS reference is reused for evaluation only; NOT independent GNSS truth.",
            "0016/0020 held-out drives not accessed; 0027 post-VAL source rows may be loaded from the archived NPZ but are not used in this graph or its metrics.",
            "No-GPS and Mamba sigma controls added AFTER seeing seed43 Fixed/Mamba-R outcomes. These are exploratory, not prespecified confirmatory tests.",
            "Axis mapping uses only synthetic oracle reliability and position-error RMS from frozen 0027 TRAIN anchors, never validation/test reference.",
            "Axis sigma ignores cross-axis and common-origin temporal correlation induced by GPS rebasing; not a full measurement noise model.",
            "The 4x cap is a POST-HOC single-setting exploratory test introduced after inspecting V7 development diagnostics. No VAL GT is used to construct sigma; outcomes are not independent proof.",
            "The constant-mean control preserves mean Mamba sigma, and permutation preserves the full empirical sigma distribution while breaking its time alignment.",
        ],
    }
    return sigma, metadata


def reference_diagnostics(trajectory: np.ndarray, clean: np.ndarray) -> dict:
    """Explicitly non-independent, fixed-origin OXTS-relative diagnostics."""
    trajectory = np.asarray(trajectory, dtype=np.float64)
    clean = np.asarray(clean, dtype=np.float64)
    if (trajectory.shape != (END-BEGIN, 3) or clean.shape != trajectory.shape
            or not np.isfinite(trajectory).all()
            or not np.isfinite(clean).all()):
        raise ValueError("invalid GPS and reference trajectories")
    clean0 = clean - clean[0].copy()
    # 3D local-origin point-wise error is NOT absolute geodetic ATE.
    diff = trajectory - clean0
    def rmse(a):
        return float(np.sqrt(np.mean(np.sum(a*a, axis=1))))
    return {
        "origin_aligned_oxTS_relative_3d_rmse_m": rmse(diff),
        "origin_aligned_oxTS_relative_2d_rmse_m": rmse(diff[:, :2]),
        "relative_displacement_1_frame_rmse_m": rmse(
            np.diff(trajectory, axis=0) - np.diff(clean0, axis=0)),
        "relative_displacement_10_frame_rmse_m": rmse(
            trajectory[10:] - trajectory[:-10]
            - (clean0[10:] - clean0[:-10])),
        "independent_ATE": False,
        "independent_RPE": False,
    }


def run(case: str, seed: int, *, preflight_only: bool) -> dict:
    sigma, metadata = prepare_case(case, seed)
    if preflight_only:
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        print("PILOT_PREFLIGHT_OK; READ_ONLY; NO_FGO; NO_HELDOUT")
        return metadata
    destination = OUTPUT / f"seed_{seed}" / case
    if destination.exists():
        raise FileExistsError(f"refuse overwrite of {destination}")
    m = load_degraded_four_sensor_measurements(DEGRADED)
    sub = subgraph_measurements(m)
    # All variants retain the same LiDAR-based initialization,
    # initial anchor and unchanged IMU/LiDAR/Camera covariances.
    # No-GPS control removes GPS factors entirely (zero GPS factors).
    result, poses, counts = run_four_sensor_factor_graph(
        sub, destination, mode="fixed",
        gps_sigmas_override=(None if case == "no_gps" else sigma),
        use_gps=(case != "no_gps"),
        use_imu=True, use_lidar=True, use_camera=True,
    )
    if (result.shape != (END-BEGIN, 3) or poses.shape != (END-BEGIN, 4, 4)
            or not np.isfinite(result).all()):
        raise RuntimeError("invalid optimized trajectory")
    # Critical: clean reference loaded only *after* graph optimization.
    with np.load(DEGRADED / "degraded_sensor_data.npz",
                 allow_pickle=False) as data:
        reference = np.asarray(data["clean_gps_local"][BEGIN:END], float)
    evaluation = reference_diagnostics(result, reference)
    metadata["factor_counts"] = {k: int(v) for k, v in counts.items()}
    metadata["evaluation"] = evaluation
    metadata["trajectory_sha256"] = sha256(destination / "trajectory.txt")
    metadata["status"] = "COMPLETE_DEVELOPMENT_PILOT"
    if case != "no_gps":
        np.save(destination / "gps_sigmas.npy", sigma)
    (destination / "pilot_metrics.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    print("GPS_FGO_PILOT_COMPLETE", case, seed)
    return metadata


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, choices=CASES)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--run-pilot", action="store_true",
                        help="explicitly run development-only GTSAM FGO; writes new folder")
    args = parser.parse_args(argv)
    run(args.case, args.seed, preflight_only=not args.run_pilot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
