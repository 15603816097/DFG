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
    CAPACITY_OUTPUT, DEFAULT_OUTPUT,
)
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
CASES = ("fixed", "gru96_r", "gru106_r", "mamba_r", "mamba_r_u")
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
    if name == "fixed":
        return None
    if name == "gru106_r":
        run = read_run(CAPACITY_OUTPUT, "gru", seed, candidate)
        if (run["metrics"].get("run_family") != "capacity_gru_h106"
                or run["metrics"]["config"]["hidden"] != 106
                or run["metrics"]["parameter_count"] != 139290):
            raise ValueError("incorrect GRU106 capacity-control provenance")
    else:
        backend = "gru" if name == "gru96_r" else "mamba"
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
    if run is None:
        sigma = build_gps_sigmas(None)
        prediction_sha = None
    else:
        arrays = run["arrays"]
        if (not np.array_equal(arrays["anchors"], candidate["anchors"])
                or not np.array_equal(candidate["anchors"],
                                      np.arange(BEGIN, END, dtype=np.int64))):
            raise ValueError("wrong timestamp or label source")
        sigma = build_gps_sigmas(
            arrays["reliability"][:, 0],
            (arrays["evidence_uncertainty"][:, 0]
             if case == "mamba_r_u" else None),
        )
        backend = "gru" if case.startswith("gru") else "mamba"
        path = (CAPACITY_OUTPUT if case == "gru106_r"
                else DEFAULT_OUTPUT) / backend / f"seed_{seed}" / "val_predictions.npz"
        prediction_sha = sha256(path)
    metadata = {
        "schema": "isj_v7_0027_dev_gps_fgo_pilot_v1",
        "development_only": True,
        "not_independent_test": True,
        "not_online_incremental_fgo": True,
        "reference_is_not_independent": True,
        "not_calibrated_covariance": True,
        "gps_sigma_mapping": (
            "5m * sqrt(1 + 4*(1-r) + 0.5*U), "
            "omit terms for reliability-only/fixed cases"
        ),
        "case": case,
        "seed": seed,
        "interval_half_open": [BEGIN, END],
        "frames": END - BEGIN,
        "gating": "GPS only; IMU/LiDAR/Camera covariances fixed",
        "used_head": "h0: GPS reliability predicted from observations <= t",
        "prediction_archive_sha256": prediction_sha,
        "gps_measurement_archive_sha256": sha256(source),
        "dataset_archive_sha256": candidate["archive_sha256"],
        "lidar_factor_archive_sha256": sha256(
            DEGRADED / "degraded_lidar_factor_data.npz"),
        "camera_factor_archive_sha256": sha256(
            DEGRADED / "degraded_camera_factor_data.npz"),
        "sigma_min": float(sigma.min()),
        "sigma_max": float(sigma.max()),
        "sigma_mean": float(sigma.mean()),
        "sigma_changed_fraction": float(np.mean(
            np.abs(sigma - BASE_SIGMA) > 1e-6)),
        "notes": [
            "Historical 0027 VAL is already used to select networks and early stop; metrics are biased developmental diagnostics.",
            "The FGO uses all sensor observations inside the 680-frame VAL batch (offline future context); NOT an online claim.",
            "Synthetic GPS target is read only for file-provenance checks, never for covariance or factor construction.",
            "Saved clean OXTS-derived GPS reference is reused for evaluation only; NOT independent GNSS truth.",
            "0016/0020 held-out drives not accessed; 0027 post-VAL source rows may be loaded from the archived NPZ but are not used in this graph or its metrics.",
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
    # All four FGO variants run from the same initial LiDAR trajectory
    # with identical initial prior and fixed other-sensor covariances.
    # The sole varying graph input is the explicit GPS sigma vector.
    result, poses, counts = run_four_sensor_factor_graph(
        sub, destination, mode="fixed",
        gps_sigmas_override=sigma,
        use_gps=True, use_imu=True, use_lidar=True, use_camera=True,
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
