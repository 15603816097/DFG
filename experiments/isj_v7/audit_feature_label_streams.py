from __future__ import annotations

"""0027 read-only V7 feature/label sensor-stream consistency audit.

Compares legacy V3 archived model INPUT features against features
reconstructed from the actual DEGRADED GPS/IMU measurements that V3
used to calculate its oracle LABELS. The GPS candidate is deliberately
not substituted for the frozen V3 inputs. No training, no model load,
no OXTS raw reads, no optimization and no writes.

This audit tests *realization consistency*, not full feature causality,
independent labels, or train/test generalization.
"""

import argparse
import json
from pathlib import Path
import numpy as np

from experiments.isj_v7.protocol import ROOT, DEFAULT_SOURCE, sha256_file
from src.multisensor_reliability.features import gps_features, imu_features
from src.multisensor_reliability.config import MultiSensorReliabilityConfig

DEFAULT_DEGRADED = (ROOT / "results/degraded_four_sensor_measurements"
                    / "degraded_sensor_data.npz")
DEFAULT_DEBUG = (ROOT / "results/predictive_factor_reliability_v2_corrected"
                 / "corrected_feature_debug.npz")


def discrepancy(left, right, *, atol=1e-4, rtol=1e-5) -> dict:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if x.shape != y.shape or x.ndim not in (1, 2):
        raise ValueError(f"shape mismatch: {x.shape} vs {y.shape}")
    if x.size == 0 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("invalid or nonfinite data")
    diff = np.abs(x - y)
    fail = ~np.isclose(x, y, atol=atol, rtol=rtol)
    fail_rows = fail if x.ndim == 1 else np.any(fail, axis=1)
    return {
        "shape": list(x.shape),
        "allclose": bool(not fail.any()),
        "mismatched_values": int(np.count_nonzero(fail)),
        "mismatched_rows": int(np.count_nonzero(fail_rows)),
        "max_abs_delta": float(np.max(diff)),
        "mean_abs_delta": float(np.mean(diff)),
        "atol": atol,
        "rtol": rtol,
    }


def _checked_npz(path: Path, fields: tuple[str, ...]) -> dict[str, np.ndarray]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as d:
        for name in fields:
            if name not in d.files:
                raise KeyError(f"{path}: missing {name}")
        return {name: np.asarray(d[name]) for name in fields}


def audit_sources(
    v3_path: Path = DEFAULT_SOURCE,
    degraded_path: Path = DEFAULT_DEGRADED,
    debug_path: Path = DEFAULT_DEBUG,
) -> dict:
    """Read-only comparison of V3 label measurement streams and input features."""
    v3_path = Path(v3_path)
    degraded_path = Path(degraded_path)
    debug_path = Path(debug_path)
    v3 = _checked_npz(v3_path, (
        "gps_features", "imu_features", "diagnostic_gps_error_m",
    ))
    degraded = _checked_npz(degraded_path, (
        "degraded_gps_local", "clean_gps_local",
        "degraded_imu_acc", "degraded_imu_gyro",
    ))
    cfg = MultiSensorReliabilityConfig()
    gps_saved = np.asarray(v3["gps_features"], dtype=np.float32)
    imu_saved = np.asarray(v3["imu_features"], dtype=np.float32)
    gps_raw = np.asarray(degraded["degraded_gps_local"], dtype=np.float64)
    clean = np.asarray(degraded["clean_gps_local"], dtype=np.float64)
    acc = np.asarray(degraded["degraded_imu_acc"], dtype=np.float64)
    gyro = np.asarray(degraded["degraded_imu_gyro"], dtype=np.float64)
    n = len(gps_saved)
    if (gps_saved.shape != (n, 21) or imu_saved.shape != (n, 20) or
            gps_raw.shape != (n, 3) or clean.shape != (n, 3) or
            acc.shape != (n, 3) or gyro.shape != (n, 3) or
            np.asarray(v3["diagnostic_gps_error_m"]).shape != (n,)):
        raise ValueError("V3/degraded array frame count or dimension mismatch")
    if not all(np.isfinite(v).all() for v in (
            gps_saved, imu_saved, gps_raw, clean, acc, gyro)):
        raise ValueError("V3/degraded array contains nonfinite values")
    gps_rebuilt = gps_features(gps_raw, dt=cfg.dt, window=cfg.gps_window)
    imu_rebuilt = imu_features(acc, gyro, window=cfg.imu_window)
    oracle_error = np.linalg.norm(gps_raw - clean, axis=1)
    comparisons = {
        "v3_gps_features_vs_features_from_label_gps":
            discrepancy(gps_saved, gps_rebuilt),
        "v3_imu_features_vs_features_from_label_imu":
            discrepancy(imu_saved, imu_rebuilt),
        "v3_diagnostic_gps_error_vs_degraded_clean_delta":
            discrepancy(v3["diagnostic_gps_error_m"], oracle_error,
                        atol=1e-5, rtol=1e-6),
    }
    debug = {"available": debug_path.is_file(),
             "path": str(debug_path)}
    if debug_path.is_file():
        source = _checked_npz(debug_path, ("gps_local_corrupted",))
        source_gps = np.asarray(source["gps_local_corrupted"], dtype=np.float64)
        debug["v2_debug_gps_vs_v3_label_gps"] = discrepancy(source_gps, gps_raw)
        debug["v3_gps_features_vs_v2_debug_gps_features"] = discrepancy(
            gps_saved,
            gps_features(source_gps, dt=cfg.dt, window=cfg.gps_window),
        )
        debug["source_sha256"] = sha256_file(debug_path)

    gps_same = comparisons[
        "v3_gps_features_vs_features_from_label_gps"]["allclose"]
    imu_same = comparisons[
        "v3_imu_features_vs_features_from_label_imu"]["allclose"]
    gps_label_consistent = comparisons[
        "v3_diagnostic_gps_error_vs_degraded_clean_delta"]["allclose"]
    return {
        "schema": "isj_v7_stream_consistency_v1",
        "frames": n,
        "v3_archive_sha256": sha256_file(v3_path),
        "degraded_sensor_archive_sha256": sha256_file(degraded_path),
        "input_feature_source_reconstructed": {
            "gps": "src.multisensor_reliability.features.gps_features",
            "imu": "src.multisensor_reliability.features.imu_features",
        },
        "comparisons": comparisons,
        "debug": debug,
        "gps_input_matches_gps_label_measurement": gps_same,
        "imu_input_matches_imu_label_measurement": imu_same,
        "gps_diagnostic_matches_label_measurement": gps_label_consistent,
        "source_static_warning": (
            "V2 corrected feature builder calls corrupt_gps with RNG stream 1 "
            "(9176 offset); degraded measurement builder calls corrupt_local_gps "
            "with offset 31. Similar plan severity does not imply identical "
            "GPS realizations. Verify on saved data using the numeric comparisons."
        ),
        "feature_time_causality_assessment": (
            "gps_features and imu_features use past/current windows in inspected "
            "source, but offline preprocessing and all V3 label dependencies "
            "have not been proven causal or independent."
        ),
        "camera_source_assessment": (
            "V2 camera features are extracted from corrupted single-frame "
            "left images; V3 camera targets from degraded stereo VO + full "
            "trajectory FGO. This script does not numerically reconstruct "
            "camera pixels or prove same stereo factor realization."
        ),
        "approved_for_final_training": False,
        "status": (
            "GPS_REALIZATION_MISMATCH" if not gps_same else
            "GPS_MATCH_BUT_LABEL_PROVENANCE_UNAPPROVED"
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v3", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--degraded", type=Path, default=DEFAULT_DEGRADED)
    parser.add_argument("--v2-debug", type=Path, default=DEFAULT_DEBUG)
    args = parser.parse_args(argv)
    result = audit_sources(args.v3, args.degraded, args.v2_debug)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("STOP: no V7 training on V3 labels until source alignment and "
          "independent-supervision policy are resolved.")


if __name__ == "__main__":
    main()
