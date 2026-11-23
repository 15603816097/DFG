from __future__ import annotations

"""Read-only GPS same-realization V7 candidate preflight, development 0027.

Rebuild input GPS features from the exact DEGRADED GPS measurements used
to generate V3 oracle labels. Verify that the saved V3 GPS labels equal
the known synthetic oracle error mapping. This is a candidate design
check, not permission to train or an independent ground-truth label.
No output files, raw KITTI access, checkpoints, or FGO.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.isj_v7.protocol import (
    DEFAULT_SOURCE,
    SENSOR_DIMS,
    aligned_targets,
    plan_splits,
    robust_train_scaler,
    sha256_file,
)
from experiments.isj_v7.audit_feature_label_streams import (
    DEFAULT_DEGRADED,
    discrepancy,
)
from src.multisensor_reliability.config import MultiSensorReliabilityConfig
from src.multisensor_reliability.features import gps_features
from src.reliability.oracle_factor_reliability import OracleReliabilityConfig


def _read(path: Path, fields: tuple[str, ...]) -> dict[str, np.ndarray]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        absent = [field for field in fields if field not in archive.files]
        if absent:
            raise KeyError(f"{path}: missing {absent}")
        return {field: np.asarray(archive[field]) for field in fields}


def _array_sha(values: np.ndarray) -> str:
    x = np.ascontiguousarray(np.asarray(values, dtype="<f4"))
    hash_value = hashlib.sha256()
    hash_value.update(str(tuple(x.shape)).encode("ascii"))
    hash_value.update(x.tobytes())
    return hash_value.hexdigest()


def preview_gps_rebuild(
    v3_path: Path = DEFAULT_SOURCE,
    degraded_path: Path = DEFAULT_DEGRADED,
    *,
    window: int = 64,
    horizon: int = 3,
) -> dict:
    """Describe *in memory* a candidate x(t) from the SAME label GPS stream.

    Important: the current labels are derived from clean OXTS/GPS. That
    clean reference is ONLY used in this label audit, never as input to
    gps_features(). No training, checkpoints, side effects or file writes.
    """
    v3_path = Path(v3_path)
    degraded_path = Path(degraded_path)
    for path in (v3_path, degraded_path):
        if any(part in {"0016", "0020"} for part in path.parts):
            raise ValueError("Reserved 0016/0020 drives are not allowed")
    old = _read(v3_path, (
        "gps_features", "gps_current_factor_reliability",
        "diagnostic_gps_error_m",
    ))
    physical = _read(degraded_path, (
        "degraded_gps_local", "clean_gps_local",
    ))
    x_old = np.asarray(old["gps_features"], dtype=np.float32)
    y_old = np.asarray(old["gps_current_factor_reliability"],
                       dtype=np.float32)
    diag = np.asarray(old["diagnostic_gps_error_m"], dtype=np.float64)
    z = np.asarray(physical["degraded_gps_local"], dtype=np.float64)
    clean = np.asarray(physical["clean_gps_local"], dtype=np.float64)
    n = len(y_old)
    if (n == 0 or x_old.shape != (n, SENSOR_DIMS["gps"]) or
            y_old.shape != (n,) or diag.shape != (n,) or
            z.shape != (n, 3) or clean.shape != (n, 3)):
        raise ValueError("GPS sources have inconsistent shapes/lengths")
    if not all(np.isfinite(a).all() for a in (
            x_old, y_old, diag, z, clean)):
        raise ValueError("GPS sources contain nonfinite values")
    if np.any((y_old < 0) | (y_old > 1)):
        raise ValueError("GPS labels must lie in [0,1]")

    # Only degraded (online-available) GPS stream is used for x.
    cfg = MultiSensorReliabilityConfig()
    x_candidate = gps_features(z, dt=cfg.dt, window=cfg.gps_window)
    error_m = np.linalg.norm(z - clean, axis=1)
    oracle_cfg = OracleReliabilityConfig()
    expected_labels = np.clip(
        np.exp(-np.square(error_m / oracle_cfg.gps_position_scale_m)),
        oracle_cfg.minimum_reliability,
        oracle_cfg.maximum_reliability,
    ).astype(np.float32)

    existing_feature_gap = discrepancy(x_old, x_candidate)
    diagnosis = discrepancy(diag, error_m, atol=1e-5, rtol=1e-6)
    label_check = discrepancy(y_old, expected_labels,
                              atol=3e-6, rtol=2e-5)
    plan = plan_splits(n, window=window, horizon=horizon)
    center, scale = robust_train_scaler(x_candidate, plan.train_end)

    splits = {}
    for part in ("train", "val", "test"):
        anchors = plan.indices(part)
        left, right = plan.boundary(part)
        if (anchors[0] < left and part != "train" or
                anchors[-1] + horizon - 1 >= right):
            raise AssertionError("cross-split target")
        target = aligned_targets(expected_labels, anchors, horizon)
        splits[part] = {
            "count": int(len(anchors)),
            "first_anchor": int(anchors[0]),
            "last_anchor": int(anchors[-1]),
            "last_target": int(anchors[-1] + horizon - 1),
            "targets_from_same_stream_sha256": _array_sha(target),
        }

    good_label = diagnosis["allclose"] and label_check["allclose"]
    return {
        "schema": "isj_v7_gps_canonical_dryrun_v1",
        "v3_source_sha256": sha256_file(v3_path),
        "degraded_source_sha256": sha256_file(degraded_path),
        "frames": n,
        "old_v3_features_vs_canonical": existing_feature_gap,
        "saved_gps_error_vs_same_stream_oracle_error": diagnosis,
        "saved_gps_reliability_vs_same_stream_oracle_mapping": label_check,
        "canonical_feature_sha256_float32": _array_sha(x_candidate),
        "candidate_labels_sha256_float32": _array_sha(expected_labels),
        "candidate_train_median_sha256_float32": _array_sha(center),
        "candidate_train_iqr_sha256_float32": _array_sha(scale),
        "window": window, "horizon": horizon,
        "head_offsets": list(plan.offsets),
        "split_train_end": plan.train_end,
        "split_val_end": plan.val_end,
        "samples": splits,
        "gps_same_realization_candidate_consistent": bool(good_label),
        "gps_clean_reference_used_as_network_input": False,
        "gps_supervision": (
            "Synthetic clean-OXTS-vs-degraded GPS oracle; available offline "
            "for supervised diagnostics, NOT a live deployable measurement "
            "or independent physical GPS ground truth."
        ),
        "independent_test_0020_accessed": False,
        "approved_for_final_training": False,
        "next_gate": (
            "Define robust source and measurement/reference independence "
            "for all modalities, check feature and label provenance, freeze "
            "evaluation and model config before final training."
        ),
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--v3", type=Path, default=DEFAULT_SOURCE)
    p.add_argument("--degraded", type=Path, default=DEFAULT_DEGRADED)
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--horizon", type=int, default=3)
    a = p.parse_args(argv)
    report = preview_gps_rebuild(
        a.v3, a.degraded, window=a.window, horizon=a.horizon,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("DRY-RUN ONLY. Do not overwrite V3 or launch final training.")


if __name__ == "__main__":
    main()
