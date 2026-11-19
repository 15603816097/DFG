from __future__ import annotations

"""READ-ONLY V7 factor-label lineage audit on known 0027 archives.

Reconstruct the V3 Camera and physical-LiDAR proxy labels from exactly
their saved measured relative-pose factors and the saved fixed-FGO
trajectory. Confirm saved diagnostics/labels match source calculation.
This is a provenance audit, NOT independent sensor-truth validation.

The fixed-FGO reference was optimized over full 0027 and may include
the same sensors being labeled. No V7 training approval follows.
No raw KITTI, 0016/0020, inference, FGO or writes.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.isj_v7.protocol import DEFAULT_SOURCE, ROOT, sha256_file
from src.reliability.oracle_factor_reliability import (
    OracleReliabilityConfig,
    relative_transform,
    relative_pose_error,
)

CAMERA_DEFAULT = (
    ROOT / "results/degraded_four_sensor_measurements"
    / "degraded_camera_factor_data.npz"
)
LIDAR_DEFAULT = (
    ROOT / "results/lidar_physical_oracle_benchmark"
    / "physical_lidar_factor_data.npz"
)
REFERENCE_DEFAULT = ROOT / "results/four_sensor_fixed_fg/poses.npy"


def _npz(path: Path, keys: tuple[str, ...]) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as d:
        absent = [name for name in keys if name not in d.files]
        if absent:
            raise KeyError(f"{path}: missing {absent}")
        return {k: np.asarray(d[k]) for k in keys}


def compare_special(saved: np.ndarray, expected: np.ndarray,
                    *, atol: float = 2e-5, rtol: float = 3e-5) -> dict:
    """Numerical match including identical signed infinities on invalid pairs."""
    left = np.asarray(saved, dtype=np.float64)
    right = np.asarray(expected, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("expected aligned 1D arrays")
    good_inf = np.isinf(left) & np.isinf(right) & (
        np.signbit(left) == np.signbit(right))
    finite = np.isfinite(left) & np.isfinite(right)
    matching = good_inf.copy()
    matching[finite] = np.isclose(left[finite], right[finite],
                                  atol=atol, rtol=rtol)
    differences = np.abs(left[finite] - right[finite])
    return {
        "allclose": bool(np.all(matching)),
        "mismatched_frames": int(np.count_nonzero(~matching)),
        "finite_compared_frames": int(np.count_nonzero(finite)),
        "matched_infinite_frames": int(np.count_nonzero(good_inf)),
        "max_abs_delta_finite": (float(np.max(differences))
                                 if len(differences) else None),
    }


def derive_proxy(
    between: np.ndarray,
    valid: np.ndarray,
    reference_poses: np.ndarray,
    *,
    translation_scale_m: float,
    rotation_scale_deg: float,
    minimum: float = 0.01,
    maximum: float = 1.0,
) -> dict[str, np.ndarray]:
    """Replicate the old proxy mapping; NOT independent label generation."""
    measured = np.asarray(between, dtype=np.float64)
    flags = np.asarray(valid, dtype=bool)
    reference = np.asarray(reference_poses, dtype=np.float64)
    n = len(reference)
    if (n < 2 or reference.shape != (n, 4, 4) or
            measured.shape != (n - 1, 4, 4) or
            flags.shape != (n - 1,) or
            not np.isfinite(reference).all() or
            not 0 < minimum <= maximum <= 1 or
            translation_scale_m <= 0 or rotation_scale_deg <= 0):
        raise ValueError("invalid factor/reference arrays")
    if not np.all(np.isfinite(measured[flags])):
        raise ValueError("valid factor contains nonfinite values")

    te = np.zeros(n, dtype=np.float64)
    re = np.zeros(n, dtype=np.float64)
    reliability = np.ones(n, dtype=np.float64)
    for i in range(n - 1):
        j = i + 1  # each pair (i -> i+1) aligns to TARGET frame j
        if not flags[i]:
            te[j] = np.inf
            re[j] = np.inf
            reliability[j] = minimum
            continue
        ref_between = relative_transform(reference[i], reference[j])
        te[j], re[j] = relative_pose_error(measured[i], ref_between)
        reliability[j] = np.clip(
            np.exp(-((te[j] / translation_scale_m) ** 2)
                   - ((re[j] / rotation_scale_deg) ** 2)),
            minimum, maximum,
        )
    return {"translation_error": te,
            "rotation_error_deg": re,
            "reliability": reliability}


def audit_factor_label_lineage(
    v3_path: Path = DEFAULT_SOURCE,
    lidar_path: Path = LIDAR_DEFAULT,
    camera_path: Path = CAMERA_DEFAULT,
    reference_path: Path = REFERENCE_DEFAULT,
) -> dict:
    sources = [Path(v3_path), Path(lidar_path), Path(camera_path),
               Path(reference_path)]
    if any("0016" in p.parts or "0020" in p.parts for p in sources):
        raise ValueError("reserved drives 0016/0020 may not be accessed")
    v3_keys = tuple(
        f"{sensor}_{suffix}" for sensor in ("lidar", "camera")
        for suffix in (
            "current_factor_reliability",
            "features",
        )
    ) + tuple(
        f"diagnostic_{sensor}_{suffix}" for sensor in ("lidar", "camera")
        for suffix in ("translation_error_m", "rotation_error_deg")
    )
    v3 = _npz(Path(v3_path), v3_keys)
    lidar = _npz(Path(lidar_path), ("body_between", "converged"))
    camera = _npz(Path(camera_path), ("between_measurements", "valid"))
    if not Path(reference_path).is_file():
        raise FileNotFoundError(reference_path)
    reference = np.load(reference_path, allow_pickle=False)
    n = len(reference)
    if (reference.shape != (n, 4, 4) or
            np.asarray(v3["lidar_features"]).shape != (n, 19) or
            np.asarray(v3["camera_features"]).shape != (n, 16) or
            np.asarray(lidar["body_between"]).shape != (n - 1, 4, 4) or
            np.asarray(camera["between_measurements"]).shape != (n - 1, 4, 4) or
            np.asarray(lidar["converged"]).shape != (n - 1,) or
            np.asarray(camera["valid"]).shape != (n - 1,)):
        raise ValueError("V3 factor and reference shapes do not match")
    lid_between = np.asarray(lidar["body_between"], dtype=np.float64)
    lid_valid = np.all(np.isfinite(lid_between), axis=(1, 2)) & np.asarray(
        lidar["converged"], dtype=bool)
    cam_valid = np.asarray(camera["valid"], dtype=bool)
    cfg = OracleReliabilityConfig()
    source_data = {
        "lidar": (
            lid_between, lid_valid,
            cfg.lidar_translation_scale_m, cfg.lidar_rotation_scale_deg,
        ),
        "camera": (
            camera["between_measurements"], cam_valid,
            cfg.camera_translation_scale_m, cfg.camera_rotation_scale_deg,
        ),
    }
    audits = {}
    for sensor, (between, valid, t_scale, r_scale) in source_data.items():
        calculated = derive_proxy(
            between, valid, reference,
            translation_scale_m=t_scale,
            rotation_scale_deg=r_scale,
            minimum=cfg.minimum_reliability,
            maximum=cfg.maximum_reliability,
        )
        comparisons = {
            "current_reliability": compare_special(
                v3[f"{sensor}_current_factor_reliability"],
                calculated["reliability"], atol=5e-6, rtol=3e-5),
            "translation_error_m": compare_special(
                v3[f"diagnostic_{sensor}_translation_error_m"],
                calculated["translation_error"]),
            "rotation_error_deg": compare_special(
                v3[f"diagnostic_{sensor}_rotation_error_deg"],
                calculated["rotation_error_deg"]),
        }
        audits[sensor] = {
            "factor_pairs": int(len(valid)),
            "valid_pairs": int(np.count_nonzero(valid)),
            "invalid_pairs": int(np.count_nonzero(~valid)),
            "source_factor_vs_v3_label": comparisons,
            "all_saved_proxy_fields_match": all(
                v["allclose"] for v in comparisons.values()),
            "feature_input_vs_factor_raw_measurement_verified": False,
        }
    return {
        "schema": "isj_v7_camera_lidar_factor_proxy_lineage_v1",
        "frames": int(n),
        "source_sha256": {
            "v3": sha256_file(Path(v3_path)),
            "physical_lidar_factors": sha256_file(Path(lidar_path)),
            "degraded_camera_factors": sha256_file(Path(camera_path)),
            "full_trajectory_fixed_fgo_reference": sha256_file(Path(reference_path)),
        },
        "label_aligned_to": "factor target frame i+1",
        "reference_dependency": (
            "FULL-TRAJECTORY fixed-FGO poses, not independent sensor truth. "
            "Fixed-FGO optimization uses multiple sensors and future measurements."
        ),
        "features": (
            "V3 LiDAR features are per-scan physical degradation statistics, "
            "Camera features are per-left-image degradation statistics; "
            "this audit does NOT verify original pixels/point clouds were "
            "the factor estimator's exact inputs."
        ),
        "modalities": audits,
        "saved_factor_label_lineage_matches": all(
            r["all_saved_proxy_fields_match"] for r in audits.values()),
        "approved_for_final_training": False,
        "independent_evaluation_established": False,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--v3", type=Path, default=DEFAULT_SOURCE)
    p.add_argument("--lidar", type=Path, default=LIDAR_DEFAULT)
    p.add_argument("--camera", type=Path, default=CAMERA_DEFAULT)
    p.add_argument("--reference", type=Path, default=REFERENCE_DEFAULT)
    args = p.parse_args(argv)
    result = audit_factor_label_lineage(
        args.v3, args.lidar, args.camera, args.reference)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("READ-ONLY V7 FACTOR PROVENANCE: proxy matching is NOT independent "
          "truth and does NOT authorize final-model training.")


if __name__ == "__main__":
    main()
