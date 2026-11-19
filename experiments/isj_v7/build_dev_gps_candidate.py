from __future__ import annotations

"""Materialize a strictly DEVELOPMENT-ONLY GPS same-realization dataset.

Outputs only a 0027 GPS training/validation candidate, never the 0027
internal-test partition. Uses only the degraded GPS stream for features;
clean GPS is touched exclusively by the existing oracle-label check.
No V3/V5/V6 overwrite, no model training, no FGO, no 0016/0020 access.

CLI --write-dev-candidate checks frozen policy and both source SHA256s,
and refuses to overwrite an existing candidate directory.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.isj_v7.audit_gps_canonical_dryrun import (
    _array_sha, preview_gps_rebuild,
)
from experiments.isj_v7.audit_feature_label_streams import DEFAULT_DEGRADED
from experiments.isj_v7.protocol import (
    DEFAULT_SOURCE, ROOT, aligned_targets, plan_splits,
    robust_train_scaler,
)
from experiments.isj_v7.supervision_gate import (
    read_policy, validate_policy, verify_file_hashes,
)
from src.multisensor_reliability.config import MultiSensorReliabilityConfig
from src.multisensor_reliability.features import gps_features


DEST_BASE = ROOT / "results" / "isj_v7"
DEFAULT_DEST = DEST_BASE / "dev_gps_0027_same_source_v1"
SENSOR = "gps"


def _load_single(path: Path, key: str) -> np.ndarray:
    if not Path(path).is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        if key not in data.files:
            raise KeyError(f"{path}: {key}")
        return np.asarray(data[key])


def prepare_dev_candidate(
    v3_path: Path = DEFAULT_SOURCE,
    degraded_path: Path = DEFAULT_DEGRADED,
    *,
    window: int = 64,
    horizon: int = 3,
) -> tuple[dict[str, np.ndarray], dict]:
    """In-memory artifact plan. No writes and no old-checkpoint loading."""
    audit = preview_gps_rebuild(
        v3_path, degraded_path, window=window, horizon=horizon,
    )
    if not audit["gps_same_realization_candidate_consistent"]:
        raise ValueError("GPS oracle/source consistency preflight failed")
    label = np.asarray(
        _load_single(v3_path, "gps_current_factor_reliability"),
        dtype=np.float32,
    )
    degraded = np.asarray(
        _load_single(degraded_path, "degraded_gps_local"),
        dtype=np.float64,
    )
    cfg = MultiSensorReliabilityConfig()
    features = gps_features(
        degraded, dt=cfg.dt, window=cfg.gps_window,
    )
    n = len(label)
    if (features.shape != (n, 21) or
            not np.isfinite(features).all() or
            not np.isfinite(label).all()):
        raise ValueError("invalid GPS candidate dimensions/values")
    if _array_sha(features) != audit["canonical_feature_sha256_float32"]:
        raise ValueError("candidate changed since source-lineage preflight")
    if _array_sha(label) != audit["candidate_labels_sha256_float32"]:
        raise ValueError("label changed since source-lineage preflight")
    plan = plan_splits(n, window=window, horizon=horizon)
    if n != 4544:
        # This helper supports small synthetic fixtures via n>0; CLI
        # production gate below separately requires 4544 source frames.
        pass
    mean, scale = robust_train_scaler(features, plan.train_end)
    train = plan.indices("train")
    val = plan.indices("val")
    if train[-1] + horizon - 1 >= plan.train_end:
        raise AssertionError("train targets cross into VAL")
    if val[0] < plan.train_end or val[-1] + horizon - 1 >= plan.val_end:
        raise AssertionError("VAL targets cross into internal TEST")
    # Drop 0027 internal test completely, including its raw features
    # and ALL labels. Training arrays contain no clean/oracle positions.
    data = {
        "gps_features": features[:plan.val_end].copy(),
        "train_anchors": train,
        "train_targets": aligned_targets(label, train, horizon),
        "val_anchors": val,
        "val_targets": aligned_targets(label, val, horizon),
        "train_median": mean,
        "train_iqr": scale,
    }
    if _array_sha(data["train_median"]) != audit[
        "candidate_train_median_sha256_float32"
    ] or _array_sha(data["train_iqr"]) != audit[
        "candidate_train_iqr_sha256_float32"
    ]:
        raise ValueError("scaler train-prefix digest disagrees with preflight")
    fingerprints = {key: _array_sha(value) for key, value in data.items()}
    # _array_sha canonicalizes values to float32, so dtype is documented
    # separately for integer anchor arrays.
    dtypes = {key: str(value.dtype) for key, value in data.items()}
    info = {
        "schema": "isj_v7_dev_gps_same_source_v1",
        "development_only": True,
        "approved_for_final_training": False,
        "approved_for_independent_test": False,
        "source_drive": "2011_10_03_drive_0027_sync",
        "source_frames": n,
        "exported_feature_frames": plan.val_end,
        "omitted_internal_test_range": [plan.val_end, n],
        "omitted_raw_clean_gps": True,
        "omitted_corruption_severity_and_mode": True,
        "label_meaning": (
            "Synthetic GPS corruption reliability from original clean "
            "OXTS/GPS vs the same degraded GPS realization; NOT "
            "independent absolute positioning ground truth."
        ),
        "input_meaning": (
            "Past/current degraded GPS only; rolling gps_features with "
            f"dt={cfg.dt} and window={cfg.gps_window}."
        ),
        "forecast_heads": list(plan.offsets),
        "window": window,
        "train_end_exclusive": plan.train_end,
        "val_end_exclusive": plan.val_end,
        "train_anchor_count": int(len(train)),
        "val_anchor_count": int(len(val)),
        "normalize": "(x - train_median) / train_iqr; fit on x[:train_end] only",
        "source_archive_sha256": {
            "v3": audit["v3_source_sha256"],
            "degraded_sensor": audit["degraded_source_sha256"],
        },
        "array_sha256_float32": fingerprints,
        "array_dtypes": dtypes,
        "do_not_treat_as_independent_truth": True,
        "release_gates_unchanged": True,
    }
    return data, info


def write_candidate(
    destination: Path, arrays: dict[str, np.ndarray], metadata: dict,
    *, allowed_parent: Path = DEST_BASE,
) -> tuple[Path, Path]:
    """New-directory-only write; no overwriting any existing files."""
    allowed_parent = Path(allowed_parent).resolve()
    target = Path(destination).resolve()
    if target == allowed_parent or not target.is_relative_to(allowed_parent):
        raise ValueError("candidate destination must be inside allowed parent")
    if target.exists():
        raise FileExistsError(f"refusing to overwrite {target}")
    if metadata.get("approved_for_final_training") is not False or not (
        metadata.get("development_only") is True
    ):
        raise ValueError("candidate artifact must remain development-only")
    allowed_parent.mkdir(parents=True, exist_ok=True)
    target.mkdir(parents=True, exist_ok=False)
    archive = target / "gps_train_val.npz"
    manifest = target / "provenance.json"
    np.savez_compressed(archive, **arrays)
    manifest.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return archive, manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-dev-candidate", action="store_true",
                        help="create new GPS-only train+val development archive")
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    args = parser.parse_args(argv)
    policy = read_policy()
    validate_policy(policy)
    verify_file_hashes({
        key: policy["raw_sources"][key]
        for key in ("v3", "degraded_sensor")
    })
    data, metadata = prepare_dev_candidate()
    if metadata["source_frames"] != 4544 or metadata["exported_feature_frames"] != 3862:
        raise ValueError("frozen 0027 development shape unexpectedly changed")
    if not args.write_dev_candidate:
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        print("READ-ONLY PREVIEW: No candidate files written.")
        return 0
    archive, manifest = write_candidate(args.dest, data, metadata)
    print("GPS_DEV_ARCHIVE:", archive)
    print("GPS_DEV_MANIFEST:", manifest)
    print("Development-only candidate written; no training launched; "
          "old V3/V5/V6 unaffected, 0027 internal test omitted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
