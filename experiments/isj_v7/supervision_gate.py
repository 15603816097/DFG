from __future__ import annotations

"""Read-only V7 research supervision and evaluation policy gate.

This is a development-only *review* gate, not training authorisation.
It refuses treating whole-trajectory FGO proxy labels or synthetic
OXTS/GPS supervision as independent physical ground truth.
Only known legacy 0027 hashes are checked; never inspect reserved
0016/0020. No network training, FGO, model evaluation or data writes.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = Path(__file__).with_name("supervision_policy_v1.json")
HEX_SHA = re.compile(r"^[0-9a-f]{64}$")
SENSORS = ("gps", "imu", "lidar", "camera")
SOURCES = (
    "v3", "degraded_sensor", "physical_lidar_factors",
    "degraded_camera_factors", "full_trajectory_fgo_poses",
)


def validate_policy(policy: dict) -> dict:
    """Fail closed when a historical oracle proxy is marked as approved."""
    if not isinstance(policy, dict):
        raise ValueError("policy must be an object")
    if policy.get("schema") != "isj_v7_supervision_policy_v1":
        raise ValueError("unsupported supervision policy schema")
    if policy.get("status") != "development_only":
        raise ValueError("V7 Stage 3 status must be development_only")
    if policy.get("development_drive") != "2011_10_03_drive_0027_sync":
        raise ValueError("this gate is scoped only to 0027 development")
    if policy.get("heldout_drives") != ["0016", "0020"]:
        raise ValueError("0016 and 0020 must remain reserved")
    if policy.get("legacy_0027_internal_test_is_independent") is not False:
        raise ValueError("0027 internal test is not independent")
    time = policy.get("temporal_protocol", {})
    if (time.get("window") != 64 or
            time.get("head_offsets") != [0, 1, 2] or
            time.get("train_end_exclusive") != 3180 or
            time.get("val_end_exclusive") != 3862 or
            time.get("normalizer") != "train_only_median_iqr"):
        raise ValueError("unexpected or unreviewed time split configuration")

    sources = policy.get("raw_sources")
    if not isinstance(sources, dict) or set(sources) != set(SOURCES):
        raise ValueError("missing or unexpected source archives")
    for name, record in sources.items():
        if not isinstance(record, dict):
            raise ValueError(f"{name} source metadata missing")
        raw = record.get("path")
        checksum = record.get("sha256")
        if (not isinstance(raw, str) or not raw or
                not isinstance(checksum, str) or
                not HEX_SHA.fullmatch(checksum)):
            raise ValueError(f"{name} requires path and SHA256")
        path = Path(raw)
        if (path.is_absolute() or ".." in path.parts or
                any("0016" in part or "0020" in part
                    for part in path.parts)):
            raise ValueError(f"{name}: unsafe or reserved source path")

    modality = policy.get("modalities")
    if not isinstance(modality, dict) or set(modality) != set(SENSORS):
        raise ValueError("missing or unexpected modalities")
    for name in SENSORS:
        row = modality[name]
        if not isinstance(row, dict) or row.get("independent_ground_truth") is not False:
            raise ValueError(f"{name} must not claim independent ground truth")
    if (modality["gps"].get("legacy_input_matches_label_measurement") is not False or
            modality["gps"].get("corrected_candidate_verified_in_memory") is not True or
            modality["gps"].get("review") != "synthetic_supervision_only"):
        raise ValueError("GPS legacy mismatch and synthetic-only policy must remain explicit")
    for name in ("imu", "lidar", "camera"):
        if (modality[name].get("legacy_label_source") !=
                "full_trajectory_fixed_fgo_proxy" or
                modality[name].get("review") != "descriptive_proxy_only"):
            raise ValueError(f"{name} FGO proxy cannot be independent supervision")
    if (modality["lidar"].get("same_raw_scan_factor_stream_verified") is not False or
            modality["camera"].get("same_raw_image_factor_stream_verified") is not False):
        raise ValueError("camera/lidar raw measurement consistency not established")

    method = policy.get("forward_method", {})
    if (method.get("offline_reference_in_network_inputs") is not False or
            method.get("use_labels_only_as_offline_training_target") is not True or
            method.get("permitted_online_features") !=
            "past_and_current_degraded_measurements_only" or
            method.get("measurement_feedback") !=
            "pre_update_residuals_only_with_causal_timing"):
        raise ValueError("online input constraints must be causal and oracle-free")

    gates = policy.get("release_gates")
    required = (
        "same_realization_multimodal_training_archive_materialized",
        "complete_feature_causality_verified",
        "independent_supervision_or_explicit_synthetic_truth_policy_approved",
        "evaluation_metrics_and_stopping_rules_frozen",
        "independent_test_protocol_frozen",
        "final_training_approved",
        "independent_evaluation_established",
    )
    if (not isinstance(gates, dict) or
            set(gates) != set(required) or
            any(gates[key] is not False for key in required)):
        raise ValueError("all Stage-3 release gates must remain CLOSED")
    return {
        "schema": policy["schema"],
        "review_status": "DEVELOPMENT_ONLY",
        "validated_source_entries": len(sources),
        "modalities": list(SENSORS),
        "heldout_drives_preserved": ["0016", "0020"],
        "offline_fgo_proxy_as_independent_truth": False,
        "gps_canonical_rebuild_verified_in_memory": True,
        "allowed_to_launch_final_training": False,
        "allowed_to_claim_independent_generalization": False,
    }


def read_policy(path: Path = MANIFEST) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def verify_file_hashes(sources: dict, *, root: Path = ROOT) -> dict:
    """Check immutable legacy sources against recorded SHA256 (read-only)."""
    root = Path(root).resolve()
    hashes: dict[str, str] = {}
    for name, record in sources.items():
        relative = Path(record["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"{name}: unsafe source path")
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"{name}: source escapes repository")
        if not path.is_file():
            raise FileNotFoundError(path)
        checksum = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                checksum.update(block)
        actual = checksum.hexdigest()
        if actual != record["sha256"]:
            raise ValueError(f"{name}: source SHA256 mismatch")
        hashes[name] = actual
    return hashes


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--verify-sources", action="store_true",
                        help="hash only the five known 0027 source artifacts")
    parser.add_argument("--require-final-training", action="store_true",
                        help="fail closed: Stage 3 is NOT approved to train")
    args = parser.parse_args(argv)
    try:
        policy = read_policy(args.manifest)
        summary = validate_policy(policy)
        if args.verify_sources:
            summary["source_sha256_verified"] = verify_file_hashes(
                policy["raw_sources"])
        print(json.dumps(summary, indent=2))
        if args.require_final_training:
            print("BLOCKED: V7 Stage-3 policy disallows final training.",
                  file=sys.stderr)
            return 2
        print("POLICY_CHECK_PASSED; FINAL_TRAINING_STILL_BLOCKED")
        return 0
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(f"POLICY_CHECK_FAILED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
