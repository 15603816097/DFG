from __future__ import annotations

import copy
import hashlib
import json

import pytest

from experiments.isj_v7.supervision_gate import (
    MANIFEST,
    main,
    read_policy,
    validate_policy,
    verify_file_hashes,
)


def test_stage3_manifest_is_strictly_development_only():
    p = read_policy(MANIFEST)
    status = validate_policy(p)
    assert status["review_status"] == "DEVELOPMENT_ONLY"
    assert status["modalities"] == ["gps", "imu", "lidar", "camera"]
    assert status["allowed_to_launch_final_training"] is False
    assert status["allowed_to_claim_independent_generalization"] is False
    assert status["heldout_drives_preserved"] == ["0016", "0020"]


def test_claiming_fgo_proxy_as_independent_truth_is_blocked():
    p = read_policy()
    p["modalities"]["camera"]["independent_ground_truth"] = True
    with pytest.raises(ValueError, match="independent ground truth"):
        validate_policy(p)
    p = read_policy()
    p["modalities"]["imu"]["legacy_label_source"] = "independent_gt"
    with pytest.raises(ValueError, match="FGO proxy"):
        validate_policy(p)


def test_incorrect_horizon_or_unreviewed_split_fails_closed():
    p = read_policy()
    p["temporal_protocol"]["head_offsets"] = [0, 3, 2]
    with pytest.raises(ValueError, match="time split"):
        validate_policy(p)


def test_gps_legacy_provenance_cannot_be_overwritten_in_manifest():
    p = read_policy()
    p["modalities"]["gps"]["legacy_input_matches_label_measurement"] = True
    with pytest.raises(ValueError, match="GPS legacy mismatch"):
        validate_policy(p)


def test_final_release_flag_cannot_be_marked_true():
    p = read_policy()
    p["release_gates"]["final_training_approved"] = True
    with pytest.raises(ValueError, match="CLOSED"):
        validate_policy(p)
    p = read_policy()
    p["release_gates"]["independent_evaluation_established"] = True
    with pytest.raises(ValueError, match="CLOSED"):
        validate_policy(p)


def test_heldout_paths_and_0027_test_mislabel_are_forbidden():
    p = read_policy()
    p["raw_sources"]["v3"]["path"] = "dataset/kitti/0020/data.npz"
    with pytest.raises(ValueError, match="reserved source"):
        validate_policy(p)
    p = read_policy()
    p["legacy_0027_internal_test_is_independent"] = True
    with pytest.raises(ValueError, match="not independent"):
        validate_policy(p)


def test_model_inputs_must_not_include_offline_reference():
    p = read_policy()
    p["forward_method"]["offline_reference_in_network_inputs"] = True
    with pytest.raises(ValueError, match="causal"):
        validate_policy(p)


def test_file_fingerprint_guard_detects_edits_without_writes(tmp_path):
    raw = tmp_path / "example.bin"
    raw.write_bytes(b"immutable data")
    record = {"v": {
        "path": raw.name,
        "sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
    }}
    assert verify_file_hashes(record, root=tmp_path)["v"] == record["v"]["sha256"]
    raw.write_bytes(b"tampered data")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        verify_file_hashes(record, root=tmp_path)


def test_file_fingerprint_guard_rejects_parent_escape(tmp_path):
    with pytest.raises(ValueError, match="unsafe"):
        verify_file_hashes({
            "p": {"path": "../elsewhere", "sha256": "0" * 64},
        }, root=tmp_path)


def test_require_final_training_cli_returns_nonzero():
    assert main(["--manifest", str(MANIFEST)]) == 0
    assert main(["--manifest", str(MANIFEST),
                 "--require-final-training"]) == 2
