"""V9.1 training-side method lock, no new training or test release.

Produces nine frozen H0 mappings under the exact V8.5.3 fitting rule.
All three models receive the same calibration procedure. Existing V7
release policy is untouched. No held-out files or scores are opened.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from experiments.isj_v8.audit_forecast_dev import CASES
from experiments.isj_v8.run_c8_prediction_calibration import (
    verify_all, fit_prediction_bins, TRAIN_START, TRAIN_END, OUT as V853,
)
from experiments.isj_v8.run_c8_numeric_replay import preflight
from experiments.isj_v7.compare_dev_gps import read_candidate, read_run, sha256
from experiments.isj_v7.train_gps_dev import (
    load_dev_data, load_frozen_v7_gps_checkpoint, predict_frozen_gps_features_causally,
)
from experiments.isj_v7.run_gps_covariance_pilot import (
    DEGRADED, load_train_axis_noise, AXIS_STD_FLOOR_M,
)
from experiments.isj_v7.supervision_gate import read_policy, validate_policy

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/isj_v9/train_h0_method_lock_v1"
SEEDS = (43, 44, 45)
MODELS = ("gru96_r", "gru106_r", "mamba_r")
CAL_HASH = "52a7d2d091724c7c8f5bc7b96b9e0ca11a5bca2029137a5c25fa5d30bc9c428f"
H0 = "mamba_r__h0_current_observation_available"


def source_check():
    verify_all()
    if sha256(V853 / "report.json") != CAL_HASH:
        raise ValueError("V8.5.3 report changed")
    cal = json.loads((V853 / "report.json").read_text())
    for name in ("poses", "sigmas"):
        if sha256(V853 / (name + ".npz")) != cal[name+"_sha256"]:
            raise ValueError("V8.5.3 archive changed: "+name)
    if cal["cross_fitted"] is not False or cal["heldout_accessed"] is not False:
        raise ValueError("unexpected calibration provenance")
    return cal


def audit_inputs():
    old = source_check()
    sigma = preflight()
    policy = read_policy()
    validate_policy(policy)
    axis = load_train_axis_noise(policy)
    if not np.array_equal(axis["sigma_fixed_xyz"], sigma["axis_fixed"][0]):
        raise ValueError("fixed axis mapping drift")
    candidate = read_candidate()
    data, meta = load_dev_data()
    if meta["archive_sha256"] != candidate["archive_sha256"]:
        raise ValueError("candidate source drift")
    records = {}
    for name in MODELS:
        root, backend, params, family, expected = CASES[name]
        for seed in SEEDS:
            run = read_run(root, backend, seed, candidate)
            metrics = run["metrics"]
            cfg = metrics["config"]
            ckpt = Path(root)/backend/f"seed_{seed}"/"best.pt"
            if (not ckpt.is_file() or metrics["parameter_count"] != params
                    or metrics.get("run_family", "baseline") != family
                    or cfg["window"] != 64 or cfg["heads"] != [0,1,2]
                    or cfg["layers"] != 2):
                raise ValueError(f"missing or incompatible frozen run: {name}/{seed}")
            digest = sha256(ckpt)
            if seed == 43 and digest != expected:
                raise ValueError("seed43 checkpoint drift")
            records[f"{name}__seed{seed}"] = dict(
                model=name, seed=seed, parameters=params, checkpoint_sha256=digest,
                prediction_sha256=sha256(ckpt.parent/"val_predictions.npz"))
            print("CHECKPOINT_OK", name, seed, digest, flush=True)
    return old, data, meta, axis, records


def map_json(f):
    return {k: np.asarray(v).tolist() for k,v in f.items()}


def run():
    if OUT.exists():
        raise FileExistsError(f"refuse overwrite: {OUT}")
    old, data, meta, axis, records = audit_inputs()
    with np.load(DEGRADED/"degraded_sensor_data.npz", allow_pickle=False) as z:
        err = (np.asarray(z["degraded_gps_local"][TRAIN_START:TRAIN_END], float)
               - np.asarray(z["clean_gps_local"][TRAIN_START:TRAIN_END], float))
    fits = {}
    for tag, entry in records.items():
        model, med, iqr, provenance = load_frozen_v7_gps_checkpoint(
            entry["model"], seed=entry["seed"], device="cuda")
        if provenance["checkpoint_sha256"] != entry["checkpoint_sha256"]:
            raise ValueError("checkpoint changed after preflight")
        ids, pred, _ = predict_frozen_gps_features_causally(
            model, data["gps_features"][:TRAIN_END], med, iqr,
            window=64, clip=provenance["input_clip"], batch_size=64)
        if not np.array_equal(ids, np.arange(63,TRAIN_END)):
            raise ValueError("training inference alignment drift")
        fitted = fit_prediction_bins(pred[TRAIN_START-63:,0], err)
        fits[tag] = map_json(fitted)
        entry["checkpoint_provenance"] = provenance
        print("FIT", tag, json.dumps(fits[tag]), flush=True)
        del model

    # Reproduce the selected V8.5.3 map; do not silently replace it.
    tag = "mamba_r__seed43"
    previous = old["fits"][H0]
    current = fits[tag]
    for k in ("centers", "sigma_bins_xyz", "raw_second_moment"):
        if not np.allclose(current[k], previous[k], rtol=1e-5, atol=1e-7):
            print("MAMBA43_REPRODUCTION_MISMATCH", k, flush=True)
            raise ValueError("training calibration reproduction failed")
    if current["counts"] != previous["counts"]:
        raise ValueError("training bin count drift")
    fits[tag] = previous
    print("V8_5_3_MAMBA43_MAPPING_REPRODUCED", flush=True)
    source_check()

    # Method lock only: future test selection and release remain blocked.
    manifest = dict(
        schema="isj_v9_train_h0_method_lock_v1",
        status="TRAIN_METHOD_LOCKED_TEST_PROTOCOL_PENDING",
        development_only_checkpoint=True, independent_physical_gt=False,
        heldout_content_read=False, new_training=False, test_release=False,
        prior_test_drives=["2011_09_30_drive_0016_sync","2011_09_30_drive_0020_sync"],
        prior_development_or_validation_drives=[
            "2011_10_03_drive_0027_sync","2011_10_03_drive_0034_sync",
            "2011_10_03_drive_0042_sync","2011_10_03_drive_0047_sync"],
        calibration=dict(train_target_range=[TRAIN_START,TRAIN_END],
                         model_in_sample=True, cross_fitted=False,
                         bins=5, binning="stable_sort_equal_count",
                         monotonic="weighted_PAVA_second_moment",
                         interpolation="linear_sigma_endpoint_clamp",
                         sigma_floor_m=float(AXIS_STD_FLOOR_M), cap_information_ratio=4.0),
        inference=dict(feature_dim=21, window=64, head="h0", cold_start_frames=63,
                       cold_start="train_axis_fixed", no_future_features=True),
        graph=dict(gps="GPSFactor_world_xyz", relative="unchanged_IMU_LiDAR_Camera",
                   alignment="full_window_offline_yaw_from_degraded_gps_and_lidar",
                   anchor="C8_weak_rotation", jacobian="C8_numerical_reference",
                   optimizer="V8.4.4_warm_and_corrected_then_V8.4.5_two_pass_refinement",
                   stationarity_certified=False),
        seeds=list(SEEDS), models=list(MODELS), checkpoint_records=records,
        original_oracle_mapping=map_json(axis), calibrated_h0_mappings=fits,
        primary_comparison="Mamba vs GRU106 with identical prediction-conditioned fitting",
        secondary_comparisons=["GRU96 matched fitting","axis fixed",
                               "original oracle-indexed Cap4 vs prediction-conditioned Cap4"],
        future_baseline_to_implement="fixed axis plus predeclared Huber GPS robust kernel",
        future_prediction="exploratory DEV only; not primary localization claim",
        primary_metric="origin_aligned_oxTS_relative_3d_rmse_m",
        evaluation="no per-method rigid or scale fit; report per-drive and per-seed",
        test_release_blockers=["new sequence usage-history audit",
                               "test degradation plan and seeds freeze",
                               "arbitrary-length corrected backend and robust baseline tests",
                               "separate explicit synthetic-evaluation release protocol"],
        train_archive_sha256=meta["archive_sha256"], source_calibration_sha256=CAL_HASH,
    )
    OUT.mkdir(parents=True, exist_ok=False)
    p = OUT/"method_lock.json"
    p.write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n")
    print("METHOD_LOCK_SHA256", sha256(p))
    print("FROZEN_MAPPING_COUNT", len(fits))
    print("V9_1_TRAIN_METHOD_LOCK_COMPLETE NO_NEW_TRAINING NO_FGO NO_TEST_RELEASE")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--lock", action="store_true")
    args = p.parse_args()
    if args.preflight:
        audit_inputs()
        print("V9_1_PREFLIGHT_OK 9_CHECKPOINTS NO_INFERENCE NO_WRITES NO_TEST_RELEASE")
    else:
        run()


if __name__ == "__main__":
    main()
