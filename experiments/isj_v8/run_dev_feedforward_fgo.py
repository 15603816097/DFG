from __future__ import annotations

"""V8.3 exploratory DEV-only 0027 FGO: current GPS h0 vs CAUSAL forecasts.

Code/parameters/cases are locked before viewing V8.3 graph outcomes.
Reserved KITTI 0016/0020 must not be read, tuned, or rerun.
Run --preflight (no solver/writes) then explicitly --run-graph with token.
No feedback mechanism, no U-driven noise, no iSAM2, no independent GT.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.isj_v7.compare_dev_gps import read_candidate, read_run, sha256
from experiments.isj_v7.run_gps_covariance_pilot import (
    DEGRADED, BEGIN, END, cap_axis_information, axis_sigma_from_prediction,
    load_train_axis_noise, reference_diagnostics, subgraph_measurements,
)
from experiments.isj_v7.supervision_gate import read_policy, validate_policy
from experiments.isj_v7.cross_sequence_v7 import _solve_graph, _positions
from experiments.isj_v8.audit_forecast_dev import CASES, EXPECTED_ANCHORS, SEED
from experiments.isj_v8.causal_prior_dev import (
    LOCKED_START, preobservation_candidates,
)
from src.factor_graph.degraded_measurements import (
    load_degraded_four_sensor_measurements,
)
from src.factor_graph.four_sensor_graph import run_four_sensor_factor_graph

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "results/isj_v8/dev_0027_feedforward_fgo_v1"
SOURCE = DEGRADED / "degraded_sensor_data.npz"
MODEL_METHODS = (
    "h0_current_observation_available",
    "h0_tminus1_persistence",
    "h1_tminus1_causal_prior",
    "h2_tminus2_causal_prior",
    "h1h2_equal_causal_prior",
)
EXPECTED_MODEL_CASES = tuple(
    f"{model}__{name}" for model in ("gru96_r", "gru106_r", "mamba_r")
    for name in MODEL_METHODS
)
ALL_CASES = ("axis_fixed",) + EXPECTED_MODEL_CASES
EXPLICIT_TOKEN = "V8_DEV_ONLY_CAUSAL_FGO_0027"


def sigma_cases_from_frozen_predictions(
    anchors: np.ndarray, predictions: np.ndarray, axis_fit: dict,
) -> dict[str, np.ndarray]:
    """Map only frozen degraded-GPS predictions -> fixed-axis Cap4 sigmas.

    All cases use train-fixed XYZ for the *same first two frames* to
    maintain comparable full-680-frame graph factors. Each later causal
    prior reads predictions from t-1/t-2. h0(t) is a post-observation
    comparator, not part of any pre-observation forecast.
    """
    if (not np.array_equal(anchors, EXPECTED_ANCHORS)
            or BEGIN != 3180 or END != 3860):
        raise ValueError("not the approved 0027 development anchors")
    ids, candidates = preobservation_candidates(anchors, predictions)
    if not np.array_equal(ids, EXPECTED_ANCHORS[LOCKED_START:]):
        raise AssertionError("V8 causal time offset mismatch")
    fixed = np.asarray(axis_fit["sigma_fixed_xyz"], dtype=np.float64)
    if (fixed.shape != (3,) or not np.isfinite(fixed).all()
            or np.any(fixed <= 0)):
        raise ValueError("invalid train-only fixed axis sigmas")
    sources = {
        "h0_current_observation_available":
            candidates["post_observation_h0_t_diagnostic_only"],
        "h0_tminus1_persistence":
            candidates["persistence_h0_t_minus_1"],
        "h1_tminus1_causal_prior":
            candidates["forecast_h1_t_minus_1"],
        "h2_tminus2_causal_prior":
            candidates["forecast_h2_t_minus_2"],
        "h1h2_equal_causal_prior":
            candidates["forecast_h1h2_equal_mean"],
    }
    mapped = {}
    for name, tail in sources.items():
        p = np.full((END-BEGIN,), .5, dtype=np.float64)
        p[LOCKED_START:] = tail
        sigmas = axis_sigma_from_prediction(p, axis_fit)
        sigmas = cap_axis_information(sigmas, fixed)
        sigmas[:LOCKED_START] = fixed
        if (sigmas.shape != (680, 3)
                or not np.isfinite(sigmas).all()
                or np.any(sigmas <= 0)
                or np.any((fixed[None, :] / sigmas)**2 > 4.00000000001)
                or not np.array_equal(
                    sigmas[:LOCKED_START],
                    np.tile(fixed, (LOCKED_START, 1)))):
            raise ValueError(f"{name}: invalid V8 capped GPS covariance")
        mapped[name] = sigmas
    return mapped


def prepare_cases() -> tuple[dict, dict[str, np.ndarray]]:
    """Only 0027 train-fitted mapping + frozen DEV prediction archives.

    No FGO, no clean VAL reference, no novel model weights, no files
    written and no reserved-drive code paths used. Synthetic VAL targets
    are read for archive-integrity checks by frozen V7 read_run(), not
    passed to covariance construction or optimization.
    """
    policy = read_policy()
    validate_policy(policy)
    if (not SOURCE.is_file()
            or sha256(SOURCE) !=
            policy["raw_sources"]["degraded_sensor"]["sha256"]):
        raise ValueError("V7 0027 degraded sensor archive SHA mismatch")
    candidate = read_candidate()
    anchors = candidate["anchors"]
    if not np.array_equal(anchors, EXPECTED_ANCHORS):
        raise ValueError("V8 source is not exact 0027 VAL")
    axis = load_train_axis_noise(policy)
    fixed = np.asarray(axis["sigma_fixed_xyz"], np.float64)
    all_sigmas: dict[str, np.ndarray] = {
        "axis_fixed": np.tile(fixed, (680, 1)),
    }
    provenance = {
        "schema": "v8_0027_dev_feedforward_fgo_v1",
        "status": "LOCKED_DEV_EXPLORATORY_NOT_INDEPENDENT",
        "source": "2011_10_03_drive_0027_sync",
        "development_only": True,
        "independent_position_ground_truth": False,
        "heldout_0016_0020_accessed": False,
        "forecast_head_future_observation_leakage": False,
        "predictive_feedback_used": False,
        "imu_lidar_camera_covariances_fixed": True,
        "batch_Levenberg_Marquardt_not_iSAM2": True,
        "all_cases_use_same_frame_range": [BEGIN, END],
        "all_cases_same_startup_fixed_frames": LOCKED_START,
        "predicted_uncertainty_U_used": False,
        "cap4_fixed_relative_precision": 4.0,
        "checkpoint_seed": SEED,
        "prediction_reference_used_for_weight_selection": False,
        "comparisons_frozen_before_dev_FGO": list(ALL_CASES),
        "gps_degraded_sha256": sha256(SOURCE),
        "gps_candidate_sha256": candidate["archive_sha256"],
        "lidar_factors_sha256":
            sha256(DEGRADED / "degraded_lidar_factor_data.npz"),
        "camera_factors_sha256":
            sha256(DEGRADED / "degraded_camera_factor_data.npz"),
        "train_fixed_axis_sigma_xyz": fixed.tolist(),
        "methods": {},
    }
    for name, (root, backend, params, family, expected_sha) in CASES.items():
        run = read_run(root, backend, SEED, candidate)
        meta = run["metrics"]
        if (meta.get("parameter_count") != params
                or meta.get("run_family", "baseline") != family
                or meta.get("config", {}).get("heads") != [0, 1, 2]
                or meta.get("config", {}).get("window") != 64):
            raise ValueError(f"{name}: unexpected V7 frozen model")
        ckpt = root / backend / f"seed_{SEED}" / "best.pt"
        if not ckpt.is_file() or sha256(ckpt) != expected_sha:
            raise ValueError(f"{name}: frozen checkpoint SHA mismatch")
        arrays = run["arrays"]
        if not np.array_equal(arrays["anchors"], anchors):
            raise ValueError("saved predictions not exactly val-aligned")
        pred = np.asarray(arrays["reliability"], np.float64)
        mapped = sigma_cases_from_frozen_predictions(
            anchors, pred, axis
        )
        for method, sigma in mapped.items():
            all_sigmas[f"{name}__{method}"] = sigma
        provenance["methods"][name] = {
            "checkpoint_sha256": expected_sha,
            "saved_predictions_sha256": sha256(
                ckpt.parent / "val_predictions.npz"
            ),
            "head_schedule": {
                "h0_current_observation_available": "h0(t), uses GPS(t)",
                "h0_tminus1_persistence": "h0(t-1)",
                "h1_tminus1_causal_prior": "h1(t-1)",
                "h2_tminus2_causal_prior": "h2(t-2)",
                "h1h2_equal_causal_prior":
                    "0.5*h1(t-1)+0.5*h2(t-2)",
            },
        }
    if tuple(all_sigmas) != ALL_CASES:
        raise AssertionError("V8 locked dev graph case ordering changed")
    provenance["case_sigmas"] = {
        case: {
            "min_m": float(sigma.min()),
            "mean_xyz_m": sigma.mean(axis=0).tolist(),
            "max_m": float(sigma.max()),
        }
        for case, sigma in all_sigmas.items()
    }
    return provenance, all_sigmas


def evaluate_development_graph() -> dict:
    """Run all 16 common-initialization dev graphs BEFORE reading reference.

    No writes on any failure; results are written once *after* solving
    all cases and computing retrospective 0027 VAL diagnostics.
    """
    if DEST.exists():
        raise FileExistsError(
            f"V8.3 result directory already exists: {DEST}; no retry/overwrite")
    report, sigmas = prepare_cases()
    measurement = load_degraded_four_sensor_measurements(DEGRADED)
    sub = subgraph_measurements(measurement, begin=BEGIN, end=END)
    import gtsam
    baseline_initial = None
    expected_factor_counts = None
    solved = {}
    for case, sigma in sigmas.items():
        graph, initial, counts = run_four_sensor_factor_graph(
            sub, DEST / "NO_GRAPH_FILE_WRITTEN", mode="fixed",
            gps_sigmas_override=sigma, build_graph_only=True,
        )
        initial_poses = _positions(gtsam, initial, END-BEGIN)
        if baseline_initial is None:
            baseline_initial = initial_poses.copy()
            expected_factor_counts = dict(counts)
        elif (not np.array_equal(initial_poses, baseline_initial)
              or counts != expected_factor_counts):
            raise RuntimeError("V8 DEV graph initial state/factors differ")
        result = _solve_graph(graph, initial, END-BEGIN)
        if (not result["stabilized"] or
                result["poses"].shape != (680, 4, 4)):
            raise RuntimeError(
                f"{case}: failed locked LM convergence; do not select cases")
        solved[case] = {
            "solve": {
                k:v for k,v in result.items() if k != "poses"
            },
            "position": result["poses"][:, :3, 3].copy(),
        }
        print(
            "V8_DEV_GRAPH_SOLVED", case,
            "passes=", len(result["passes"]),
            "stable=", result["stabilized"], flush=True,
        )
    # Clean reference is ONLY opened after all 16 graphs converge.
    with np.load(SOURCE, allow_pickle=False) as z:
        q = np.asarray(z["clean_gps_local"][BEGIN:END], np.float64)
    if q.shape != (680, 3) or not np.isfinite(q).all():
        raise ValueError("invalid retrospective DEV reference")
    report["factors_per_graph"] = expected_factor_counts
    report["results"] = {}
    for case, obj in solved.items():
        report["results"][case] = {
            "solve": obj["solve"],
            "evaluation": reference_diagnostics(obj["position"], q),
        }
    report["evaluation_uses_correlated_oxTS_reference"] = True
    report["all_cases_stable"] = True
    report["status"] = "COMPLETE_EXPLORATORY_0027_DEV_ONLY"
    # Append-only: one output directory created at the very end.
    DEST.mkdir(parents=True, exist_ok=False)
    target = DEST / "v8_dev_feedforward_fgo_results.json"
    target.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        "V8_DEV_FGO_COMPLETE; 16_CASES; DEV_ONLY; NOT_INDEPENDENT; "
        "NO_HELDOUT", flush=True,
    )
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    actions = p.add_mutually_exclusive_group(required=True)
    actions.add_argument(
        "--preflight", action="store_true",
        help="no solver, no output writes; check 0027 dev inputs/sigma")
    actions.add_argument(
        "--run-graph", action="store_true",
        help="exploratory 0027 16-case LM graph; write new result once")
    p.add_argument("--token", default=None)
    args = p.parse_args(argv)
    if args.preflight:
        if args.token is not None:
            p.error("--preflight does not use a release token")
        report, _ = prepare_cases()
        print(json.dumps(report, indent=2, ensure_ascii=False))
        print(
            "V8_DEV_FGO_PREFLIGHT_OK; 16_CASES; "
            "NO_SOLVER; NO_WRITES; NO_HELDOUT", flush=True,
        )
        return
    if args.token != EXPLICIT_TOKEN:
        p.error("explicit V8 development graph token required")
    evaluate_development_graph()


if __name__ == "__main__":
    main()
