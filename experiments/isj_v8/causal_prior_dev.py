from __future__ import annotations

"""V8.2 GPS reliability pre-observation causal-prior audit (0027 DEV ONLY).

No model retraining, heldout sequence reads, factor graph, or output writes.
Compare frozen forecast h1(t-1), h2(t-2) to strictly causal persistence.
Inputs and synthetic targets originate only from the already-explored
0027 development-validation archive. NEVER claim independent test skill.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.isj_v7.compare_dev_gps import read_candidate, read_run, sha256
from experiments.isj_v8.audit_forecast_dev import (
    CASES, EXPECTED_ANCHORS, SEED, validate_development_alignment,
)

N_VAL = len(EXPECTED_ANCHORS)
LOCKED_START = 2
TRANSITION_THRESHOLD = 0.15


def preobservation_candidates(
    anchors: np.ndarray,
    reliability: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Build predictions for GPS(t) from information strictly before t.

    The current h0(t) is included as a labeled POST-OBSERVATION
    diagnostic comparator only, and must never enter a prior.
    Other candidates use t-1 and/or t-2 frozen sequence predictions.
    """
    a = np.asarray(anchors)
    p = np.asarray(reliability, dtype=np.float64)
    if (a.shape != (N_VAL,)
            or not np.array_equal(a, EXPECTED_ANCHORS)
            or p.shape != (N_VAL, 3)
            or not np.isfinite(p).all()
            or np.any((p <= 0) | (p >= 1))):
        raise ValueError("invalid V8.2 frozen validation stream")
    rows = np.arange(LOCKED_START, N_VAL, dtype=np.int64)
    if not (np.array_equal(a[rows - 1], a[rows] - 1)
            and np.array_equal(a[rows - 2], a[rows] - 2)):
        raise ValueError("noncontiguous causal prediction anchors")
    source = {
        "persistence_h0_t_minus_1": p[rows - 1, 0].copy(),
        "persistence_h0_t_minus_2": p[rows - 2, 0].copy(),
        "forecast_h1_t_minus_1": p[rows - 1, 1].copy(),
        "forecast_h2_t_minus_2": p[rows - 2, 2].copy(),
        # Locked zero-parameter blend: do NOT select a weight on 0016/0020.
        "forecast_h1h2_equal_mean": (
            .5 * (p[rows - 1, 1] + p[rows - 2, 2])
        ).copy(),
        "post_observation_h0_t_diagnostic_only": p[rows, 0].copy(),
    }
    return a[rows].copy(), source


def _score(pred: np.ndarray, truth: np.ndarray) -> dict:
    e = np.asarray(pred, dtype=np.float64) - np.asarray(
        truth, dtype=np.float64)
    return {
        "mae": float(np.mean(np.abs(e))),
        "rmse": float(np.sqrt(np.mean(e * e))),
    }


def score_causal_priors(
    anchors: np.ndarray, targets: np.ndarray,
    reliability: np.ndarray, evidence: np.ndarray,
) -> dict:
    """Descriptive DEV skill: never select a hyperparameter from TEST."""
    validate_development_alignment(anchors, targets)
    u = np.asarray(evidence, dtype=np.float64)
    if (u.shape != (N_VAL, 3) or not np.isfinite(u).all()
            or np.any((u <= 0) | (u > 1))):
        raise ValueError("invalid frozen evidential proxy")
    ids, candidates = preobservation_candidates(
        anchors, reliability,
    )
    y = np.asarray(targets, dtype=np.float64)
    truth = y[LOCKED_START:, 0]
    metric = {name: _score(pred, truth)
              for name, pred in candidates.items()}
    changes = np.abs(
        y[LOCKED_START:, 0] - y[LOCKED_START - 1:-1, 0],
    ) >= TRANSITION_THRESHOLD
    counts = int(changes.sum())
    change_scores = (
        {name: _score(pred[changes], truth[changes])
         for name, pred in candidates.items()}
        if counts else None
    )
    base_1 = metric["persistence_h0_t_minus_1"]["rmse"]
    base_2 = metric["persistence_h0_t_minus_2"]["rmse"]
    comparisons = {}
    for forecast, baseline in (
        ("forecast_h1_t_minus_1", base_1),
        ("forecast_h2_t_minus_2", base_2),
        ("forecast_h1h2_equal_mean", base_1),
    ):
        v = float(baseline - metric[forecast]["rmse"])
        comparisons[forecast] = {
            "rmse_gain_vs_past_h0_persistence": v,
            "gain_positive": bool(v > 0),
        }
    return {
        "schema": "v8_0027_dev_preobservation_reliability_prior_v1",
        "source": "0027_development_val_only",
        "evaluated_anchor_start_inclusive": int(ids[0]),
        "evaluated_anchor_end_exclusive": int(ids[-1] + 1),
        "evaluated_anchor_count": int(len(ids)),
        "target": "synthetic_oracle_current_reliability_at_t",
        "prior_only_uses_observations_through": "t_minus_1",
        "two_head_blend_weights": [0.5, 0.5],
        "decision_threshold_tuned": False,
        "performance": metric,
        "rmse_improvement_comparisons": comparisons,
        "retrospective_target_selected_transitions": {
            "not_online_or_confirmatory": True,
            "threshold_abs_target_change": TRANSITION_THRESHOLD,
            "count": counts,
            "performance": change_scores,
        },
        "evidence_proxy_mean_h1_at_t_minus_1": float(
            np.mean(u[LOCKED_START - 1:-1, 1])
        ),
        "evidence_proxy_mean_h2_at_t_minus_2": float(
            np.mean(u[:-LOCKED_START, 2])
        ),
    }


def run_read_only() -> dict:
    candidate = read_candidate()
    validate_development_alignment(
        candidate["anchors"], candidate["targets"])
    report = {
        "schema": "v8_0027_dev_causal_prior_preflight_v1",
        "heldout_content_read": False,
        "no_training": True,
        "no_graph": True,
        "no_output_written": True,
        "forecast_released_for_heldout": False,
        "validation_anchor_count_before_prefix": N_VAL,
        "provenance_candidate_sha256": candidate["archive_sha256"],
        "models": {},
    }
    for name, (root, backend, params, family, expected_sha) in CASES.items():
        run = read_run(root, backend, SEED, candidate)
        meta = run["metrics"]
        if (meta.get("parameter_count") != params
                or meta.get("run_family", "baseline") != family
                or meta.get("config", {}).get("heads") != [0, 1, 2]
                or meta.get("config", {}).get("window") != 64):
            raise ValueError(f"{name}: frozen checkpoint config mismatch")
        ckpt = Path(root) / backend / f"seed_{SEED}" / "best.pt"
        if not ckpt.is_file() or sha256(ckpt) != expected_sha:
            raise ValueError(f"{name}: frozen checkpoint SHA256 mismatch")
        arrays = run["arrays"]
        report["models"][name] = {
            "checkpoint_sha256": expected_sha,
            "validation_predictions_sha256": sha256(
                ckpt.parent / "val_predictions.npz"),
            "prior_scores": score_causal_priors(
                arrays["anchors"], arrays["synthetic_targets"],
                arrays["reliability"], arrays["evidence_uncertainty"],
            ),
        }
    return report


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--preflight", action="store_true", required=True,
        help="strictly read-only 0027 DEV temporal-prior analysis")
    p.parse_args(argv)
    print(json.dumps(run_read_only(), indent=2), flush=True)
    print(
        "V8_CAUSAL_PRIOR_DEV_PREFLIGHT_OK; "
        "NO_HELDOUT; NO_TRAINING; NO_FGO; NO_WRITES",
        flush=True,
    )


if __name__ == "__main__":
    main()
