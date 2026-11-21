from __future__ import annotations

"""V8.1 DEV-ONLY forecast skill gate; no heldout data, training or FGO.

Uses the original V7 0027 frozen validation predictions (seed 43) and
synthetic reliability targets. First verify whether h1/h2 predict future
reliability better than *causal current-head persistence* at time t.
This is an exploratory 0027 diagnostic, NEVER independent evaluation.

Default CLI is strictly read-only and makes no experiment artifacts.
"""

import argparse
import json

import numpy as np

from experiments.isj_v7.compare_dev_gps import (
    read_candidate, read_run, sha256,
)
from experiments.isj_v7.train_gps_dev import (
    DEFAULT_OUTPUT, CAPACITY_OUTPUT,
)

CASES = {
    "gru96_r": (DEFAULT_OUTPUT, "gru", 114630, "baseline",
                "8b0fb710124bb2c0ca519bd6d8c9e399ebcb6a66a4cf04e7f59b15184d93931d"),
    "gru106_r": (CAPACITY_OUTPUT, "gru", 139290, "capacity_gru_h106",
                 "73de422afb6753cee11cabc3ea9098bd18ff0711d11adbc8e959844563080b7c"),
    "mamba_r": (DEFAULT_OUTPUT, "mamba", 139590, "baseline",
                "f548bdbe229c709bd6514817714815c9507f6673b65c3519169e15849c812dcc"),
}
SEED = 43
EXPECTED_ANCHORS = np.arange(3180, 3860, dtype=np.int64)
CHANGE_THRESHOLD = 0.15


def validate_development_alignment(anchors: np.ndarray,
                                   targets: np.ndarray) -> None:
    """The val future targets must match succeeding h0 labels, exactly."""
    a = np.asarray(anchors)
    y = np.asarray(targets)
    if (a.shape != (680,) or not np.array_equal(a, EXPECTED_ANCHORS)
            or y.shape != (680, 3) or not np.isfinite(y).all()
            or np.any((y < 0) | (y > 1))):
        raise ValueError("invalid V8 development anchors/target shapes")
    if (not np.array_equal(y[:-1, 1], y[1:, 0])
            or not np.array_equal(y[:-2, 2], y[2:, 0])):
        raise ValueError("V8 future target shift or horizon leakage")


def _errors(prediction: np.ndarray, target: np.ndarray) -> dict:
    err = np.asarray(prediction, np.float64) - np.asarray(target, np.float64)
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
    }


def score_development_forecasts(
        anchors: np.ndarray, targets: np.ndarray,
        reliability: np.ndarray, evidence: np.ndarray) -> dict:
    """Quantify h1/h2 skill relative to predicting with h0(t) unchanged.

    The persistence baseline is a usable causal *MODEL* forecast,
    not the future label and not an online available oracle.
    The change-point subset is selected with retrospective synthetic
    targets on previously explored VAL; descriptive only.
    """
    validate_development_alignment(anchors, targets)
    p = np.asarray(reliability, dtype=np.float64)
    u = np.asarray(evidence, dtype=np.float64)
    if (p.shape != (680, 3) or u.shape != (680, 3)
            or not np.isfinite(p).all() or not np.isfinite(u).all()
            or np.any((p <= 0) | (p >= 1))
            or np.any((u <= 0) | (u > 1))):
        raise ValueError("invalid V8 saved reliability/evidence arrays")
    scores = {
        "h0": {
            "model_vs_synthetic_target": _errors(p[:, 0], targets[:, 0]),
            "mean_evidence_proxy": float(np.mean(u[:, 0])),
        },
    }
    for horizon in (1, 2):
        y = targets[:, horizon]
        forecast = _errors(p[:, horizon], y)
        current_persist = _errors(p[:, 0], y)
        selector = np.abs(y - targets[:, 0]) >= CHANGE_THRESHOLD
        count = int(np.sum(selector))
        if count:
            forecast_change = _errors(p[selector, horizon], y[selector])
            persist_change = _errors(p[selector, 0], y[selector])
            changes_correct = float(np.mean(
                np.sign(p[selector, horizon] - p[selector, 0]) ==
                np.sign(y[selector] - targets[selector, 0])
            ))
        else:
            forecast_change = None
            persist_change = None
            changes_correct = None
        delta = current_persist["rmse"] - forecast["rmse"]
        scores[f"h{horizon}"] = {
            "anchor_count": 680,
            "target_frame_offset": horizon,
            "forecast": forecast,
            "causal_h0_persistence": current_persist,
            "rmse_change_vs_persistence_magnitude": float(delta),
            "forecast_beats_persistence_rmse": bool(delta > 0),
            "mean_evidence_proxy": float(np.mean(u[:, horizon])),
            "oracle_defined_change_subset": {
                "threshold_abs_reliability": CHANGE_THRESHOLD,
                "count": count,
                "fraction": float(count / 680),
                "forecast": forecast_change,
                "causal_h0_persistence": persist_change,
                "predicted_change_sign_accuracy": changes_correct,
                "use": "retrospective_development_only_not_online_detection",
            },
        }
    return scores


def run_read_only() -> dict:
    """Only 0027 frozen DEV candidate + seed43 saved val predictions."""
    candidate = read_candidate()
    validate_development_alignment(
        candidate["anchors"], candidate["targets"]
    )
    result = {
        "schema": "isj_v8_0027_dev_horizon_readonly_v1",
        "status": "DEV_DIAGNOSTIC_ONLY",
        "input_sequence": "2011_10_03_drive_0027_sync",
        "training_anchor_range": [63, 3178],
        "validation_anchor_range": [3180, 3860],
        "independent_generalization": False,
        "heldout_sensor_content_read": False,
        "model_retrained": False,
        "factor_graph_evaluated": False,
        "forecast_heads_not_yet_deployed": [1, 2],
        "comparison": "h1(t)/h2(t) versus frozen predicted h0(t) persistence",
        "oracle_change_subset_is_retrospective": True,
        "candidate_archive_sha256": candidate["archive_sha256"],
        "models": {},
    }
    for name, (root, backend, parameters, family, expected_sha) in CASES.items():
        loaded = read_run(root, backend, SEED, candidate)
        metrics = loaded["metrics"]
        config = metrics.get("config", {})
        run_family = metrics.get("run_family", "baseline")
        if (metrics.get("parameter_count") != parameters
                or run_family != family
                or config.get("window") != 64
                or config.get("heads") != [0, 1, 2]):
            raise ValueError(f"{name}: frozen V7 run configuration mismatch")
        folder = root / backend / f"seed_{SEED}"
        checkpoint = folder / "best.pt"
        if not checkpoint.is_file() or sha256(checkpoint) != expected_sha:
            raise ValueError(f"{name}: frozen checkpoint SHA256 mismatch")
        arrays = loaded["arrays"]
        score = score_development_forecasts(
            arrays["anchors"], arrays["synthetic_targets"],
            arrays["reliability"], arrays["evidence_uncertainty"],
        )
        result["models"][name] = {
            "backend": backend,
            "trainable_parameters": parameters,
            "checkpoint_sha256": expected_sha,
            "validation_predictions_sha256": sha256(
                folder / "val_predictions.npz"
            ),
            "scores": score,
        }
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preflight", action="store_true", required=True,
        help="development-only forecast diagnostic; no writes or FGO",
    )
    args = parser.parse_args(argv)
    if not args.preflight:
        parser.error("only --preflight is allowed")
    report = run_read_only()
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    print(
        "V8_DEV_HORIZON_PREFLIGHT_OK; 0027_DEV_ONLY; "
        "NO_HELDOUT; NO_WRITES; NO_GRAPH",
        flush=True,
    )


if __name__ == "__main__":
    main()
