from __future__ import annotations

"""V7 0027 GPS *development-only* paired GRU/Mamba validation analysis.

Never loads held-out 0016/0020, internal 0027 test targets, the clean GPS
reference, or any model checkpoint. The sole source of comparison is the
immutable GPS-only exported train/validation candidate plus the recorded
validation predictions of both networks.

This does NOT grant final training or independent-generalization approval.
By default read-only. --write-report writes an append-only report directory.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import beta as beta_dist
from scipy.stats import spearmanr

from experiments.isj_v7.build_dev_gps_candidate import DEFAULT_DEST
from experiments.isj_v7.train_gps_dev import DEFAULT_OUTPUT
from experiments.isj_v7.supervision_gate import read_policy, validate_policy

COMPARISON_ROOT = DEFAULT_OUTPUT.parent / "dev_gps_comparison_v1"
REQUIRED_KEYS = ("anchors", "reliability", "evidence_uncertainty",
                 "synthetic_targets")
HORIZONS = ("h0", "h1", "h2")
CONFIG_KEYS = ("seed", "hidden", "layers", "dropout", "lr",
               "weight_decay", "batch_size", "epochs", "patience",
               "min_delta", "input_clip", "window", "heads")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_candidate() -> dict:
    policy = read_policy()
    validate_policy(policy)
    p = DEFAULT_DEST / "gps_train_val.npz"
    provenance = DEFAULT_DEST / "provenance.json"
    if not (p.is_file() and provenance.is_file()):
        raise FileNotFoundError("original frozen V7 GPS train/val candidate missing")
    metadata = json.loads(provenance.read_text(encoding="utf-8"))
    if (metadata.get("schema") != "isj_v7_dev_gps_same_source_v1"
            or metadata.get("development_only") is not True
            or metadata.get("approved_for_final_training") is not False
            or metadata.get("source_drive") != "2011_10_03_drive_0027_sync"
            or metadata.get("omitted_internal_test_range") != [3862, 4544]):
        raise ValueError("candidate is not frozen GPS-only 0027 development data")
    for k in ("v3", "degraded_sensor"):
        if (metadata["source_archive_sha256"][k] !=
                policy["raw_sources"][k]["sha256"]):
            raise ValueError("source provenance differs from frozen policy")
    with np.load(p, allow_pickle=False) as data:
        if set(data.files) != {
            "gps_features", "train_anchors", "train_targets",
            "val_anchors", "val_targets", "train_median", "train_iqr",
        }:
            raise ValueError("unexpected feature/label fields in candidate")
        anchors = np.asarray(data["val_anchors"], dtype=np.int64)
        targets = np.asarray(data["val_targets"], dtype=np.float32)
    if not np.array_equal(anchors, np.arange(3180, 3860)):
        raise ValueError("validation anchors are not the frozen 0027 split")
    if targets.shape != (680, 3) or not np.isfinite(targets).all():
        raise ValueError("invalid frozen validation targets")
    return {
        "anchors": anchors, "targets": targets,
        "archive_sha256": sha256(p),
        "manifest_sha256": sha256(provenance),
    }


def read_run(root: Path, backend: str, seed: int, candidate: dict) -> dict:
    root = Path(root)
    base = root / backend / f"seed_{seed}"
    metric_path = base / "metrics.json"
    pred_path = base / "val_predictions.npz"
    if not metric_path.is_file() or not pred_path.is_file():
        raise FileNotFoundError(
            f"incomplete {backend} seed={seed}: {base}")
    metrics = json.loads(metric_path.read_text(encoding="utf-8"))
    if (metrics.get("schema") != "isj_v7_dev_gps_trainval_baseline_v1"
            or metrics.get("backend") != backend
            or metrics.get("development_only") is not True
            or metrics.get("approved_for_final_training") is not False
            or metrics.get("independent_test_run") is not False
            or metrics.get("validation_is_from_0027_development") is not True
            or metrics.get("config", {}).get("seed") != seed
            or metrics.get("dataset_sha256") != candidate["archive_sha256"]
            or metrics.get("dataset_manifest_sha256") !=
            candidate["manifest_sha256"]):
        raise ValueError(f"{backend}: unverifiable training provenance")
    with np.load(pred_path, allow_pickle=False) as stored:
        if set(stored.files) != set(REQUIRED_KEYS):
            raise ValueError(f"{backend}: unexpected prediction archive fields")
        arrays = {name: np.asarray(stored[name]).copy()
                  for name in REQUIRED_KEYS}
    if not np.array_equal(arrays["anchors"], candidate["anchors"]):
        raise ValueError(f"{backend}: mismatched validation anchors")
    if not np.array_equal(arrays["synthetic_targets"],
                          candidate["targets"]):
        raise ValueError(f"{backend}: stored validation labels mismatch")
    for key in ("reliability", "evidence_uncertainty",
                "synthetic_targets"):
        arr = arrays[key]
        if arr.shape != (680, 3) or not np.isfinite(arr).all():
            raise ValueError(f"{backend}: invalid {key}")
    if (np.any((arrays["reliability"] <= 0) |
               (arrays["reliability"] >= 1))
            or np.any((arrays["evidence_uncertainty"] <= 0) |
                      (arrays["evidence_uncertainty"] > 1))):
        raise ValueError(f"{backend}: invalid reliability/evidence domain")
    for j, h in enumerate(HORIZONS):
        prediction = arrays["reliability"][:, j]
        truth = arrays["synthetic_targets"][:, j]
        mae = float(np.mean(np.abs(prediction - truth)))
        rmse = float(np.sqrt(np.mean(np.square(prediction - truth))))
        previous = metrics.get("val_heads", {}).get(h, {})
        if (not math.isclose(mae, previous.get("MAE", float("nan")),
                             rel_tol=0.0, abs_tol=1e-6)
                or not math.isclose(rmse,
                                    previous.get("RMSE", float("nan")),
                                    rel_tol=0.0, abs_tol=1e-6)):
            raise ValueError(f"{backend}: saved metrics do not match predictions")
    return {"metrics": metrics, "arrays": arrays}


def uncertainty_diagnostics(
    predictions: np.ndarray, uncertainties: np.ndarray,
    truth: np.ndarray,
) -> dict:
    """Check evidence-error ranking and uncalibrated Beta 80% coverage.

    Coverage is descriptive only: the Beta head is optimized against
    synthetic reliability targets, not an independently calibrated
    probabilistic physical measurement model.
    """
    if (predictions.shape != truth.shape
            or uncertainties.shape != truth.shape
            or predictions.ndim != 2
            or not all(np.isfinite(a).all() for a in
                       (predictions, uncertainties, truth))):
        raise ValueError("uncertainty diagnostic arrays mismatch")
    if (np.any(uncertainties <= 0)
            or np.any(predictions <= 0)
            or np.any(predictions >= 1)):
        raise ValueError("uncertainty domain invalid")
    concentration = 2.0 / uncertainties.astype(np.float64)
    alpha = predictions.astype(np.float64) * concentration
    beta = (1.0 - predictions.astype(np.float64)) * concentration
    lower = beta_dist.ppf(0.10, alpha, beta)
    upper = beta_dist.ppf(0.90, alpha, beta)
    if not (np.isfinite(lower).all() and np.isfinite(upper).all()):
        raise ValueError("invalid reconstructed Beta credible intervals")
    out = {}
    for j, name in enumerate(HORIZONS):
        err = np.abs(predictions[:, j] - truth[:, j])
        u = uncertainties[:, j]
        if np.std(err) < 1e-12 or np.std(u) < 1e-12:
            rank_corr = None
        else:
            rho = float(spearmanr(u, err).statistic)
            rank_corr = rho if np.isfinite(rho) else None
        out[name] = {
            "evidence_mean": float(np.mean(u)),
            "evidence_std": float(np.std(u)),
            "spearman_evidence_vs_absolute_error": rank_corr,
            "assumed_beta_80pct_interval_coverage": float(
                np.mean((truth[:, j] >= lower[:, j])
                        & (truth[:, j] <= upper[:, j]))),
            "assumed_beta_80pct_interval_mean_width": float(
                np.mean(upper[:, j] - lower[:, j])),
        }
    return out


def block_bootstrap_delta(
    gru_abs_error: np.ndarray, mamba_abs_error: np.ndarray,
    *, block: int = 32, repeats: int = 1000, seed: int = 20261120,
) -> dict:
    """Exploratory paired contiguous-block bootstrap of MAE deltas.

    Difference convention: Mamba minus GRU. Negative favors Mamba.
    This is *not* an independent-drive confidence interval: a single
    validation drive and overlapping windows violate simple iid tests.
    """
    g = np.asarray(gru_abs_error, dtype=np.float64)
    m = np.asarray(mamba_abs_error, dtype=np.float64)
    if (g.shape != m.shape or g.ndim != 1
            or len(g) < block or block < 2 or repeats < 1
            or not np.isfinite(g).all() or not np.isfinite(m).all()):
        raise ValueError("invalid paired temporal bootstrap inputs")
    diff = m - g
    rng = np.random.default_rng(seed)
    n = len(diff)
    nblocks = math.ceil(n / block)
    starts = rng.integers(0, n - block + 1, size=(repeats, nblocks))
    offsets = np.arange(block)[None, None, :]
    indices = (starts[:, :, None] + offsets).reshape(repeats, -1)[:, :n]
    means = diff[indices].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return {
        "mae_delta_mamba_minus_gru": float(np.mean(diff)),
        "exploratory_block_bootstrap_95pct_interval": [float(lo), float(hi)],
        "block_length_frames": int(block),
        "bootstrap_repeats": int(repeats),
        "caveat": "Single 0027 development trajectory, NOT independent generalization",
    }


def compare_seed(
    root: Path, candidate: dict, seed: int, *,
    block: int = 32, repeats: int = 1000,
) -> dict:
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    g = read_run(root, "gru", seed, candidate)
    m = read_run(root, "mamba", seed, candidate)
    gc = g["metrics"]["config"]
    mc = m["metrics"]["config"]
    if any(gc.get(name) != mc.get(name) for name in CONFIG_KEYS):
        raise ValueError(
            "GRU and Mamba have different run configurations; "
            "do not label them matched.")
    ga, ma = g["arrays"], m["arrays"]
    if not np.array_equal(ga["synthetic_targets"],
                          ma["synthetic_targets"]):
        raise ValueError("not a paired validation comparison")
    result = {
        "schema": "isj_v7_gps_dev_pairwise_v1",
        "development_only": True,
        "independent_drive_claim": False,
        "seed": int(seed),
        "dataset_sha256": candidate["archive_sha256"],
        "validation_frame_count": int(len(candidate["anchors"])),
        "training_device": {
            "gru": g["metrics"]["device"],
            "mamba": m["metrics"]["device"],
        },
        "parameter_count": {
            "gru": int(g["metrics"]["parameter_count"]),
            "mamba": int(m["metrics"]["parameter_count"]),
        },
        "parameters_are_matched": (
            g["metrics"]["parameter_count"] ==
            m["metrics"]["parameter_count"]
        ),
        "epochs_best": {
            "gru": int(g["metrics"]["best_epoch"]),
            "mamba": int(m["metrics"]["best_epoch"]),
        },
        "comparison": {},
    }
    for j, h in enumerate(HORIZONS):
        target = candidate["targets"][:, j]
        gp = ga["reliability"][:, j]
        mp = ma["reliability"][:, j]
        gerr = np.abs(gp - target)
        merr = np.abs(mp - target)
        gmae = float(np.mean(gerr))
        mmae = float(np.mean(merr))
        grmse = float(np.sqrt(np.mean((gp - target) ** 2)))
        mrmse = float(np.sqrt(np.mean((mp - target) ** 2)))
        result["comparison"][h] = {
            "gru_mae": gmae,
            "mamba_mae": mmae,
            "mae_relative_improvement_pct": (
                100.0 * (gmae - mmae) / gmae if gmae > 0 else None
            ),
            "gru_rmse": grmse,
            "mamba_rmse": mrmse,
            "rmse_relative_improvement_pct": (
                100.0 * (grmse - mrmse) / grmse if grmse > 0 else None
            ),
            "paired_temporal_bootstrap": block_bootstrap_delta(
                gerr, merr, block=block, repeats=repeats,
                seed=seed * 7 + j,
            ),
        }
    for backend, run in (("gru", g), ("mamba", m)):
        a = run["arrays"]
        result[f"{backend}_uncertainty_diagnostics"] = (
            uncertainty_diagnostics(
                a["reliability"], a["evidence_uncertainty"],
                a["synthetic_targets"])
        )
    return result


def print_summary(result: dict) -> None:
    print(
        f"V7 DEV seed={result['seed']} "
        f"GRU({result['training_device']['gru']},"
        f"params={result['parameter_count']['gru']}) "
        f"MAMBA({result['training_device']['mamba']},"
        f"params={result['parameter_count']['mamba']})"
    )
    print("head   GRU MAE  Mamba MAE   MAE gain    GRU RMSE  Mamba RMSE")
    for h, item in result["comparison"].items():
        print(
            f"{h:>4s}   {item['gru_mae']:.5f}    "
            f"{item['mamba_mae']:.5f}   "
            f"{item['mae_relative_improvement_pct']:+7.2f}%    "
            f"{item['gru_rmse']:.5f}    {item['mamba_rmse']:.5f}"
        )
    print("Evidence diagnostics (Spearman: larger positive suggests"
          " evidence tracks larger absolute error):")
    for backend in ("gru", "mamba"):
        print(backend.upper())
        for h, score in result[f"{backend}_uncertainty_diagnostics"].items():
            print(
                f"  {h}: U_mean={score['evidence_mean']:.4f}, "
                f"U-error Spearman="
                f"{score['spearman_evidence_vs_absolute_error']}, "
                f"nominal80_coverage="
                f"{score['assumed_beta_80pct_interval_coverage']:.3f}"
            )
    print("WARNING: this is a previously explored 0027 development "
          "validation result; no cross-drive evidence.")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p.add_argument("--block-frames", type=int, default=32)
    p.add_argument("--bootstrap-repeats", type=int, default=1000)
    p.add_argument("--write-report", action="store_true")
    args = p.parse_args(argv)
    if len(set(args.seeds)) != len(args.seeds) or any(s < 0 for s in args.seeds):
        raise ValueError("duplicate or invalid seed")
    candidate = read_candidate()
    # Compute ALL first, no partially committed comparison report if a
    # backend or its validation predictions are missing/corrupt.
    reports = [
        compare_seed(
            DEFAULT_OUTPUT, candidate, seed,
            block=args.block_frames, repeats=args.bootstrap_repeats)
        for seed in args.seeds
    ]
    for result in reports:
        print_summary(result)
    if args.write_report:
        for result in reports:
            destination = COMPARISON_ROOT / f"seed_{result['seed']}"
            if destination.exists():
                raise FileExistsError(f"refusing overwrite: {destination}")
        COMPARISON_ROOT.mkdir(parents=True, exist_ok=True)
        for result in reports:
            destination = COMPARISON_ROOT / f"seed_{result['seed']}"
            destination.mkdir(exist_ok=False)
            (destination / "paired_metrics.json").write_text(
                json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            print("PAIR_REPORT_WRITTEN", destination / "paired_metrics.json")
    else:
        print("READ_ONLY: no comparison report files written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
