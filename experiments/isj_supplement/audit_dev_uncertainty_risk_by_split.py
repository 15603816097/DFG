from __future__ import annotations

"""Read-only audit: does evidential U rank reliability-prediction error?

Uses original V5/V6 chronological split definitions (not a guessed common
70/15/15 split): V5 train/val/test on full n, V6 on n-horizon+1.
Reports U-error Pearson/Spearman and AUROC of predicting the worst 20%
prediction errors WITHIN each split. No training, GT used for weights,
inference, GTSAM optimization, threshold fitting, or file writing.

V3 factor-error labels use a Fixed FGO measurement-derived pose reference,
NOT independent truth. The 0027 data are model-development data and have
already been inspected many times: TEST segment is only an internal
chronological holdout, not a fresh independent KITTI drive. V6 checkpoint
--data provenance is still unverified.
"""

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SENSORS = ("gps", "imu", "camera")
HORIZON = 3
WINDOW = 64
DATA = (ROOT / "results/predictive_factor_reliability_v3_same_source"
        / "factor_reliability_training_data.npz")
MODELS = {
    "GRU_V5": ROOT / "results/sensor_specific_reliability_v5_uncertainty",
    "MAMBA_V6": ROOT / "results/v6_mamba_uncertainty",
}


def finite_corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.shape != y.shape or x.ndim != 1 or len(x) < 3:
        return float("nan")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        return float("nan")
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0,1])


def ranks_with_ties(x):
    """Average ranks, 1-based, without scipy dependency."""
    a = np.asarray(x, float)
    if a.ndim != 1:
        raise ValueError("Ranking requires 1D values")
    order = np.argsort(a, kind="mergesort")
    ranked = np.empty(len(a), dtype=float)
    i = 0
    while i < len(a):
        j = i + 1
        while j < len(a) and a[order[j]] == a[order[i]]:
            j += 1
        ranked[order[i:j]] = (i+1 + j)/2.0
        i = j
    return ranked


def rank_auc(scores, positive):
    scores = np.asarray(scores, dtype=float)
    y = np.asarray(positive, dtype=bool)
    if scores.ndim != 1 or y.shape != scores.shape:
        raise ValueError("AUROC requires matching 1D arrays")
    if len(scores) == 0 or not np.isfinite(scores).all():
        return float("nan")
    npos = int(y.sum())
    nneg = len(y) - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    ranks = ranks_with_ties(scores)
    return float((ranks[y].sum() - npos*(npos+1)/2.0)/(npos*nneg))


def split_ranges(model: str, n: int):
    if model == "GRU_V5":
        # V5 split cutoffs use original n, but build_targets creates an
        # NaN second-future label in the final HORIZON-1 frames; its
        # `valid` index mask excludes these final placeholder frames.
        cut1, cut2, usable_end = int(.70*n), int(.85*n), n-HORIZON+1
    elif model == "MAMBA_V6":
        usable_end = n - HORIZON + 1
        cut1, cut2 = int(.70*usable_end), int(.85*usable_end)
    else:
        raise ValueError(model)
    first = WINDOW-1
    return {
        "TRAIN": (first, cut1),
        "VAL": (cut1, cut2),
        "TEST": (cut2, usable_end),
    }


def test_risk_metrics(u, error):
    n = len(u)
    if n < 10 or np.std(error) < 1e-12:
        return float("nan"), float("nan"), float("nan")
    pearson = finite_corr(u, error)
    spearman = finite_corr(ranks_with_ties(u), ranks_with_ties(error))
    # AUC is a ranking metric, with the threshold fixed by the 80th
    # percentile of THIS audited segment; no classifier is trained here.
    threshold = float(np.quantile(error, .8))
    label = error >= threshold
    auc = rank_auc(u, label)
    return pearson, spearman, auc


def main():
    if not DATA.is_file():
        raise FileNotFoundError(DATA)
    print("="*108)
    print("READ-ONLY 0027 U->PREDICTION-ERROR RISK AUDIT; NO FILES WRITTEN")
    print("Intra-drive chronological subsets follow EXACT V5/V6 training code.")
    print("Warning: 0027 labels use non-independent FGO proxy; V6 source not proven.")
    print("Warning: V5/V6 test partitions are NOT new external held-out KITTI drives.")
    print("AUC=0.5 represents chance ranking for high-error frames.")
    print("="*108)
    with np.load(DATA, allow_pickle=False) as d:
        for sensor in SENSORS:
            label_key = f"{sensor}_current_factor_reliability"
            if label_key not in d.files:
                raise KeyError(label_key)
            target = np.asarray(d[label_key], float).reshape(-1)
            n = len(target)
            if not np.isfinite(target).all():
                raise ValueError(f"Nonfinite V3 target for {sensor}")
            print(f"\n========== {sensor.upper()} n={n} ==========")
            for name, root in MODELS.items():
                pr = root / f"{sensor}_reliability_target_aligned.txt"
                pu = root / f"{sensor}_uncertainty_target_aligned.txt"
                if not pr.is_file() or not pu.is_file():
                    raise FileNotFoundError(f"{name}/{sensor} prediction missing")
                r = np.asarray(np.loadtxt(pr), float).reshape(-1)
                u = np.asarray(np.loadtxt(pu), float).reshape(-1)
                if len(r) != n or len(u) != n:
                    raise ValueError(f"{name}/{sensor} length mismatch")
                if (not np.isfinite(r).all() or not np.isfinite(u).all()
                        or np.any((r<0)|(r>1)) or np.any((u<0)|(u>1))):
                    raise ValueError(f"{name}/{sensor}: invalid prediction values")
                splits = split_ranges(name, n)
                print(f"  {name:<8} label window and split boundaries={splits}")
                for split, (start, end) in splits.items():
                    if end <= start:
                        raise ValueError(f"{name}/{split}: empty split")
                    err = np.abs(r[start:end] - target[start:end])
                    us = u[start:end]
                    pear, spear, auc = test_risk_metrics(us, err)
                    # Critical negative control: network uncertainty must
                    # add information beyond the trivially derived 1-r risk.
                    # The same within-split top-20%-error labels are reused.
                    _, _, r_auc = test_risk_metrics(1.0-r[start:end], err)
                    print(
                        f"    {split:<5} n={len(err):4} "
                        f"rMAE={err.mean():.5f} "
                        f"U_mean={us.mean():.5f} "
                        f"Pearson(U,abs_err)={pear:+.4f} "
                        f"Spearman={spear:+.4f} "
                        f"AUC(U)={auc:.4f} "
                        f"AUC(1-r)={r_auc:.4f} "
                        f"deltaU_minus_r={auc-r_auc:+.4f}"
                    )
    print("\nDONE. Do not tune on 0016 or treat this as independent calibration proof.")


if __name__ == "__main__":
    main()
