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

import argparse
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



def load_aligned_predictions(model, sensor, n):
    """Same frozen V5/V6 predictions for all read-only diagnostics."""
    root = MODELS[model]
    pr = root / f"{sensor}_reliability_target_aligned.txt"
    pu = root / f"{sensor}_uncertainty_target_aligned.txt"
    if not pr.is_file() or not pu.is_file():
        raise FileNotFoundError(f"{model}/{sensor} prediction missing")
    r = np.asarray(np.loadtxt(pr), float).reshape(-1)
    u = np.asarray(np.loadtxt(pu), float).reshape(-1)
    if len(r) != n or len(u) != n:
        raise ValueError(f"{model}/{sensor} length mismatch")
    if (not np.isfinite(r).all() or not np.isfinite(u).all()
            or np.any((r < 0) | (r > 1))
            or np.any((u < 0) | (u > 1))):
        raise ValueError(f"{model}/{sensor}: invalid prediction values")
    return r, u


def common_val_test_ranges(n):
    """Use exactly the same 0027 target frames across V5 and V6.

    VAL is fit-only; TEST is scored-only. These are internal same-drive
    segments, not clean held-out acquisition. Original three-step forecast
    horizons are preserved; 3860..3861 are omitted as a short gap.
    """
    v5 = split_ranges("GRU_V5", n)
    v6 = split_ranges("MAMBA_V6", n)
    val = (max(v5["VAL"][0], v6["VAL"][0]),
           min(v5["VAL"][1], v6["VAL"][1]))
    test = (max(v5["TEST"][0], v6["TEST"][0]),
            min(v5["TEST"][1], v6["TEST"][1]))
    if val[1] <= val[0] or test[1] <= test[0] or val[1] > test[0]:
        raise ValueError("Invalid common chronological VAL and TEST ranges")
    return val, test


def fit_val_risk_calibrator(x_val, y_val, x_test):
    """L2 logistic calibration; NO TEST labels/future errors used in fitting.

    StandardScaler is fitted on VAL only. Requires scikit-learn
    (the same ecosystem already used for reliability research).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    x_val = np.asarray(x_val, float)
    x_test = np.asarray(x_test, float)
    y_val = np.asarray(y_val, bool)
    if (x_val.ndim != 2 or x_test.ndim != 2
            or x_val.shape[1] != x_test.shape[1]
            or len(x_val) != len(y_val)
            or len(x_val) < 50
            or not np.isfinite(x_val).all()
            or not np.isfinite(x_test).all()
            or y_val.sum() < 5 or (~y_val).sum() < 5):
        raise ValueError("Invalid VAL calibration inputs")
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000),
    )
    model.fit(x_val, y_val.astype(int))
    return model.predict_proba(x_test)[:, 1]


def paired_auc_block_ci(y, a, b, *, block=32, reps=400, seed=20261119):
    """Exploratory paired moving-block bootstrap of AUC(b)-AUC(a).

    Blocks retain local temporal dependence. This CI is descriptive, not
    external generalization evidence. Never use it for parameter search.
    """
    y = np.asarray(y, bool)
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if (y.ndim != 1 or a.shape != y.shape or b.shape != y.shape
            or len(y) < block or block < 2 or reps < 10
            or not np.isfinite(a).all() or not np.isfinite(b).all()):
        raise ValueError("Invalid blocked bootstrap inputs")
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    n = len(y)
    starts = np.arange(n - block + 1)
    deltas = []
    for _ in range(reps):
        chosen = rng.choice(starts, size=int(np.ceil(n / block)), replace=True)
        index = np.concatenate([
            np.arange(i, i + block, dtype=int) for i in chosen
        ])[:n]
        z = y[index]
        if not z.any() or z.all():
            continue
        deltas.append(
            float(roc_auc_score(z, b[index]) - roc_auc_score(z, a[index]))
        )
    if len(deltas) < max(10, reps // 4):
        return (float("nan"), float("nan"), len(deltas))
    left, right = np.percentile(deltas, (2.5, 97.5))
    return float(left), float(right), len(deltas)


def conditional_added_value(target, r, u, val, test):
    """Evaluate the conditional contribution of U after knowing 1-r.

    Defines a HIGH r-ERROR outcome with a cutoff fixed at VAL's 80th
    percentile, then fits BOTH competing risk models on VAL and assesses
    both exclusively on the common future TEST indices. Labels come from
    non-independent V3 factor proxy, never from genuine GPS fault truth.

    The VAL-defined absolute cutoff is applied *unchanged* to TEST,
    unlike the earlier within-TEST top-20% descriptive AUC.
    """
    from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
    n = len(target)
    if any(np.asarray(x).shape != (n,) for x in (target, r, u)):
        raise ValueError("Reliability target/prediction dimension mismatch")
    if not all(np.isfinite(x).all() for x in (target, r, u)):
        raise ValueError("Nonfinite reliability target/prediction")
    a, b = val
    c, d = test
    if not (0 <= a < b <= c < d <= n):
        raise ValueError("Invalid ordered VAL/TEST range")
    error = np.abs(r - target)
    threshold = float(np.quantile(error[a:b], 0.8))
    y_val = error[a:b] >= threshold
    y_test = error[c:d] >= threshold
    if y_test.sum() < 2 or (~y_test).sum() < 2:
        raise ValueError("TEST contains too few outcomes at VAL cutoff")
    x = np.column_stack([1.0-r, u])
    base = fit_val_risk_calibrator(x[a:b, :1], y_val, x[c:d, :1])
    full = fit_val_risk_calibrator(x[a:b, :], y_val, x[c:d, :])
    base_auc = float(roc_auc_score(y_test, base))
    full_auc = float(roc_auc_score(y_test, full))
    base_brier = float(brier_score_loss(y_test, base))
    full_brier = float(brier_score_loss(y_test, full))
    base_loss = float(log_loss(y_test, base, labels=[0, 1]))
    full_loss = float(log_loss(y_test, full, labels=[0, 1]))
    ci_l, ci_u, accepted = paired_auc_block_ci(y_test, base, full)
    return {
        "val_n":int(b-a), "test_n":int(d-c),
        "val_high_error_threshold":threshold,
        "test_positive_fraction":float(y_test.mean()),
        "auc_ronly":base_auc,
        "auc_rplusu":full_auc,
        "delta_auc_rplusu_minus_r":full_auc-base_auc,
        "delta_auc_block_ci95":(ci_l, ci_u),
        "bootstrap_valid_reps":int(accepted),
        "brier_ronly":base_brier, "brier_rplusu":full_brier,
        "delta_brier_rminus_rplusu":base_brier-full_brier,
        "logloss_ronly":base_loss, "logloss_rplusu":full_loss,
        "delta_logloss_rminus_rplusu":base_loss-full_loss,
    }


def run_conditional_audit():
    """Optional, read-only, predeclared extra-information test."""
    with np.load(DATA, allow_pickle=False) as d:
        print("="*108)
        print("0027 VAL->TEST CONDITIONAL U INFORMATION AUDIT (NO TRAINING OF V5/V6, NO FGO)")
        print("Only simple L2 logistic risk calibrators are FIT on common VAL; "
              "common TEST is scored once. NO file writes.")
        print("VAL/TEST are SAME DRIVE and heavily explored, not fresh held-out drives.")
        print("V3 target is FGO-proxy label and model-specific abs-error; "
              "not independent sensor failure truth.")
        print("Beta evidence constraints may correlate U and r; "
              "AUC(U)>AUC(1-r) is insufficient on its own.")
        print("="*108)
        for sensor in SENSORS:
            label_key=f"{sensor}_current_factor_reliability"
            if label_key not in d.files:
                raise KeyError(label_key)
            target=np.asarray(d[label_key], float).reshape(-1)
            n=len(target)
            val,test=common_val_test_ranges(n)
            print(f"\n{sensor.upper()} common_VAL={val} common_TEST={test} "
                  f"(same matched frame indices for GRU and Mamba)")
            for name in MODELS:
                r,u=load_aligned_predictions(name,sensor,n)
                result=conditional_added_value(target,r,u,val,test)
                print(f"  {name:<9} VAL high-error threshold={result['val_high_error_threshold']:.5f} "
                      f"TEST positive={result['test_positive_fraction']:.2%} "
                      f"AUC(r)={result['auc_ronly']:.4f} "
                      f"AUC(r+U)={result['auc_rplusu']:.4f} "
                      f"deltaAUC={result['delta_auc_rplusu_minus_r']:+.4f} "
                      f"block95={result['delta_auc_block_ci95']} "
                      f"(valid={result['bootstrap_valid_reps']}) "
                      f"Brier_gain={result['delta_brier_rminus_rplusu']:+.5f} "
                      f"LogLoss_gain={result['delta_logloss_rminus_rplusu']:+.5f}",
                      flush=True)
        print("\nDONE. This is diagnostic intra-drive calibration; "
              "it does NOT demonstrate FGO gains or independent test generalization.")


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--conditional",action="store_true",
        help="Fit simple matched-frame VAL risk calibrators and evaluate "
             "r-only vs r+U on future TEST; never change V5/V6 models",
    )
    args=parser.parse_args(argv)
    if not DATA.is_file():
        raise FileNotFoundError(DATA)
    if args.conditional:
        run_conditional_audit()
        return
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
            for name in MODELS:
                r, u = load_aligned_predictions(name, sensor, n)
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
