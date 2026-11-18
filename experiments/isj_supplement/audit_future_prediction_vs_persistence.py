from __future__ import annotations

"""READ-ONLY: true 3-frame-ahead reliability vs no-forecast persistence.

The V3 target generator stores future_factor_reliability[t] = current[t+3].
Both V5/V6 training datasets assign:
  head[0] = current[t]
  head[1] = current[t+3]  (NOT t+1)
  head[2] = current[t+2]
This script compares r_future[t+3|t]=head1 to persistence=head0[t|t],
using *identical information ending at time t*. Target is current[t+3].

Dev: 0027 saved V3 same-source labels, evaluate chronological 85-100%
portion for diagnostic; earlier iterative use of this sequence means this
is NOT a fresh untouched test set.
Cross: 0016 V3-*style* oracle formula against cached Fixed FGO reference;
not independent physical truth, and same 0016 drive was explored before.
No FGO optimization, no training, no file writes except optionally stdout
redirected by shell. Do NOT use 0016 numbers for hyperparameter selection.

WARNING: V6's frozen checkpoint training-data path is not recorded in the
checkpoint, so confirm whether its target definition matches V3 separately.
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

WINDOW = 64
LEAD = 3
SENSORS = ("gps", "imu", "camera")
V3 = ROOT / "results/predictive_factor_reliability_v3_same_source" / "factor_reliability_training_data.npz"


def source_indices(n: int, dataset: str):
    if n < WINDOW + LEAD + 1:
        raise ValueError(f"Need at least {WINDOW+LEAD+1} frames, got {n}")
    t = np.arange(WINDOW - 1, n - LEAD, dtype=np.int64)
    if dataset == "dev":
        # Report only the held-out chronological tail of the same 0027
        # training sequence. Earlier project analysis of this split makes
        # it an exploratory baseline, not independently unseen evidence.
        t = t[t >= int(n * 0.85)]
    return t


def evaluate_pair(
    reference_future: np.ndarray,
    current_prediction: np.ndarray,
    future_prediction: np.ndarray,
    reference_current: np.ndarray,
) -> dict:
    y = np.asarray(reference_future, float).reshape(-1)
    now = np.asarray(current_prediction, float).reshape(-1)
    pred = np.asarray(future_prediction, float).reshape(-1)
    previous_y = np.asarray(reference_current, float).reshape(-1)
    if not (y.shape == now.shape == pred.shape == previous_y.shape):
        raise ValueError("All arrays must share the same 1D shape")
    if len(y) < 3 or not all(np.isfinite(v).all() for v in (y, now, pred, previous_y)):
        raise ValueError("Expected >=3 finite, aligned source-target samples")

    errors_p = now - y
    errors_f = pred - y
    rmse_p = float(np.sqrt(np.mean(errors_p**2)))
    rmse_f = float(np.sqrt(np.mean(errors_f**2)))
    mae_p = float(np.mean(np.abs(errors_p)))
    mae_f = float(np.mean(np.abs(errors_f)))
    result = {
        "n": len(y),
        "target_mean": float(np.mean(y)),
        "target_std": float(np.std(y)),
        "pred_now_mean": float(np.mean(now)),
        "pred_future_mean": float(np.mean(pred)),
        "rmse_persistence": rmse_p,
        "rmse_forecast": rmse_f,
        "mae_persistence": mae_p,
        "mae_forecast": mae_f,
        "rmse_improvement_pct": 100 * (rmse_p-rmse_f)/max(rmse_p,1e-12),
        "mae_improvement_pct": 100 * (mae_p-mae_f)/max(mae_p,1e-12),
        "fraction_forecast_lower_abs_err": float(np.mean(np.abs(errors_f)<np.abs(errors_p))),
        "true_change_mean_abs": float(np.mean(np.abs(y-previous_y))),
    }
    # Predeclared 0.2 reliability-drop threshold for an exploratory
    # transition diagnostic. Do not fit/change it using 0016.
    worsening = (y-previous_y) <= -0.2
    result["worsening_n"] = int(np.sum(worsening))
    if worsening.any():
        result["worsening_rmse_persistence"] = float(
            np.sqrt(np.mean(errors_p[worsening]**2))
        )
        result["worsening_rmse_forecast"] = float(
            np.sqrt(np.mean(errors_f[worsening]**2))
        )
    return result


def get_data(dataset: str, sensor: str):
    if dataset == "dev":
        if not V3.is_file():
            raise FileNotFoundError(V3)
        with np.load(V3, allow_pickle=False) as z:
            x = np.asarray(z[f"{sensor}_features"], dtype=np.float32)
            labels = np.asarray(z[f"{sensor}_current_factor_reliability"],
                                dtype=np.float64).reshape(-1)
        return x, labels, None
    from experiments.isj_supplement.run_cross_date import (
        RESULT_ROOT, crossdate_bindings,
    )
    from experiments.isj_supplement.audit_cross_date_factor_targets import (
        reconstruct_targets,
    )
    seq = dataset
    out = RESULT_ROOT / seq
    with crossdate_bindings(seq, RESULT_ROOT, "fgo"):
        lab, n = reconstruct_targets(out, seq)
    with np.load(out / "features.npz", allow_pickle=False) as z:
        x = np.asarray(z[f"{sensor}_features"], dtype=np.float32)
    y = np.asarray(lab[f"{sensor}_reliability"], float)
    if len(x) != n or len(y) != n:
        raise ValueError("Cross-date data and labels disagree on length")
    return x, y, out


def load_model(model: str, sensor: str, device):
    import torch
    if model == "gru":
        from experiments.cross_sequence.stage7_config import V5_DIR
        from experiments.uncertainty_v5.reliability_uncertainty_model_v5 import (
            ReliabilityUncertaintyV5,
        )
        path = V5_DIR / sensor / "model.pt"
        if not path.is_file():
            raise FileNotFoundError(path)
        ck = torch.load(path, map_location="cpu", weights_only=False)
        m = ReliabilityUncertaintyV5(
            int(ck["input_dim"]), int(ck["hidden_dim"]),
            int(ck["layers"]), float(ck["dropout"]), int(ck["horizon"])
        )
        if ck.get("backend") != "gru" or m.backend != "gru":
            raise RuntimeError(
                "The frozen V5 checkpoint uses GRU but current Python "
                "backend did not instantiate GRU. Run V5 in the original "
                "GRU environment without mamba_ssm; do not substitute weights."
            )
        m.load_state_dict(ck["state_dict"], strict=True)
        mean = np.asarray(ck["mean"], np.float32).reshape(-1)
        scale = np.asarray(ck["std"], np.float32).reshape(-1)
        horizon = int(ck["horizon"])
        window = int(ck["window"])
    elif model == "mamba":
        from experiments.cross_sequence.stage7_config import V6_DIR
        from experiments.cross_sequence.infer_v6_mamba import (
            _load_checkpoint, _build_model,
        )
        ck = _load_checkpoint(V6_DIR / sensor / "best.pt")
        m = _build_model(ck, device)
        mean = np.asarray(ck["mean"], np.float32).reshape(-1)
        scale = np.asarray(ck["scale"], np.float32).reshape(-1)
        horizon = int(ck["horizon"])
        window = int(ck["window"])
        scale[scale < 1e-6] = 1.0
    else:
        raise ValueError(f"Unsupported model {model}")
    if window != WINDOW or horizon < 3:
        raise ValueError(f"Expected window=64 and horizon>=3, got {window}/{horizon}")
    m.eval()
    return m.to(device), mean, scale, int(ck["input_dim"])


def predict(model, net, features, idx, mean, scale, device, batch_size):
    import torch
    if features.ndim != 2 or features.shape[1] != len(mean):
        raise ValueError("Feature normalization dimension mismatch")
    if np.any(~np.isfinite(features)):
        raise ValueError("Features contain NaN/Inf")
    if np.any(scale <= 0) or not np.all(np.isfinite(scale)):
        raise ValueError("Invalid checkpoint normalizer")
    now = np.empty(len(idx), dtype=float)
    future = np.empty(len(idx), dtype=float)
    with torch.no_grad():
        for start in range(0, len(idx), batch_size):
            tt = idx[start:start+batch_size]
            batches = np.stack([
                (features[t-WINDOW+1:t+1]-mean)/scale for t in tt
            ], axis=0).astype(np.float32)
            output = net(torch.from_numpy(batches).to(device))
            head = (
                output["reliability"] if model == "mamba" else output[0]
            )
            now[start:start+len(tt)] = head[:,0].detach().cpu().numpy()
            future[start:start+len(tt)] = head[:,1].detach().cpu().numpy()
    if not np.isfinite(now).all() or not np.isfinite(future).all():
        raise ValueError("Nonfinite frozen predictions")
    return now, future


def check_current_cache(out, sensor, model, idx, prediction):
    if out is None:
        return
    source = "v5" if model == "gru" else "v6"
    path = out / "predictions" / source / f"{sensor}_reliability.txt"
    if not path.is_file():
        raise FileNotFoundError(path)
    saved = np.loadtxt(path, dtype=float).reshape(-1)
    if len(saved) <= int(idx[-1]):
        raise ValueError("Saved current predictions shorter than model input")
    delta = float(np.max(np.abs(saved[idx]-prediction)))
    print(f"  cached head0 consistency max abs difference={delta:.8f}")
    if delta > 1e-4:
        raise RuntimeError(
            f"Frozen {model}/{sensor} current head does NOT match existing "
            "0016 prediction files; check checkpoint/environment before "
            "interpreting future-head results"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="dev",
                        help="dev or a supported cross-date drive name")
    parser.add_argument("--model", choices=("gru","mamba"), default="gru")
    parser.add_argument("--device", choices=("cpu","cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if args.dataset != "dev":
        from experiments.isj_supplement.run_cross_date import date_for
        date_for(args.dataset)
    if args.batch_size < 1:
        parser.error("batch-size must be >=1")

    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available()=False")
    dev = torch.device(args.device)
    print("="*112)
    print("READ-ONLY FROZEN 3-FRAME-AHEAD FORECAST vs PERSISTENCE")
    print("head0[t]=current[t]; head1[t]=future[t+3]; head2[t]=current[t+2]")
    print(f"model={args.model}, dataset={args.dataset}, device={args.device}")
    print("Persistence uses head0[t], forecast uses head1[t]: same input ending at t.")
    print("Diagnostic only; 0016 is already explored and 0027 is development data.")
    print("="*112)

    for sensor in SENSORS:
        x, target, out = get_data(args.dataset, sensor)
        n = len(x)
        if len(target) != n:
            raise ValueError("Label-feature length mismatch")
        t = source_indices(n, args.dataset)
        net, mean, scale, inp = load_model(args.model, sensor, dev)
        if inp != x.shape[1]:
            raise ValueError(f"{sensor}: checkpoint vs feature dimension mismatch")
        now, future = predict(
            args.model, net, x, t, mean, scale, dev, args.batch_size
        )
        check_current_cache(out, sensor, args.model, t, now)
        scores = evaluate_pair(target[t+LEAD], now, future, target[t])
        print(
            f"\n{sensor.upper():7} n={scores['n']:4d} "
            f"source_range={int(t[0])}..{int(t[-1])} "
            f"future_target_range={int(t[0]+LEAD)}..{int(t[-1]+LEAD)}"
        )
        print(
            f"  RMSE: persistence={scores['rmse_persistence']:.6f}, "
            f"forecast={scores['rmse_forecast']:.6f}, "
            f"improvement={scores['rmse_improvement_pct']:+.2f}%"
        )
        print(
            f"  MAE : persistence={scores['mae_persistence']:.6f}, "
            f"forecast={scores['mae_forecast']:.6f}, "
            f"improvement={scores['mae_improvement_pct']:+.2f}%"
        )
        print(
            f"  Mean true change |r[t+3]-r[t]|={scores['true_change_mean_abs']:.6f} "
            f"paired win fraction={scores['fraction_forecast_lower_abs_err']:.3f}"
        )
        if scores["worsening_n"] > 0:
            print(
                f"  sharp drops (oracle target delta<=-0.2) "
                f"n={scores['worsening_n']}, "
                f"RMSE persistence={scores['worsening_rmse_persistence']:.6f}, "
                f"forecast={scores['worsening_rmse_forecast']:.6f}"
            )
        else:
            print("  sharp drops n=0 (no estimate)")
    print("\nFUTURE FORECAST AUDIT COMPLETE (no files written)")


if __name__ == "__main__":
    main()
