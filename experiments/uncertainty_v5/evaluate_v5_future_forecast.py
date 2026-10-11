from __future__ import annotations

"""Frozen V5-GRU +3-frame reliability forecast validation (CPU only).

Runs on the chronological test tail of development sequence KITTI 0027.
The V3 dataset's stored future[t,0] equals current[t+3]; the V5 training
heads mean h0=t, h1=t+3, h2=t+2 (not regular t,t+1,t+2).

Compare h1 future predictions with two no-future-input baselines:
  1. h0-persistence: use predicted current reliability h0[t] for t+3;
  2. oracle-persistence: use ground-truth current label current[t] for t+3.
The oracle baseline is a diagnostic, NOT a deployable inference method.

This is retrospective/exploratory because the sequence and models have
already been investigated. It does NOT run the factor graph or prove any
pose-estimation improvement from the forecasting head.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.uncertainty_v5.reliability_uncertainty_model_v5 as model_module
from experiments.cross_sequence.stage7_config import V5_DIR

V3_DATA = (
    ROOT / "results" / "predictive_factor_reliability_v3_same_source"
    / "factor_reliability_training_data.npz"
)
OUT = ROOT / "results" / "forecast_validation_v5"
SENSORS = ("gps", "imu", "camera")
LEAD = 3


def score(y, pred):
    y = np.asarray(y, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    if y.shape != pred.shape or not np.isfinite(pred).all():
        raise ValueError(f"Invalid predictions: {y.shape} vs {pred.shape}")
    return {
        "mae": float(np.mean(np.abs(y - pred))),
        "rmse": float(np.sqrt(np.mean((y - pred) ** 2))),
        "corr": (
            float(np.corrcoef(y, pred)[0, 1])
            if np.std(y) > 1e-12 and np.std(pred) > 1e-12
            else None
        ),
    }


def evaluate_sensor(sensor: str, batch_size: int):
    with np.load(V3_DATA, allow_pickle=False) as data:
        x = np.asarray(data[f"{sensor}_features"], dtype=np.float32)
        current = np.asarray(
            data[f"{sensor}_current_factor_reliability"],
            dtype=np.float32,
        ).reshape(-1)
        future = np.asarray(
            data[f"{sensor}_future_factor_reliability"],
            dtype=np.float32,
        )
        if future.ndim == 1:
            future = future[:, None]
    n = len(x)
    if future.shape != (n, 1) or n != len(current):
        raise ValueError("Expected V3 one-column future label and equal lengths")
    if not np.array_equal(future[:-LEAD, 0], current[LEAD:]):
        raise ValueError("V3 future target is not exactly current[t+3]")

    ckpt_path = V5_DIR / sensor / "model.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if str(ckpt.get("backend", "")).lower() != "gru":
        raise RuntimeError(
            f"Expected V5 GRU checkpoint, got backend={ckpt.get('backend')!r}"
        )

    # V5 automatically selects Mamba when mamba_ssm is importable.
    # Force the architecture recorded in this GRU checkpoint, without
    # changing its source file or touching any trained weights.
    original_flag = model_module.HAS_MAMBA
    model_module.HAS_MAMBA = False
    try:
        model = model_module.ReliabilityUncertaintyV5(
            int(ckpt["input_dim"]),
            int(ckpt["hidden_dim"]),
            int(ckpt["layers"]),
            float(ckpt["dropout"]),
            int(ckpt["horizon"]),
        )
    finally:
        model_module.HAS_MAMBA = original_flag
    if model.backend != "gru" or int(ckpt["horizon"]) != 3:
        raise RuntimeError("Unexpected V5 model architecture")
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    mu = np.asarray(ckpt["mean"], dtype=np.float32).reshape(1, -1)
    sd = np.asarray(ckpt["std"], dtype=np.float32).reshape(1, -1)
    window = int(ckpt["window"])
    if x.shape[1] != mu.shape[1] or np.any(sd <= 0):
        raise ValueError("Checkpoint normalization does not fit V3 features")

    cut2 = int(0.85 * n)
    first = max(window - 1, cut2)
    # Avoid the last LEAD padded target positions, which have no future GT.
    indices = np.arange(first, n - LEAD, dtype=np.int64)
    if len(indices) == 0:
        raise RuntimeError("No chronological test-tail samples")

    predictions = []
    with torch.inference_mode():
        for i in range(0, len(indices), batch_size):
            ids = indices[i : i + batch_size]
            batch = np.stack(
                [(x[t - window + 1 : t + 1] - mu) / sd for t in ids]
            ).astype(np.float32)
            r, _u, _a, _b = model(torch.from_numpy(batch))
            predictions.append(r.cpu().numpy())
    forecasts = np.concatenate(predictions, axis=0)
    target = current[indices + LEAD]
    h0 = forecasts[:, 0]
    h1 = forecasts[:, 1]

    result = {
        "sensor": sensor,
        "data_source": str(V3_DATA.relative_to(ROOT)),
        "checkpoint": str(ckpt_path.relative_to(ROOT)),
        "status": "retrospective_exploratory_development_sequence",
        "model_backend": "gru",
        "sequence": "2011_10_03_drive_0027_sync",
        "input_window_frames": window,
        "lead_frames": LEAD,
        "test_start_index": int(indices[0]),
        "test_end_index_inclusive": int(indices[-1]),
        "samples": len(indices),
        "h0_target": "current[t]",
        "h1_target": "current[t+3]",
        "h2_target": "current[t+2]",
        "future_h1": score(target, h1),
        "model_h0_persistence": score(target, h0),
        "oracle_current_persistence_non_deployable": score(
            target, current[indices]
        ),
        "future_mae_improvement_vs_h0_persistence_pct": float(
            (np.mean(np.abs(target - h0)) - np.mean(np.abs(target - h1)))
            / max(np.mean(np.abs(target - h0)), 1e-12) * 100.0
        ),
        "does_not_prove_pose_estimation_improvement": True,
    }

    OUT.mkdir(parents=True, exist_ok=True)
    report = OUT / f"v5_{sensor}_h1_tplus3.json"
    report.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\n{sensor.upper()}  n_test={len(indices)}  t+3")
    print(f"  Future head (h1):     MAE={result['future_h1']['mae']:.6f}")
    print(f"  Model h0 persistence: MAE={result['model_h0_persistence']['mae']:.6f}")
    print(
        "  Oracle persistence:   "
        f"MAE={result['oracle_current_persistence_non_deployable']['mae']:.6f}"
    )
    print(
        "  Future vs h0 persistence: "
        f"{result['future_mae_improvement_vs_h0_persistence_pct']:+.2f}%"
    )
    print(f"  Report: {report}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sensor", choices=SENSORS, default="gps")
    p.add_argument("--batch-size", type=int, default=128)
    a = p.parse_args()
    if a.batch_size < 1:
        raise ValueError("batch-size must be positive")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    evaluate_sensor(a.sensor, a.batch_size)


if __name__ == "__main__":
    main()
