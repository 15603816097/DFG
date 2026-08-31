from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.dataflow_fix.sensor_specific_predictor_v4 import (
    SensorSpecificPredictor,
)

SENSORS = ("gps", "imu", "camera")
SEED = 20260827
WINDOW = 64
BATCH = 64
EPOCHS = 120
PATIENCE = 18
LR = 5e-4
WEIGHT_DECAY = 1e-4
CURRENT_WEIGHT = 0.30
FUTURE_WEIGHT = 1.00
LOW_REL_EMPHASIS = 2.5


class SensorDataset(Dataset):
    def __init__(self, x, yc, yf, ids, window):
        self.x = x
        self.yc = yc
        self.yf = yf
        self.ids = np.asarray(ids, dtype=np.int64)
        self.window = int(window)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        t = int(self.ids[idx])
        start = t - self.window + 1
        return {
            "feature": torch.from_numpy(
                self.x[start : t + 1]
            ).float(),
            "current": torch.tensor(
                self.yc[t], dtype=torch.float32
            ),
            "future": torch.tensor(
                self.yf[t], dtype=torch.float32
            ),
            "frame_id": torch.tensor(
                t, dtype=torch.long
            ),
        }


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def metrics(y, p):
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    c = (
        float(np.corrcoef(y, p)[0, 1])
        if np.std(y) > 1e-12 and np.std(p) > 1e-12
        else 0.0
    )
    return {
        "mae": float(np.mean(np.abs(y - p))),
        "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
        "corr": c,
    }


def train_one(sensor, raw, out, device):
    x = np.asarray(raw[f"{sensor}_features"], np.float32)
    yc = np.asarray(
        raw[f"{sensor}_current_factor_reliability"],
        np.float32,
    ).reshape(-1)
    yf = np.asarray(
        raw[f"{sensor}_future_factor_reliability"],
        np.float32,
    ).reshape(-1)

    n = min(len(x), len(yc), len(yf))
    x, yc, yf = x[:n], yc[:n], yf[:n]

    horizon = int(np.asarray(raw["horizon"]).reshape(-1)[0])
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    mean = x[:train_end].mean(axis=0)
    std = x[:train_end].std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    xn = ((x - mean) / std).astype(np.float32)

    ids = np.arange(WINDOW - 1, n - horizon, dtype=np.int64)
    train_ids = ids[ids < train_end]
    val_ids = ids[(ids >= train_end) & (ids < val_end)]
    test_ids = ids[ids >= val_end]

    def loader(frame_ids, shuffle):
        return DataLoader(
            SensorDataset(xn, yc, yf, frame_ids, WINDOW),
            batch_size=BATCH,
            shuffle=shuffle,
        )

    tl = loader(train_ids, True)
    vl = loader(val_ids, False)
    xl = loader(test_ids, False)

    model = SensorSpecificPredictor(
        input_dim=x.shape[1],
        hidden_dim=96,
        num_layers=2,
        dropout=0.10,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    sensor_dir = out / sensor
    sensor_dir.mkdir(parents=True, exist_ok=True)
    best_path = sensor_dir / "best_model.pt"

    best = float("inf")
    best_epoch = -1
    wait = 0

    print()
    print("-" * 100)
    print(f"TRAIN {sensor.upper()} SENSOR-SPECIFIC PREDICTOR")
    print("-" * 100)
    print("Backend:", model.backend)
    print("Feature dim:", x.shape[1])
    print(
        "Train / Val / Test:",
        len(train_ids), len(val_ids), len(test_ids)
    )

    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []

        for b in tl:
            xx = b["feature"].to(device)
            c = b["current"].to(device)
            f = b["future"].to(device)
            pred = model(xx)

            lc = torch.mean((pred["current"] - c) ** 2)
            wf = 1.0 + LOW_REL_EMPHASIS * (1.0 - f) ** 2
            lf = torch.mean(wf * (pred["future"] - f) ** 2)
            loss = CURRENT_WEIGHT * lc + FUTURE_WEIGHT * lf

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 5.0
            )
            optimizer.step()
            losses.append(float(loss.item()))

        model.eval()
        yy, pp = [], []
        with torch.no_grad():
            for b in vl:
                pred = model(b["feature"].to(device))
                yy.append(b["future"].numpy())
                pp.append(pred["future"].cpu().numpy())

        y = np.concatenate(yy)
        p = np.concatenate(pp)
        score = float(np.mean((y - p) ** 2))

        print(
            f"{sensor} epoch {epoch:03d} | "
            f"train={np.mean(losses):.6f} | val={score:.6f}"
        )

        if score < best:
            best = score
            best_epoch = epoch
            wait = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "input_dim": int(x.shape[1]),
                    "hidden_dim": 96,
                    "num_layers": 2,
                    "dropout": 0.10,
                    "backend": model.backend,
                    "sensor": sensor,
                    "window": WINDOW,
                    "horizon": horizon,
                },
                best_path,
            )
        else:
            wait += 1
            if wait >= PATIENCE:
                print("Early stopping.")
                break

    checkpoint = torch.load(
        best_path, map_location=device
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    yy, pp = [], []
    with torch.no_grad():
        for b in xl:
            pred = model(b["feature"].to(device))
            yy.append(b["future"].numpy())
            pp.append(pred["future"].cpu().numpy())

    test_metric = metrics(
        np.concatenate(yy),
        np.concatenate(pp),
    )

    # Full causal source-aligned inference.
    full_loader = loader(ids, False)
    source_ids = []
    current_pred = []
    future_pred = []

    with torch.no_grad():
        for b in full_loader:
            pred = model(b["feature"].to(device))
            source_ids.extend(b["frame_id"].numpy().tolist())
            current_pred.extend(
                pred["current"].cpu().numpy().tolist()
            )
            future_pred.extend(
                pred["future"].cpu().numpy().tolist()
            )

    source_ids = np.asarray(source_ids, dtype=np.int64)
    current_pred = np.asarray(current_pred, dtype=np.float64)
    future_pred = np.asarray(future_pred, dtype=np.float64)

    current_full = np.ones(n, dtype=np.float64)
    target_aligned = np.ones(n, dtype=np.float64)
    source_frame = np.full(n, -1, dtype=np.int64)

    current_full[source_ids] = current_pred

    for source, pred in zip(source_ids, future_pred):
        target = int(source + horizon)
        if target < n:
            target_aligned[target] = float(pred)
            source_frame[target] = int(source)

    # Warmup/tail fallback uses current prediction where available.
    for i in range(n):
        if source_frame[i] < 0:
            target_aligned[i] = current_full[i]

    np.savez_compressed(
        sensor_dir / "normalization.npz",
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        window=np.asarray([WINDOW], dtype=np.int64),
        horizon=np.asarray([horizon], dtype=np.int64),
    )

    np.savetxt(
        out / f"{sensor}_predictive_prior_target_aligned.txt",
        target_aligned,
        fmt="%.8f",
    )
    np.savetxt(
        out / f"{sensor}_prediction_source_frame.txt",
        source_frame,
        fmt="%d",
    )

    result = {
        "sensor": sensor,
        "backend": model.backend,
        "best_epoch": best_epoch,
        "best_validation_mse": best,
        "test": test_metric,
        "prediction_min": float(target_aligned.min()),
        "prediction_max": float(target_aligned.max()),
        "prediction_mean": float(target_aligned.mean()),
        "prediction_std": float(target_aligned.std()),
    }

    (sensor_dir / "summary.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print(
        f"{sensor.upper()} TEST:",
        test_metric,
    )
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data",
        default=str(
            ROOT
            / "results/predictive_factor_reliability_v3_same_source"
            / "factor_reliability_training_data.npz"
        ),
    )
    p.add_argument(
        "--output-dir",
        default=str(
            ROOT
            / "results/sensor_specific_reliability_v4"
        ),
    )
    a = p.parse_args()

    set_seed()
    raw = np.load(a.data, allow_pickle=False)
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 100)
    print("SENSOR-SPECIFIC PREDICTIVE RELIABILITY V4")
    print("=" * 100)
    print("Device:", device)
    print("Training source:", a.data)
    print("LiDAR shared head: DISABLED in final architecture")
    print("LiDAR final covariance: existing dedicated factor-error predictor")

    results = {}
    for sensor in SENSORS:
        results[sensor] = train_one(
            sensor, raw, out, device
        )

    summary = {
        "version": "sensor_specific_v4",
        "sensors": list(SENSORS),
        "lidar_policy": (
            "Dedicated LiDAR factor-error predictor remains unchanged; "
            "no shared LiDAR reliability head is used."
        ),
        "results": results,
    }
    (out / "training_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 100)
    print("V4 FINAL TEST CORRELATIONS")
    print("=" * 100)
    for sensor in SENSORS:
        print(
            f"{sensor:8s}: "
            f"{results[sensor]['test']['corr']:.6f}"
        )
    print("Saved:", out)


if __name__ == "__main__":
    main()
