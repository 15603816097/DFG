from __future__ import annotations
import csv
import os
import random
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.dataset.factor_reliability_v2_dataset import (
    SENSORS,
    FactorReliabilityV2Dataset,
    compute_normalization,
    apply_normalization,
)
from src.model.multisensor_predictive_reliability_model import MultiSensorPredictiveReliabilityModel

DATA_PATH = os.path.join(ROOT, "results", "predictive_factor_reliability_v2", "factor_reliability_v2_data.npz")
OUTPUT_DIR = os.path.join(ROOT, "results", "predictive_factor_reliability_v2")

SEED = 20260827
SEQUENCE_LENGTH = 64
BATCH_SIZE = 64
EPOCHS = 140
PATIENCE = 20
LEARNING_RATE = 4e-4
WEIGHT_DECAY = 1e-4
CURRENT_WEIGHT = 0.25
FUTURE_WEIGHT = 1.0

LOW_RELIABILITY_EMPHASIS = {
    "gps": 2.0,
    "imu": 2.5,
    "lidar": 4.0,
    "camera": 4.0,
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
    mae = float(np.mean(np.abs(y - p)))
    rmse = float(np.sqrt(np.mean((y - p) ** 2)))
    if np.std(y) > 1e-12 and np.std(p) > 1e-12:
        corr = float(np.corrcoef(y, p)[0, 1])
    else:
        corr = 0.0

    bad = y < 0.5
    bad_recall = float(np.mean(p[bad] < 0.5)) if np.any(bad) else 1.0

    severe = y < 0.2
    severe_recall = float(np.mean(p[severe] < 0.2)) if np.any(severe) else 1.0

    return {
        "MAE": mae,
        "RMSE": rmse,
        "Correlation": corr,
        "BadRecall@0.5": bad_recall,
        "SevereRecall@0.2": severe_recall,
    }

def load_data():
    raw = np.load(DATA_PATH, allow_pickle=False)
    horizon = int(np.asarray(raw["horizon"]).reshape(-1)[0])
    features, current, future = {}, {}, {}
    lengths = []

    for sensor in SENSORS:
        features[sensor] = np.asarray(raw[f"{sensor}_features"], dtype=np.float32)
        current[sensor] = np.asarray(raw[f"{sensor}_current_label"], dtype=np.float32).reshape(-1)
        future[sensor] = np.asarray(raw[f"{sensor}_future_label"], dtype=np.float32).reshape(-1)
        lengths.extend([len(features[sensor]), len(current[sensor]), len(future[sensor])])

    n = min(lengths)

    for sensor in SENSORS:
        features[sensor] = features[sensor][:n]
        current[sensor] = current[sensor][:n]
        future[sensor] = future[sensor][:n]

    return features, current, future, horizon, n

def to_device(batch, device):
    return {s: batch[f"{s}_feature"].to(device) for s in SENSORS}

@torch.no_grad()
def collect(model, loader, device):
    model.eval()
    frame_ids = []
    output = {
        s: {k: [] for k in ("current_y", "current_p", "future_y", "future_p")}
        for s in SENSORS
    }

    for batch in loader:
        x = to_device(batch, device)
        pred = model(x["gps"], x["imu"], x["lidar"], x["camera"])
        frame_ids.extend(batch["frame_id"].numpy().tolist())

        for sensor in SENSORS:
            output[sensor]["current_y"].extend(batch[f"{sensor}_current_label"].numpy().tolist())
            output[sensor]["current_p"].extend(pred[sensor]["current"].detach().cpu().numpy().tolist())
            output[sensor]["future_y"].extend(batch[f"{sensor}_future_label"].numpy().tolist())
            output[sensor]["future_p"].extend(pred[sensor]["future"].detach().cpu().numpy().tolist())

    frame_ids = np.asarray(frame_ids, dtype=np.int64)
    for sensor in SENSORS:
        for key in output[sensor]:
            output[sensor][key] = np.asarray(output[sensor][key], dtype=np.float64)

    return frame_ids, output

def main():
    set_seed()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    features, current, future, horizon, n = load_data()

    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    normalization = compute_normalization(features, train_end)
    normalized = apply_normalization(features, normalization)

    valid_source_frames = np.arange(SEQUENCE_LENGTH - 1, n - horizon, dtype=np.int64)
    train_ids = valid_source_frames[valid_source_frames < train_end]
    val_ids = valid_source_frames[(valid_source_frames >= train_end) & (valid_source_frames < val_end)]
    test_ids = valid_source_frames[valid_source_frames >= val_end]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 112)
    print("PREDICTIVE FACTOR RELIABILITY V2 TRAINING")
    print("=" * 112)
    print("Device:", device)
    print("Frames:", n)
    print("Horizon:", horizon)
    print("Train / Val / Test:", len(train_ids), len(val_ids), len(test_ids))
    for sensor in SENSORS:
        print(f"{sensor:8s} dim={normalized[sensor].shape[1]}")

    def make_loader(ids, shuffle):
        return DataLoader(
            FactorReliabilityV2Dataset(
                normalized,
                current,
                future,
                ids,
                sequence_length=SEQUENCE_LENGTH,
            ),
            batch_size=BATCH_SIZE,
            shuffle=shuffle,
        )

    train_loader = make_loader(train_ids, True)
    val_loader = make_loader(val_ids, False)
    test_loader = make_loader(test_ids, False)

    model = MultiSensorPredictiveReliabilityModel(
        gps_dim=normalized["gps"].shape[1],
        imu_dim=normalized["imu"].shape[1],
        lidar_dim=normalized["lidar"].shape[1],
        camera_dim=normalized["camera"].shape[1],
        sensor_embed_dim=48,
        hidden_dim=160,
        num_layers=2,
        dropout=0.10,
    ).to(device)

    print("Temporal backend:", model.backend)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_score = float("inf")
    best_epoch = -1
    wait = 0
    best_path = os.path.join(OUTPUT_DIR, "best_model.pt")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []

        for batch in train_loader:
            x = to_device(batch, device)
            pred = model(x["gps"], x["imu"], x["lidar"], x["camera"])

            total = torch.zeros((), device=device)

            for sensor in SENSORS:
                yc = batch[f"{sensor}_current_label"].to(device)
                yf = batch[f"{sensor}_future_label"].to(device)

                current_loss = torch.mean((pred[sensor]["current"] - yc) ** 2)

                weight = 1.0 + LOW_RELIABILITY_EMPHASIS[sensor] * (1.0 - yf) ** 2
                future_loss = torch.mean(weight * (pred[sensor]["future"] - yf) ** 2)

                total = total + CURRENT_WEIGHT * current_loss + FUTURE_WEIGHT * future_loss

            total = total / len(SENSORS)

            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            losses.append(float(total.item()))

        _, val_out = collect(model, val_loader, device)
        val_mse = []
        text = []

        for sensor in SENSORS:
            y = val_out[sensor]["future_y"]
            p = val_out[sensor]["future_p"]
            val_mse.append(float(np.mean((y - p) ** 2)))
            m = metrics(y, p)
            text.append(f"{sensor}:{m['Correlation']:.3f}/{m['BadRecall@0.5']:.2f}")

        score = float(np.mean(val_mse))

        print(
            f"Epoch {epoch:03d} | "
            f"train {np.mean(losses):.6f} | "
            f"val {score:.6f} | "
            + " ".join(text)
        )

        if score < best_score:
            best_score = score
            best_epoch = epoch
            wait = 0
            torch.save(model.state_dict(), best_path)
        else:
            wait += 1
            if wait >= PATIENCE:
                print("Early stopping.")
                break

    model.load_state_dict(torch.load(best_path, map_location=device))

    test_frame_ids, test_out = collect(model, test_loader, device)

    print()
    print("=" * 112)
    print("BEST EPOCH:", best_epoch)
    print("=" * 112)

    rows = []

    for sensor in SENSORS:
        cm = metrics(test_out[sensor]["current_y"], test_out[sensor]["current_p"])
        fm = metrics(test_out[sensor]["future_y"], test_out[sensor]["future_p"])

        print()
        print(sensor.upper())
        print("Current:", cm)
        print("Future :", fm)

        rows.append({"sensor": sensor, "head": "current", **cm})
        rows.append({"sensor": sensor, "head": "future", **fm})

    with open(os.path.join(OUTPUT_DIR, "test_metrics.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sensor", "head", "MAE", "RMSE", "Correlation",
                "BadRecall@0.5", "SevereRecall@0.2"
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    norm_payload = {
        "sequence_length": np.asarray([SEQUENCE_LENGTH], dtype=np.int64),
        "horizon": np.asarray([horizon], dtype=np.int64),
    }
    for sensor in SENSORS:
        norm_payload[f"{sensor}_mean"] = normalization[sensor]["mean"]
        norm_payload[f"{sensor}_std"] = normalization[sensor]["std"]

    np.savez(os.path.join(OUTPUT_DIR, "normalization.npz"), **norm_payload)

    full_loader = make_loader(valid_source_frames, False)
    source_ids, full_out = collect(model, full_loader, device)

    for sensor in SENSORS:
        current_prediction = np.ones(n, dtype=np.float64)
        target_aligned = np.ones(n, dtype=np.float64)
        source_frame = np.full(n, -1, dtype=np.int64)

        current_prediction[source_ids] = full_out[sensor]["current_p"]

        for sid, pred_value in zip(source_ids, full_out[sensor]["future_p"]):
            target_id = int(sid) + horizon
            if target_id < n:
                target_aligned[target_id] = float(pred_value)
                source_frame[target_id] = int(sid)

        for i in range(n):
            if source_frame[i] < 0:
                target_aligned[i] = current_prediction[i]

        np.savetxt(
            os.path.join(OUTPUT_DIR, f"{sensor}_predictive_prior_target_aligned.txt"),
            target_aligned,
            fmt="%.8f",
        )

        np.savetxt(
            os.path.join(OUTPUT_DIR, f"{sensor}_prediction_source_frame.txt"),
            source_frame,
            fmt="%d",
        )

    np.savetxt(
        os.path.join(OUTPUT_DIR, "test_source_frame_ids.txt"),
        test_frame_ids,
        fmt="%d",
    )

    print()
    print("Saved:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
