from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset.factor_reliability_dataset import (
    SENSORS,
    PredictiveFactorReliabilityDataset,
    compute_normalization,
    apply_normalization,
)
from src.model.multisensor_predictive_reliability_model import (
    MultiSensorPredictiveReliabilityModel,
)

SEED = 20260827
SEQUENCE_LENGTH = 64
BATCH_SIZE = 64
EPOCHS = 120
PATIENCE = 18
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4
CURRENT_WEIGHT = 0.30
FUTURE_WEIGHT = 1.00
LOW_RELIABILITY_EMPHASIS = 2.5


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def metric(y, p):
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    corr = (
        float(np.corrcoef(y, p)[0, 1])
        if np.std(y) > 1e-12 and np.std(p) > 1e-12
        else 0.0
    )
    return {
        "mae": float(np.mean(np.abs(y - p))),
        "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
        "corr": corr,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default=str(
            ROOT
            / "results/predictive_factor_reliability_v3_same_source"
            / "factor_reliability_training_data.npz"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(
            ROOT
            / "results/predictive_factor_reliability_v3_same_source"
        ),
    )
    args = parser.parse_args()

    set_seed()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    raw = np.load(args.data, allow_pickle=False)
    horizon = int(np.asarray(raw["horizon"]).reshape(-1)[0])

    features, current, future = {}, {}, {}
    lengths = []

    for s in SENSORS:
        features[s] = np.asarray(
            raw[f"{s}_features"], dtype=np.float32
        )
        current[s] = np.asarray(
            raw[f"{s}_current_factor_reliability"],
            dtype=np.float32,
        ).reshape(-1)
        future[s] = np.asarray(
            raw[f"{s}_future_factor_reliability"],
            dtype=np.float32,
        ).reshape(-1)
        lengths.extend(
            [len(features[s]), len(current[s]), len(future[s])]
        )

    n = min(lengths)
    for s in SENSORS:
        features[s] = features[s][:n]
        current[s] = current[s][:n]
        future[s] = future[s][:n]

    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    normalization = compute_normalization(
        features, train_end
    )
    normalized = apply_normalization(
        features, normalization
    )

    normalization_payload = {
        "sequence_length": np.asarray(
            [SEQUENCE_LENGTH], dtype=np.int64
        ),
        "horizon": np.asarray(
            [horizon], dtype=np.int64
        ),
    }

    for s in SENSORS:
        normalization_payload[f"{s}_mean"] = normalization[s]["mean"]
        normalization_payload[f"{s}_std"] = normalization[s]["std"]

    np.savez_compressed(
        out / "factor_reliability_normalization.npz",
        **normalization_payload,
    )

    valid_source_frames = np.arange(
        SEQUENCE_LENGTH - 1,
        n - horizon,
        dtype=np.int64,
    )

    train_ids = valid_source_frames[
        valid_source_frames < train_end
    ]
    val_ids = valid_source_frames[
        (valid_source_frames >= train_end)
        & (valid_source_frames < val_end)
    ]
    test_ids = valid_source_frames[
        valid_source_frames >= val_end
    ]

    def make_loader(ids, shuffle):
        ds = PredictiveFactorReliabilityDataset(
            normalized,
            current,
            future,
            ids,
            sequence_length=SEQUENCE_LENGTH,
        )
        return DataLoader(
            ds,
            batch_size=BATCH_SIZE,
            shuffle=shuffle,
        )

    train_loader = make_loader(train_ids, True)
    val_loader = make_loader(val_ids, False)
    test_loader = make_loader(test_ids, False)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = MultiSensorPredictiveReliabilityModel(
        gps_dim=normalized["gps"].shape[1],
        imu_dim=normalized["imu"].shape[1],
        lidar_dim=normalized["lidar"].shape[1],
        camera_dim=normalized["camera"].shape[1],
        sensor_embed_dim=32,
        hidden_dim=128,
        num_layers=2,
        dropout=0.10,
    ).to(device)

    print("=" * 110)
    print("TRAIN SAME-SOURCE FACTOR RELIABILITY V3")
    print("=" * 110)
    print("Device:", device)
    print("Backend:", model.backend)
    print("Frames:", n)
    print("Horizon:", horizon)
    print(
        "Train / Val / Test:",
        len(train_ids),
        len(val_ids),
        len(test_ids),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_score = float("inf")
    best_epoch = -1
    wait = 0
    best_path = out / "best_factor_reliability_model.pt"

    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []

        for batch in train_loader:
            x = {
                s: batch[f"{s}_feature"].to(device)
                for s in SENSORS
            }
            pred = model(
                x["gps"],
                x["imu"],
                x["lidar"],
                x["camera"],
            )

            total_loss = torch.zeros(
                (), device=device
            )

            for s in SENSORS:
                y_current = batch[
                    f"{s}_current_label"
                ].to(device)
                y_future = batch[
                    f"{s}_future_label"
                ].to(device)

                current_loss = torch.mean(
                    (
                        pred[s]["current"]
                        - y_current
                    )
                    ** 2
                )

                future_weight = (
                    1.0
                    + LOW_RELIABILITY_EMPHASIS
                    * (1.0 - y_future) ** 2
                )

                future_loss = torch.mean(
                    future_weight
                    * (
                        pred[s]["future"]
                        - y_future
                    )
                    ** 2
                )

                total_loss = (
                    total_loss
                    + CURRENT_WEIGHT * current_loss
                    + FUTURE_WEIGHT * future_loss
                )

            total_loss = total_loss / len(SENSORS)

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 5.0
            )
            optimizer.step()

            losses.append(
                float(total_loss.item())
            )

        model.eval()
        validation_mse = []

        with torch.no_grad():
            for s in SENSORS:
                ys, ps = [], []

                for batch in val_loader:
                    x = {
                        q: batch[f"{q}_feature"].to(device)
                        for q in SENSORS
                    }

                    pred = model(
                        x["gps"],
                        x["imu"],
                        x["lidar"],
                        x["camera"],
                    )

                    ys.append(
                        batch[
                            f"{s}_future_label"
                        ].numpy()
                    )
                    ps.append(
                        pred[s]["future"]
                        .cpu()
                        .numpy()
                    )

                y = np.concatenate(ys)
                p = np.concatenate(ps)

                validation_mse.append(
                    float(
                        np.mean(
                            (y - p) ** 2
                        )
                    )
                )

        score = float(
            np.mean(validation_mse)
        )

        print(
            f"Epoch {epoch:03d} | "
            f"train={np.mean(losses):.6f} | "
            f"val={score:.6f}"
        )

        if score < best_score:
            best_score = score
            best_epoch = epoch
            wait = 0
            torch.save(
                model.state_dict(),
                best_path,
            )
        else:
            wait += 1
            if wait >= PATIENCE:
                print("Early stopping.")
                break

    model.load_state_dict(
        torch.load(
            best_path,
            map_location=device,
        )
    )
    model.eval()

    test_metrics = {}

    with torch.no_grad():
        for s in SENSORS:
            ys, ps = [], []

            for batch in test_loader:
                x = {
                    q: batch[f"{q}_feature"].to(device)
                    for q in SENSORS
                }

                pred = model(
                    x["gps"],
                    x["imu"],
                    x["lidar"],
                    x["camera"],
                )

                ys.append(
                    batch[
                        f"{s}_future_label"
                    ].numpy()
                )

                ps.append(
                    pred[s]["future"]
                    .cpu()
                    .numpy()
                )

            y = np.concatenate(ys)
            p = np.concatenate(ps)
            test_metrics[s] = metric(y, p)

    # Full source-aligned frozen inference for later factor-graph wiring.
    full_loader = make_loader(
        valid_source_frames,
        False,
    )

    source_ids = []
    full_current = {
        s: [] for s in SENSORS
    }
    full_future = {
        s: [] for s in SENSORS
    }

    with torch.no_grad():
        for batch in full_loader:
            x = {
                q: batch[f"{q}_feature"].to(device)
                for q in SENSORS
            }
            pred = model(
                x["gps"],
                x["imu"],
                x["lidar"],
                x["camera"],
            )

            source_ids.extend(
                batch["frame_id"]
                .numpy()
                .tolist()
            )

            for s in SENSORS:
                full_current[s].extend(
                    pred[s]["current"]
                    .cpu()
                    .numpy()
                    .tolist()
                )
                full_future[s].extend(
                    pred[s]["future"]
                    .cpu()
                    .numpy()
                    .tolist()
                )

    source_ids = np.asarray(
        source_ids, dtype=np.int64
    )

    for s in SENSORS:
        current_prediction = np.ones(
            n, dtype=np.float64
        )
        target_aligned = np.ones(
            n, dtype=np.float64
        )
        source_frame = np.full(
            n, -1, dtype=np.int64
        )

        cp = np.asarray(
            full_current[s],
            dtype=np.float64,
        )
        fp = np.asarray(
            full_future[s],
            dtype=np.float64,
        )

        current_prediction[
            source_ids
        ] = cp

        for source_id, prediction in zip(
            source_ids, fp
        ):
            target_id = int(
                source_id + horizon
            )
            if target_id < n:
                target_aligned[
                    target_id
                ] = float(prediction)
                source_frame[
                    target_id
                ] = int(source_id)

        for i in range(n):
            if source_frame[i] < 0:
                target_aligned[i] = (
                    current_prediction[i]
                )

        np.savetxt(
            out
            / f"{s}_predictive_prior_target_aligned.txt",
            target_aligned,
            fmt="%.8f",
        )

        np.savetxt(
            out
            / f"{s}_prediction_source_frame.txt",
            source_frame,
            fmt="%d",
        )

    summary = {
        "best_epoch": best_epoch,
        "best_validation_mse": best_score,
        "backend": model.backend,
        "sequence_length": SEQUENCE_LENGTH,
        "horizon": horizon,
        "train_windows": int(len(train_ids)),
        "val_windows": int(len(val_ids)),
        "test_windows": int(len(test_ids)),
        "test_metrics": test_metrics,
    }

    (
        out / "training_summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("Best epoch:", best_epoch)
    for s in SENSORS:
        print(
            f"{s:8s}",
            test_metrics[s],
        )
    print()
    print("Saved model:", best_path)
    print("Saved target-aligned predictions:", out)


if __name__ == "__main__":
    main()
