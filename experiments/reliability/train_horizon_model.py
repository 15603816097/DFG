"""
Train one predictive reliability model for a specified horizon.

Example
-------
python experiments/reliability/train_horizon_model.py --horizon 5

Outputs
-------
results/horizon_sensitivity/h5/
    best_model.pt
    normalization.npz
    current_prediction.txt
    future_prediction_source_aligned.txt
    predictive_prior_target_aligned.txt
    prediction_source_frame.txt
    future_label.txt
    test_metrics.txt

Important
---------
For target frame t, predictive_prior_target_aligned[t] is generated
from a model prediction made at frame t-H.
"""

from __future__ import annotations

import argparse
import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from src.loader.imu_loader import IMULoader
from src.reliability.health_features import build_health_features
from src.reliability.label_generator import future_window_min
from src.model.predictive_reliability_model import PredictiveReliabilityModel


DATASET = os.path.join(
    ROOT,
    "dataset",
    "kitti",
    "2011_10_03",
    "2011_10_03_drive_0027_sync",
)

GPS_PATH = os.path.join(
    ROOT,
    "results",
    "progressive_gps_degradation",
    "gps_corrupted.txt",
)

CURRENT_LABEL_PATH = os.path.join(
    ROOT,
    "results",
    "predictive_reliability_labels",
    "current_reliability.txt",
)


SEED = 20260826
SEQUENCE_LENGTH = 64
BATCH_SIZE = 64
EPOCHS = 80
PATIENCE = 12
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4

CURRENT_LOSS_WEIGHT = 0.50
FUTURE_LOSS_WEIGHT = 1.00


class HorizonDataset(Dataset):
    def __init__(
        self,
        features,
        current_labels,
        future_labels,
        frame_ids,
    ):
        self.features = torch.from_numpy(
            np.asarray(
                features,
                dtype=np.float32,
            )
        )

        self.current_labels = torch.from_numpy(
            np.asarray(
                current_labels,
                dtype=np.float32,
            )
        )

        self.future_labels = torch.from_numpy(
            np.asarray(
                future_labels,
                dtype=np.float32,
            )
        )

        self.frame_ids = np.asarray(
            frame_ids,
            dtype=np.int64,
        )

    def __len__(self):
        return len(self.frame_ids)

    def __getitem__(self, index):
        frame = int(
            self.frame_ids[index]
        )

        start = (
            frame
            -
            SEQUENCE_LENGTH
            +
            1
        )

        x = self.features[
            start:
            frame + 1
        ]

        return {
            "feature": x,
            "current_label":
                self.current_labels[
                    frame
                ],
            "future_label":
                self.future_labels[
                    frame
                ],
            "frame_id":
                frame,
        }


def load_imu():
    loader = IMULoader(
        DATASET
    )

    acceleration = []
    gyro = []

    for i in range(len(loader)):
        item = loader[i]

        acceleration.append(
            item["acceleration"]
        )

        gyro.append(
            item[
                "angular_velocity"
            ]
        )

    return (
        np.asarray(
            acceleration,
            dtype=np.float64,
        ),
        np.asarray(
            gyro,
            dtype=np.float64,
        ),
    )


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            SEED
        )


def regression_metrics(
    target,
    prediction,
):
    target = np.asarray(
        target,
        dtype=np.float64,
    )

    prediction = np.asarray(
        prediction,
        dtype=np.float64,
    )

    mae = float(
        np.mean(
            np.abs(
                target
                -
                prediction
            )
        )
    )

    rmse = float(
        np.sqrt(
            np.mean(
                (
                    target
                    -
                    prediction
                )
                ** 2
            )
        )
    )

    if (
        np.std(target) > 1e-12
        and
        np.std(prediction) > 1e-12
    ):
        corr = float(
            np.corrcoef(
                target,
                prediction,
            )[0, 1]
        )
    else:
        corr = 0.0

    return (
        mae,
        rmse,
        corr,
    )


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
):
    model.eval()

    current_y = []
    current_p = []
    future_y = []
    future_p = []
    frame_ids = []

    for batch in loader:
        x = batch[
            "feature"
        ].to(
            device
        )

        output = model(
            x
        )

        current_y.extend(
            batch[
                "current_label"
            ].numpy().tolist()
        )

        current_p.extend(
            output[
                "current"
            ]
            .cpu()
            .numpy()
            .tolist()
        )

        future_y.extend(
            batch[
                "future_label"
            ].numpy().tolist()
        )

        future_p.extend(
            output[
                "future"
            ]
            .cpu()
            .numpy()
            .tolist()
        )

        frame_ids.extend(
            batch[
                "frame_id"
            ].numpy().tolist()
        )

    return (
        np.asarray(
            current_y,
            dtype=np.float64,
        ),
        np.asarray(
            current_p,
            dtype=np.float64,
        ),
        np.asarray(
            future_y,
            dtype=np.float64,
        ),
        np.asarray(
            future_p,
            dtype=np.float64,
        ),
        np.asarray(
            frame_ids,
            dtype=np.int64,
        ),
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--horizon",
        type=int,
        required=True,
    )

    args = parser.parse_args()

    horizon = int(
        args.horizon
    )

    if horizon < 1:
        raise ValueError(
            "horizon must be >= 1"
        )

    set_seed()

    output_dir = os.path.join(
        ROOT,
        "results",
        "horizon_sensitivity",
        f"h{horizon}",
    )

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else
        "cpu"
    )

    print("=" * 72)
    print(
        "Horizon Sensitivity Training"
    )
    print(
        "Horizon:",
        horizon,
        "frames",
    )
    print(
        "Approx. lead time:",
        f"{horizon * 0.1:.1f} s",
    )
    print(
        "Device:",
        device,
    )
    print("=" * 72)

    gps = np.loadtxt(
        GPS_PATH,
        dtype=np.float64,
    )

    current_label = np.loadtxt(
        CURRENT_LABEL_PATH,
        dtype=np.float64,
    ).reshape(-1)

    acceleration, gyro = load_imu()

    n = min(
        len(gps),
        len(current_label),
        len(acceleration),
        len(gyro),
    )

    gps = gps[:n]
    current_label = (
        current_label[:n]
    )
    acceleration = (
        acceleration[:n]
    )
    gyro = gyro[:n]

    future_label = future_window_min(
        current_label,
        horizon=horizon,
    )

    features = build_health_features(
        gps,
        acceleration,
        gyro,
        dt=0.1,
        window=10,
    )

    train_end = int(
        n * 0.70
    )

    val_end = int(
        n * 0.85
    )

    mean = features[
        :train_end
    ].mean(
        axis=0
    )

    std = features[
        :train_end
    ].std(
        axis=0
    )

    std[
        std < 1e-6
    ] = 1.0

    normalized = (
        (
            features
            -
            mean
        )
        /
        std
    ).astype(
        np.float32
    )

    valid_frames = np.arange(
        SEQUENCE_LENGTH - 1,
        n,
        dtype=np.int64,
    )

    train_frames = valid_frames[
        valid_frames
        <
        train_end
    ]

    val_frames = valid_frames[
        (
            valid_frames
            >=
            train_end
        )
        &
        (
            valid_frames
            <
            val_end
        )
    ]

    test_frames = valid_frames[
        valid_frames
        >=
        val_end
    ]

    train_loader = DataLoader(
        HorizonDataset(
            normalized,
            current_label,
            future_label,
            train_frames,
        ),
        batch_size=
            BATCH_SIZE,
        shuffle=True,
    )

    val_loader = DataLoader(
        HorizonDataset(
            normalized,
            current_label,
            future_label,
            val_frames,
        ),
        batch_size=
            BATCH_SIZE,
        shuffle=False,
    )

    test_loader = DataLoader(
        HorizonDataset(
            normalized,
            current_label,
            future_label,
            test_frames,
        ),
        batch_size=
            BATCH_SIZE,
        shuffle=False,
    )

    model = PredictiveReliabilityModel(
        input_dim=
            normalized.shape[1],
        hidden_dim=128,
        num_layers=2,
    ).to(
        device
    )

    print(
        "Temporal backend:",
        model.backend,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=
            LEARNING_RATE,
        weight_decay=
            WEIGHT_DECAY,
    )

    best_val_loss = float(
        "inf"
    )

    best_epoch = -1
    wait = 0

    for epoch in range(
        1,
        EPOCHS + 1,
    ):
        model.train()

        losses = []

        for batch in train_loader:
            x = batch[
                "feature"
            ].to(
                device
            )

            current_target = batch[
                "current_label"
            ].to(
                device
            )

            future_target = batch[
                "future_label"
            ].to(
                device
            )

            output = model(
                x
            )

            current_error = (
                output[
                    "current"
                ]
                -
                current_target
            )

            future_error = (
                output[
                    "future"
                ]
                -
                future_target
            )

            future_weight = (
                1.0
                +
                1.5
                *
                (
                    1.0
                    -
                    future_target
                )
            )

            current_loss = (
                torch.mean(
                    current_error
                    ** 2
                )
            )

            future_loss = (
                torch.mean(
                    future_weight
                    *
                    future_error
                    ** 2
                )
            )

            loss = (
                CURRENT_LOSS_WEIGHT
                *
                current_loss
                +
                FUTURE_LOSS_WEIGHT
                *
                future_loss
            )

            optimizer.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                5.0,
            )

            optimizer.step()

            losses.append(
                float(
                    loss.item()
                )
            )

        (
            _,
            _,
            val_future_y,
            val_future_p,
            _,
        ) = evaluate(
            model,
            val_loader,
            device,
        )

        val_metrics = (
            regression_metrics(
                val_future_y,
                val_future_p,
            )
        )

        val_loss = float(
            np.mean(
                (
                    val_future_y
                    -
                    val_future_p
                )
                ** 2
            )
        )

        print(
            f"Epoch {epoch:03d} | "
            f"train {np.mean(losses):.6f} | "
            f"val {val_loss:.6f} | "
            f"future corr "
            f"{val_metrics[2]:.4f}"
        )

        if (
            val_loss
            <
            best_val_loss
        ):
            best_val_loss = (
                val_loss
            )

            best_epoch = epoch
            wait = 0

            torch.save(
                model.state_dict(),
                os.path.join(
                    output_dir,
                    "best_model.pt",
                ),
            )

        else:
            wait += 1

            if wait >= PATIENCE:
                print(
                    "Early stopping."
                )
                break

    model.load_state_dict(
        torch.load(
            os.path.join(
                output_dir,
                "best_model.pt",
            ),
            map_location=device,
        )
    )

    (
        test_current_y,
        test_current_p,
        test_future_y,
        test_future_p,
        test_ids,
    ) = evaluate(
        model,
        test_loader,
        device,
    )

    current_metrics = (
        regression_metrics(
            test_current_y,
            test_current_p,
        )
    )

    future_metrics = (
        regression_metrics(
            test_future_y,
            test_future_p,
        )
    )

    print()
    print(
        "Best epoch:",
        best_epoch,
    )
    print(
        "Current Head MAE/RMSE/Corr:",
        *current_metrics,
    )
    print(
        "Future Head MAE/RMSE/Corr:",
        *future_metrics,
    )

    all_loader = DataLoader(
        HorizonDataset(
            normalized,
            current_label,
            future_label,
            valid_frames,
        ),
        batch_size=
            BATCH_SIZE,
        shuffle=False,
    )

    (
        _,
        all_current_p,
        _,
        all_future_p,
        all_frame_ids,
    ) = evaluate(
        model,
        all_loader,
        device,
    )

    current_prediction = np.ones(
        n,
        dtype=np.float64,
    )

    source_aligned_future = np.ones(
        n,
        dtype=np.float64,
    )

    target_aligned_prior = np.ones(
        n,
        dtype=np.float64,
    )

    source_frame_for_target = np.full(
        n,
        -1,
        dtype=np.int64,
    )

    current_prediction[
        all_frame_ids
    ] = all_current_p

    source_aligned_future[
        all_frame_ids
    ] = all_future_p

    for source_frame, prediction in zip(
        all_frame_ids,
        all_future_p,
    ):
        target_frame = (
            int(
                source_frame
            )
            +
            horizon
        )

        if target_frame < n:
            target_aligned_prior[
                target_frame
            ] = prediction

            source_frame_for_target[
                target_frame
            ] = source_frame

    for i in range(n):
        if (
            source_frame_for_target[i]
            <
            0
        ):
            target_aligned_prior[i] = (
                current_prediction[i]
            )

    np.savez(
        os.path.join(
            output_dir,
            "normalization.npz",
        ),
        mean=mean,
        std=std,
        horizon=np.asarray(
            [horizon],
            dtype=np.int64,
        ),
        sequence_length=np.asarray(
            [
                SEQUENCE_LENGTH
            ],
            dtype=np.int64,
        ),
    )

    np.savetxt(
        os.path.join(
            output_dir,
            "future_label.txt",
        ),
        future_label,
        fmt="%.8f",
    )

    np.savetxt(
        os.path.join(
            output_dir,
            "current_prediction.txt",
        ),
        current_prediction,
        fmt="%.8f",
    )

    np.savetxt(
        os.path.join(
            output_dir,
            "future_prediction_source_aligned.txt",
        ),
        source_aligned_future,
        fmt="%.8f",
    )

    np.savetxt(
        os.path.join(
            output_dir,
            "predictive_prior_target_aligned.txt",
        ),
        target_aligned_prior,
        fmt="%.8f",
    )

    np.savetxt(
        os.path.join(
            output_dir,
            "prediction_source_frame.txt",
        ),
        source_frame_for_target,
        fmt="%d",
    )

    with open(
        os.path.join(
            output_dir,
            "test_metrics.txt",
        ),
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            f"horizon={horizon}\n"
        )

        f.write(
            f"best_epoch={best_epoch}\n"
        )

        f.write(
            "current_mae="
            f"{current_metrics[0]:.8f}\n"
        )

        f.write(
            "current_rmse="
            f"{current_metrics[1]:.8f}\n"
        )

        f.write(
            "current_corr="
            f"{current_metrics[2]:.8f}\n"
        )

        f.write(
            "future_mae="
            f"{future_metrics[0]:.8f}\n"
        )

        f.write(
            "future_rmse="
            f"{future_metrics[1]:.8f}\n"
        )

        f.write(
            "future_corr="
            f"{future_metrics[2]:.8f}\n"
        )

    print(
        "Saved:",
        output_dir,
    )


if __name__ == "__main__":
    main()
