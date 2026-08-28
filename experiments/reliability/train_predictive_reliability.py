"""
Train Predictive Reliability V3
===============================

Dual-head training:
    current reliability head
    future reliability head

Important alignment
-------------------
At raw frame t:
    future_head[t]
predicts future-window reliability.

For factor graph frame j:
    predictive_prior[j]
is taken from:
    future_head[j-H]

Therefore the factor graph uses a prediction produced H frames earlier.

This is the key temporal alignment missing in previous versions.
"""

from __future__ import annotations

import os
import sys
import random

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

if ROOT not in sys.path:
    sys.path.insert(
        0,
        ROOT,
    )


from src.dataset.predictive_reliability_dataset import (
    build_aligned_data,
    DualHeadReliabilityDataset,
)

from src.model.predictive_reliability_model import (
    PredictiveReliabilityModel,
)


SEED = 20260826

SEQUENCE_LENGTH = 64

PREDICTION_HORIZON = 5

BATCH_SIZE = 64

EPOCHS = 80

PATIENCE = 12

LEARNING_RATE = 5e-4

WEIGHT_DECAY = 1e-4

CURRENT_LOSS_WEIGHT = 0.50

FUTURE_LOSS_WEIGHT = 1.00


OUTPUT_DIR = os.path.join(
    ROOT,
    "results",
    "predictive_reliability_v3",
)


def set_seed():
    random.seed(
        SEED
    )

    np.random.seed(
        SEED
    )

    torch.manual_seed(
        SEED
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            SEED
        )


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

    with torch.no_grad():
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
                ]
                .numpy()
                .tolist()
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
                ]
                .numpy()
                .tolist()
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
                ]
                .numpy()
                .tolist()
            )

    return (
        np.asarray(
            current_y,
            dtype=float,
        ),
        np.asarray(
            current_p,
            dtype=float,
        ),
        np.asarray(
            future_y,
            dtype=float,
        ),
        np.asarray(
            future_p,
            dtype=float,
        ),
        np.asarray(
            frame_ids,
            dtype=int,
        ),
    )


def metrics(
    target,
    prediction,
):
    target = np.asarray(
        target,
        dtype=float,
    )

    prediction = np.asarray(
        prediction,
        dtype=float,
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
        np.std(
            target
        )
        >
        1e-12
        and
        np.std(
            prediction
        )
        >
        1e-12
    ):
        corr = float(
            np.corrcoef(
                target,
                prediction,
            )[
                0,
                1
            ]
        )

    else:
        corr = 0.0

    return (
        mae,
        rmse,
        corr,
    )


def main():
    set_seed()

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else
        "cpu"
    )

    print("=" * 70)
    print(
        "Predictive Reliability V3 "
        "Dual-Head Training"
    )
    print(
        "Prediction horizon:",
        PREDICTION_HORIZON,
        "frames",
    )
    print(
        "Device:",
        device,
    )
    print("=" * 70)

    (
        features,
        current_labels,
        future_labels,
    ) = build_aligned_data()

    n = len(
        features
    )

    train_end = int(
        n
        *
        0.70
    )

    val_end = int(
        n
        *
        0.85
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

    train_dataset = (
        DualHeadReliabilityDataset(
            normalized,
            current_labels,
            future_labels,
            train_frames,
            sequence_length=
                SEQUENCE_LENGTH,
        )
    )

    val_dataset = (
        DualHeadReliabilityDataset(
            normalized,
            current_labels,
            future_labels,
            val_frames,
            sequence_length=
                SEQUENCE_LENGTH,
        )
    )

    test_dataset = (
        DualHeadReliabilityDataset(
            normalized,
            current_labels,
            future_labels,
            test_frames,
            sequence_length=
                SEQUENCE_LENGTH,
        )
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=
            BATCH_SIZE,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=
            BATCH_SIZE,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=
            BATCH_SIZE,
        shuffle=False,
    )

    model = (
        PredictiveReliabilityModel(
            input_dim=
                normalized.shape[
                    1
                ],
            hidden_dim=128,
            num_layers=2,
        )
        .to(
            device
        )
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

    best_val = float(
        "inf"
    )

    best_epoch = -1

    wait = 0

    for epoch in range(
        1,
        EPOCHS + 1,
    ):
        model.train()

        epoch_losses = []

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

            # Future low-reliability samples receive a
            # moderate emphasis without forcing collapse.
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

            epoch_losses.append(
                float(
                    loss.item()
                )
            )

        (
            val_current_y,
            val_current_p,
            val_future_y,
            val_future_p,
            _,
        ) = evaluate(
            model,
            val_loader,
            device,
        )

        current_metric = metrics(
            val_current_y,
            val_current_p,
        )

        future_metric = metrics(
            val_future_y,
            val_future_p,
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
            f"train {np.mean(epoch_losses):.6f} | "
            f"future val {val_loss:.6f} | "
            f"future corr {future_metric[2]:.4f} | "
            f"current corr {current_metric[2]:.4f}"
        )

        if val_loss < best_val:
            best_val = (
                val_loss
            )

            best_epoch = epoch

            wait = 0

            torch.save(
                model.state_dict(),
                os.path.join(
                    OUTPUT_DIR,
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
                OUTPUT_DIR,
                "best_model.pt",
            ),
            map_location=
                device,
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

    current_metric = metrics(
        test_current_y,
        test_current_p,
    )

    future_metric = metrics(
        test_future_y,
        test_future_p,
    )

    print()
    print("=" * 70)
    print(
        "Best epoch:",
        best_epoch,
    )
    print(
        "Current Head  "
        "MAE/RMSE/Corr:",
        *current_metric,
    )
    print(
        "Future Head   "
        "MAE/RMSE/Corr:",
        *future_metric,
    )
    print("=" * 70)

    # ---------------------------------------------------------
    # Generate predictions for all valid source frames
    # ---------------------------------------------------------

    all_dataset = (
        DualHeadReliabilityDataset(
            normalized,
            current_labels,
            future_labels,
            valid_frames,
            sequence_length=
                SEQUENCE_LENGTH,
        )
    )

    all_loader = DataLoader(
        all_dataset,
        batch_size=
            BATCH_SIZE,
        shuffle=False,
    )

    (
        _,
        all_current_prediction,
        _,
        all_future_prediction,
        all_frame_ids,
    ) = evaluate(
        model,
        all_loader,
        device,
    )

    current_prediction_full = np.ones(
        n,
        dtype=np.float64,
    )

    current_prediction_full[
        all_frame_ids
    ] = (
        all_current_prediction
    )

    # ---------------------------------------------------------
    # Correct online target alignment
    #
    # future prediction produced at source frame s
    # is assigned to target frame s + H.
    # ---------------------------------------------------------

    target_aligned_prediction = np.ones(
        n,
        dtype=np.float64,
    )

    target_source_frame = np.full(
        n,
        -1,
        dtype=np.int64,
    )

    for source_frame, prediction in zip(
        all_frame_ids,
        all_future_prediction,
    ):
        target_frame = (
            int(
                source_frame
            )
            +
            PREDICTION_HORIZON
        )

        if target_frame < n:
            target_aligned_prediction[
                target_frame
            ] = (
                prediction
            )

            target_source_frame[
                target_frame
            ] = (
                source_frame
            )

    # Early frames do not yet have H-frame-ahead
    # predictions. Use current-head estimates if available.
    for i in range(n):
        if target_source_frame[
            i
        ] < 0:
            target_aligned_prediction[
                i
            ] = (
                current_prediction_full[
                    i
                ]
            )

    np.savez(
        os.path.join(
            OUTPUT_DIR,
            "normalization.npz",
        ),
        mean=mean,
        std=std,
        sequence_length=
            np.asarray(
                [
                    SEQUENCE_LENGTH
                ],
                dtype=np.int64,
            ),
        prediction_horizon=
            np.asarray(
                [
                    PREDICTION_HORIZON
                ],
                dtype=np.int64,
            ),
    )

    np.savetxt(
        os.path.join(
            OUTPUT_DIR,
            "current_prediction.txt",
        ),
        current_prediction_full,
        fmt="%.8f",
    )

    np.savetxt(
        os.path.join(
            OUTPUT_DIR,
            "future_prediction_source_aligned.txt",
        ),
        np.pad(
            all_future_prediction,
            (
                SEQUENCE_LENGTH - 1,
                n
                -
                (
                    SEQUENCE_LENGTH - 1
                )
                -
                len(
                    all_future_prediction
                ),
            ),
            constant_values=1.0,
        ),
        fmt="%.8f",
    )

    np.savetxt(
        os.path.join(
            OUTPUT_DIR,
            "predictive_prior_target_aligned.txt",
        ),
        target_aligned_prediction,
        fmt="%.8f",
    )

    np.savetxt(
        os.path.join(
            OUTPUT_DIR,
            "prediction_source_frame.txt",
        ),
        target_source_frame,
        fmt="%d",
    )

    np.savetxt(
        os.path.join(
            OUTPUT_DIR,
            "test_frame_ids.txt",
        ),
        test_ids,
        fmt="%d",
    )

    print(
        "Saved target-aligned predictive prior:"
    )

    print(
        os.path.join(
            OUTPUT_DIR,
            "predictive_prior_target_aligned.txt",
        )
    )


if __name__ == "__main__":
    main()
