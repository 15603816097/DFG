"""
Train Predictive Mamba Reliability V2
=====================================

Multi-task learning:
    1. future reliability regression
    2. future reliability state classification

Prediction horizon:
    5 frames

At 10 Hz:
    approximately 0.5 seconds ahead.

Required existing file
----------------------
src/dataset/predictive_reliability_dataset.py

Outputs
-------
results/predictive_reliability_model_v2/

    best_model.pt
    last_model.pt
    normalization.npz
    training_history.txt
    test_metrics.txt

    predicted_reliability_h5.txt

    predicted_state_h5.txt

    predicted_state_probability_h5.txt

The three output files are aligned to raw KITTI frame indices.
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(
    __file__
).resolve().parents[
    1
]


if str(
    PROJECT_ROOT
) not in sys.path:

    sys.path.insert(
        0,
        str(
            PROJECT_ROOT
        ),
    )


try:

    from src.dataset.predictive_reliability_dataset import (

        DEFAULT_DATASET,

        DEFAULT_GPS_PATH,

        DEFAULT_LABEL_PATH,

        PredictiveReliabilitySequenceDataset,

        build_predictive_frame_features,

        build_window_split_indices,

        compute_normalization,

        normalize_features,

    )


except ModuleNotFoundError:

    from src.datasets.predictive_reliability_dataset import (

        DEFAULT_DATASET,

        DEFAULT_GPS_PATH,

        DEFAULT_LABEL_PATH,

        PredictiveReliabilitySequenceDataset,

        build_predictive_frame_features,

        build_window_split_indices,

        compute_normalization,

        normalize_features,

    )


try:

    from src.model.predictive_mamba_reliability import (

        PredictiveMambaReliabilityModel,

        MultiTaskPredictiveReliabilityLoss,

        reliability_to_state,

    )


except ModuleNotFoundError:

    from src.models.predictive_mamba_reliability import (

        PredictiveMambaReliabilityModel,

        MultiTaskPredictiveReliabilityLoss,

        reliability_to_state,

    )


# ============================================================
# Configuration
# ============================================================

SEED = 42


FEATURE_WINDOW = 10


SEQUENCE_LENGTH = 64


SEQUENCE_STRIDE = 8


PREDICTION_HORIZON = 5


SPLIT_BLOCK_SIZE = 256


BATCH_SIZE = 16


NUM_WORKERS = 0


EPOCHS = 80


LEARNING_RATE = 5e-4


WEIGHT_DECAY = 1e-5


PATIENCE = 12


GRAD_CLIP = 5.0


BAD_THRESHOLD = 0.20


RELIABLE_THRESHOLD = 0.80


RESULT_DIR = (

    PROJECT_ROOT

    /

    "results"

    /

    "predictive_reliability_model_v2"

)


BEST_MODEL_PATH = (

    RESULT_DIR

    /

    "best_model.pt"

)


LAST_MODEL_PATH = (

    RESULT_DIR

    /

    "last_model.pt"

)


NORMALIZATION_PATH = (

    RESULT_DIR

    /

    "normalization.npz"

)


HISTORY_PATH = (

    RESULT_DIR

    /

    "training_history.txt"

)


TEST_METRICS_PATH = (

    RESULT_DIR

    /

    "test_metrics.txt"

)


PREDICTION_PATH = (

    RESULT_DIR

    /

    "predicted_reliability_h5.txt"

)


STATE_PATH = (

    RESULT_DIR

    /

    "predicted_state_h5.txt"

)


STATE_PROBABILITY_PATH = (

    RESULT_DIR

    /

    "predicted_state_probability_h5.txt"

)


SOURCE_FRAME_PATH = (

    RESULT_DIR

    /

    "prediction_source_frame.txt"

)


TARGET_FRAME_PATH = (

    RESULT_DIR

    /

    "prediction_target_frame.txt"

)


# ============================================================
# General utilities
# ============================================================

def set_seed(
    seed,
):

    random.seed(
        seed
    )


    np.random.seed(
        seed
    )


    torch.manual_seed(
        seed
    )


    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )


def choose_device():

    if torch.cuda.is_available():

        return torch.device(
            "cuda"
        )


    return torch.device(
        "cpu"
    )


def reliability_to_state_numpy(
    reliability,
):

    reliability = np.asarray(
        reliability,
        dtype=np.float64,
    )


    state = np.ones(
        reliability.shape,
        dtype=np.int64,
    )


    state[
        reliability
        <
        BAD_THRESHOLD
    ] = 0


    state[
        reliability
        >=
        RELIABLE_THRESHOLD
    ] = 2


    return state


def classification_metrics(
    prediction_state,
    target_state,
):

    prediction_state = np.asarray(
        prediction_state,
        dtype=np.int64,
    ).reshape(
        -1
    )


    target_state = np.asarray(
        target_state,
        dtype=np.int64,
    ).reshape(
        -1
    )


    accuracy = float(
        np.mean(
            prediction_state
            ==
            target_state
        )
    )


    confusion = np.zeros(
        (
            3,
            3,
        ),
        dtype=np.int64,
    )


    for target_value, prediction_value in zip(

        target_state,

        prediction_state,

    ):

        confusion[
            target_value,
            prediction_value
        ] += 1


    # Bad recall

    bad_target = (
        target_state
        ==
        0
    )


    if np.any(
        bad_target
    ):

        bad_recall = float(
            np.mean(
                prediction_state[
                    bad_target
                ]
                ==
                0
            )
        )


        false_reliable_rate = float(
            np.mean(
                prediction_state[
                    bad_target
                ]
                ==
                2
            )
        )


    else:

        bad_recall = 0.0

        false_reliable_rate = 0.0


    # Reliable precision

    predicted_reliable = (
        prediction_state
        ==
        2
    )


    if np.any(
        predicted_reliable
    ):

        reliable_precision = float(
            np.mean(
                target_state[
                    predicted_reliable
                ]
                ==
                2
            )
        )


    else:

        reliable_precision = 0.0


    # Reliable recall

    reliable_target = (
        target_state
        ==
        2
    )


    if np.any(
        reliable_target
    ):

        reliable_recall = float(
            np.mean(
                prediction_state[
                    reliable_target
                ]
                ==
                2
            )
        )


    else:

        reliable_recall = 0.0


    return {

        "StateAccuracy":
            accuracy,


        "BadRecall":
            bad_recall,


        "FalseReliableRate":
            false_reliable_rate,


        "ReliablePrecision":
            reliable_precision,


        "ReliableRecall":
            reliable_recall,


        "ConfusionMatrix":
            confusion,

    }


def regression_metrics(
    prediction,
    target,
):

    prediction = np.asarray(
        prediction,
        dtype=np.float64,
    ).reshape(
        -1
    )


    target = np.asarray(
        target,
        dtype=np.float64,
    ).reshape(
        -1
    )


    error = (
        prediction
        -
        target
    )


    mae = float(
        np.mean(
            np.abs(
                error
            )
        )
    )


    rmse = float(
        np.sqrt(
            np.mean(
                error
                **
                2
            )
        )
    )


    if (
        np.std(
            prediction
        )
        >
        1e-12

        and

        np.std(
            target
        )
        >
        1e-12
    ):

        correlation = float(
            np.corrcoef(

                prediction,

                target,

            )[
                0,
                1
            ]
        )


    else:

        correlation = 0.0


    return {

        "MAE":
            mae,


        "RMSE":
            rmse,


        "Correlation":
            correlation,

    }


# ============================================================
# Epoch
# ============================================================

def run_epoch(
    model,
    loader,
    loss_fn,
    device,
    optimizer=None,
):

    training = (
        optimizer
        is not None
    )


    if training:

        model.train()


    else:

        model.eval()


    total_loss_sum = 0.0

    classification_loss_sum = 0.0

    regression_loss_sum = 0.0

    safety_loss_sum = 0.0

    batch_count = 0


    regression_predictions = []

    regression_targets = []

    state_predictions = []

    state_targets = []


    for batch in loader:

        feature = (

            batch[
                "feature"
            ]

            .to(
                device
            )

        )


        target_reliability = (

            batch[
                "label"
            ]

            .to(
                device
            )

        )


        if training:

            optimizer.zero_grad(
                set_to_none=True
            )


        with torch.set_grad_enabled(
            training
        ):

            output = model(
                feature
            )


            loss_dict = loss_fn(

                output,

                target_reliability,

            )


            total_loss = (
                loss_dict[
                    "total"
                ]
            )


            if training:

                total_loss.backward()


                torch.nn.utils.clip_grad_norm_(

                    model.parameters(),

                    GRAD_CLIP,

                )


                optimizer.step()


        total_loss_sum += float(
            loss_dict[
                "total"
            ].item()
        )


        classification_loss_sum += float(
            loss_dict[
                "classification"
            ].item()
        )


        regression_loss_sum += float(
            loss_dict[
                "regression"
            ].item()
        )


        safety_loss_sum += float(
            loss_dict[
                "safety"
            ].item()
        )


        batch_count += 1


        regression = (

            output[
                "fused_reliability"
            ]

            .detach()

            .cpu()

            .numpy()

        )


        state_prediction = (

            output[
                "state_logits"
            ]

            .argmax(
                dim=-1
            )

            .detach()

            .cpu()

            .numpy()

        )


        target_numpy = (

            target_reliability

            .detach()

            .cpu()

            .numpy()

        )


        target_state = (

            reliability_to_state(

                target_reliability,

                bad_threshold=
                    BAD_THRESHOLD,

                reliable_threshold=
                    RELIABLE_THRESHOLD,

            )

            .detach()

            .cpu()

            .numpy()

        )


        regression_predictions.append(
            regression.reshape(
                -1
            )
        )


        regression_targets.append(
            target_numpy.reshape(
                -1
            )
        )


        state_predictions.append(
            state_prediction.reshape(
                -1
            )
        )


        state_targets.append(
            target_state.reshape(
                -1
            )
        )


    if batch_count == 0:

        raise RuntimeError(
            "Empty DataLoader"
        )


    regression_prediction = np.concatenate(
        regression_predictions
    )


    regression_target = np.concatenate(
        regression_targets
    )


    state_prediction = np.concatenate(
        state_predictions
    )


    state_target = np.concatenate(
        state_targets
    )


    metrics = regression_metrics(

        regression_prediction,

        regression_target,

    )


    metrics.update(

        classification_metrics(

            state_prediction,

            state_target,

        )

    )


    metrics[
        "Loss"
    ] = (
        total_loss_sum
        /
        batch_count
    )


    metrics[
        "ClassificationLoss"
    ] = (
        classification_loss_sum
        /
        batch_count
    )


    metrics[
        "RegressionLoss"
    ] = (
        regression_loss_sum
        /
        batch_count
    )


    metrics[
        "SafetyLoss"
    ] = (
        safety_loss_sum
        /
        batch_count
    )


    return metrics


# ============================================================
# Predict full target timeline
# ============================================================

@torch.no_grad()
def predict_full_timeline(
    model,
    normalized_features,
    target_frame_ids,
    raw_frame_count,
    device,
):

    model.eval()


    feature_tensor = torch.from_numpy(

        normalized_features.astype(
            np.float32
        )

    )


    fused_rows = []

    state_rows = []

    probability_rows = []


    sequence_batch = []


    batch_limit = 128


    def flush_batch():

        if len(
            sequence_batch
        ) == 0:

            return


        batch = torch.stack(

            sequence_batch,

            dim=0,

        ).to(
            device
        )


        output = model(
            batch
        )


        fused = (

            output[
                "fused_reliability"
            ][
                :,
                -1
            ]

            .detach()

            .cpu()

            .numpy()

        )


        probability = (

            output[
                "state_probability"
            ][
                :,
                -1,
                :
            ]

            .detach()

            .cpu()

            .numpy()

        )


        state = np.argmax(
            probability,
            axis=1,
        )


        fused_rows.extend(
            fused.tolist()
        )


        state_rows.extend(
            state.tolist()
        )


        probability_rows.extend(
            probability.tolist()
        )


        sequence_batch.clear()


    for i in range(
        len(
            normalized_features
        )
    ):

        start = max(

            0,

            i
            -
            SEQUENCE_LENGTH
            +
            1,

        )


        sequence = feature_tensor[
            start:
            i + 1
        ]


        if len(
            sequence
        ) < SEQUENCE_LENGTH:

            padding_count = (

                SEQUENCE_LENGTH

                -

                len(
                    sequence
                )

            )


            padding = (

                sequence[
                    0:
                    1
                ]

                .repeat(
                    padding_count,
                    1,
                )

            )


            sequence = torch.cat(

                [
                    padding,
                    sequence,
                ],

                dim=0,

            )


        sequence_batch.append(
            sequence
        )


        if len(
            sequence_batch
        ) >= batch_limit:

            flush_batch()


    flush_batch()


    aligned_reliability = np.asarray(

        fused_rows,

        dtype=np.float64,

    )


    aligned_state = np.asarray(

        state_rows,

        dtype=np.int64,

    )


    aligned_probability = np.asarray(

        probability_rows,

        dtype=np.float64,

    )


    full_reliability = np.ones(

        raw_frame_count,

        dtype=np.float64,

    )


    full_state = np.full(

        raw_frame_count,

        2,

        dtype=np.int64,

    )


    full_probability = np.zeros(

        (
            raw_frame_count,
            3,
        ),

        dtype=np.float64,

    )


    full_probability[
        :,
        2
    ] = 1.0


    full_reliability[
        target_frame_ids
    ] = aligned_reliability


    full_state[
        target_frame_ids
    ] = aligned_state


    full_probability[
        target_frame_ids
    ] = aligned_probability


    return (

        full_reliability,

        full_state,

        full_probability,

    )


# ============================================================
# Main
# ============================================================

def main():

    set_seed(
        SEED
    )


    RESULT_DIR.mkdir(

        parents=True,

        exist_ok=True,

    )


    device = choose_device()


    print(
        "="
        *
        60
    )


    print(
        "Predictive Mamba Reliability V2"
    )


    print(
        "Regression + 3-State Classification"
    )


    print(
        f"Horizon: {PREDICTION_HORIZON} frames"
    )


    print(
        "="
        *
        60
    )


    print(
        "Device:",
        device
    )


    print()


    (
        features,
        labels,
        source_frame_ids,
        target_frame_ids,
        raw_frame_count,
    ) = build_predictive_frame_features(

        sequence_path=
            DEFAULT_DATASET,

        gps_path=
            DEFAULT_GPS_PATH,

        label_path=
            DEFAULT_LABEL_PATH,

        feature_window=
            FEATURE_WINDOW,

        prediction_horizon=
            PREDICTION_HORIZON,

    )


    print(
        "Features:",
        features.shape
    )


    print(
        "Labels:",
        labels.shape
    )


    print(
        "Raw frames:",
        raw_frame_count
    )


    split = build_window_split_indices(

        target_frame_ids=
            target_frame_ids,

        sequence_length=
            SEQUENCE_LENGTH,

        stride=
            SEQUENCE_STRIDE,

        block_size=
            SPLIT_BLOCK_SIZE,

    )


    candidate_starts = np.arange(

        0,

        len(
            features
        )
        -
        SEQUENCE_LENGTH
        +
        1,

        SEQUENCE_STRIDE,

        dtype=np.int64,

    )


    train_starts = candidate_starts[
        split[
            "train"
        ]
    ]


    train_row_mask = np.zeros(

        len(
            features
        ),

        dtype=bool,

    )


    for start in train_starts:

        train_row_mask[
            start:
            start
            +
            SEQUENCE_LENGTH
        ] = True


    if not np.any(
        train_row_mask
    ):

        raise RuntimeError(
            "No training rows"
        )


    mean, std = compute_normalization(

        features[
            train_row_mask
        ]

    )


    normalized_features = normalize_features(

        features,

        mean,

        std,

    )


    np.savez(

        NORMALIZATION_PATH,

        mean=mean,

        std=std,

        feature_window=np.asarray(
            [
                FEATURE_WINDOW
            ],
            dtype=np.int64,
        ),

        sequence_length=np.asarray(
            [
                SEQUENCE_LENGTH
            ],
            dtype=np.int64,
        ),

        prediction_horizon=np.asarray(
            [
                PREDICTION_HORIZON
            ],
            dtype=np.int64,
        ),

        bad_threshold=np.asarray(
            [
                BAD_THRESHOLD
            ],
            dtype=np.float32,
        ),

        reliable_threshold=np.asarray(
            [
                RELIABLE_THRESHOLD
            ],
            dtype=np.float32,
        ),

    )


    common_kwargs = {

        "features":
            normalized_features,


        "labels":
            labels,


        "source_frame_ids":
            source_frame_ids,


        "target_frame_ids":
            target_frame_ids,


        "sequence_length":
            SEQUENCE_LENGTH,


        "stride":
            SEQUENCE_STRIDE,

    }


    train_dataset = PredictiveReliabilitySequenceDataset(

        **common_kwargs,

        allowed_window_indices=
            split[
                "train"
            ],

    )


    val_dataset = PredictiveReliabilitySequenceDataset(

        **common_kwargs,

        allowed_window_indices=
            split[
                "val"
            ],

    )


    test_dataset = PredictiveReliabilitySequenceDataset(

        **common_kwargs,

        allowed_window_indices=
            split[
                "test"
            ],

    )


    print()


    print(
        "Train windows:",
        len(
            train_dataset
        )
    )


    print(
        "Validation windows:",
        len(
            val_dataset
        )
    )


    print(
        "Test windows:",
        len(
            test_dataset
        )
    )


    pin_memory = (
        device.type
        ==
        "cuda"
    )


    train_loader = DataLoader(

        train_dataset,

        batch_size=
            BATCH_SIZE,

        shuffle=True,

        num_workers=
            NUM_WORKERS,

        pin_memory=
            pin_memory,

    )


    val_loader = DataLoader(

        val_dataset,

        batch_size=
            BATCH_SIZE,

        shuffle=False,

        num_workers=
            NUM_WORKERS,

        pin_memory=
            pin_memory,

    )


    test_loader = DataLoader(

        test_dataset,

        batch_size=
            BATCH_SIZE,

        shuffle=False,

        num_workers=
            NUM_WORKERS,

        pin_memory=
            pin_memory,

    )


    model = PredictiveMambaReliabilityModel(

        input_dim=51,

        hidden_dim=128,

        state_dim=16,

        num_layers=2,

        head_hidden_dim=64,

        dropout=0.10,

    ).to(
        device
    )


    loss_fn = MultiTaskPredictiveReliabilityLoss(

        classification_weight=1.0,

        regression_weight=0.75,

        safety_weight=0.25,

        class_weights=[
            1.25,
            1.50,
            1.00,
        ],

        bad_threshold=
            BAD_THRESHOLD,

        reliable_threshold=
            RELIABLE_THRESHOLD,

    ).to(
        device
    )


    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=
            LEARNING_RATE,

        weight_decay=
            WEIGHT_DECAY,

    )


    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(

        optimizer,

        mode="min",

        factor=0.5,

        patience=3,

    )


    best_val_loss = math.inf


    best_epoch = -1


    no_improvement = 0


    history = []


    print()


    print(
        "Training..."
    )


    print()


    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        train_metrics = run_epoch(

            model,

            train_loader,

            loss_fn,

            device,

            optimizer=
                optimizer,

        )


        val_metrics = run_epoch(

            model,

            val_loader,

            loss_fn,

            device,

            optimizer=None,

        )


        scheduler.step(
            val_metrics[
                "Loss"
            ]
        )


        learning_rate = float(

            optimizer
            .param_groups[
                0
            ][
                "lr"
            ]

        )


        history.append(
            [
                epoch,

                train_metrics[
                    "Loss"
                ],

                val_metrics[
                    "Loss"
                ],

                val_metrics[
                    "MAE"
                ],

                val_metrics[
                    "RMSE"
                ],

                val_metrics[
                    "Correlation"
                ],

                val_metrics[
                    "StateAccuracy"
                ],

                val_metrics[
                    "BadRecall"
                ],

                val_metrics[
                    "FalseReliableRate"
                ],

                val_metrics[
                    "ReliablePrecision"
                ],

                val_metrics[
                    "ReliableRecall"
                ],

                learning_rate,
            ]
        )


        print(

            f"Epoch {epoch:03d} | "

            f"train {train_metrics['Loss']:.5f} | "

            f"val {val_metrics['Loss']:.5f} | "

            f"MAE {val_metrics['MAE']:.3f} | "

            f"corr {val_metrics['Correlation']:.3f} | "

            f"state acc {val_metrics['StateAccuracy']:.3f} | "

            f"bad recall {val_metrics['BadRecall']:.3f} | "

            f"false reliable {val_metrics['FalseReliableRate']:.3f} | "

            f"good precision {val_metrics['ReliablePrecision']:.3f} | "

            f"good recall {val_metrics['ReliableRecall']:.3f}"

        )


        current_val = float(
            val_metrics[
                "Loss"
            ]
        )


        if (
            current_val
            <
            best_val_loss
        ):

            best_val_loss = current_val


            best_epoch = epoch


            no_improvement = 0


            torch.save(

                {

                    "epoch":
                        epoch,


                    "model_state_dict":
                        model.state_dict(),


                    "optimizer_state_dict":
                        optimizer.state_dict(),


                    "val_metrics":
                        val_metrics,


                    "config": {

                        "feature_window":
                            FEATURE_WINDOW,


                        "sequence_length":
                            SEQUENCE_LENGTH,


                        "prediction_horizon":
                            PREDICTION_HORIZON,


                        "bad_threshold":
                            BAD_THRESHOLD,


                        "reliable_threshold":
                            RELIABLE_THRESHOLD,

                    },

                },

                BEST_MODEL_PATH,

            )


        else:

            no_improvement += 1


        if (
            no_improvement
            >=
            PATIENCE
        ):

            print()


            print(
                "Early stopping."
            )


            break


    torch.save(

        {

            "epoch":
                epoch,


            "model_state_dict":
                model.state_dict(),


            "optimizer_state_dict":
                optimizer.state_dict(),

        },

        LAST_MODEL_PATH,

    )


    np.savetxt(

        HISTORY_PATH,

        np.asarray(
            history,
            dtype=np.float64,
        ),

        fmt="%.8f",

        header=(
            "epoch train_loss val_loss val_mae val_rmse "
            "val_corr state_accuracy bad_recall false_reliable "
            "reliable_precision reliable_recall learning_rate"
        ),

    )


    checkpoint = torch.load(

        BEST_MODEL_PATH,

        map_location=
            device,

        weights_only=False,

    )


    model.load_state_dict(

        checkpoint[
            "model_state_dict"
        ]

    )


    test_metrics = run_epoch(

        model,

        test_loader,

        loss_fn,

        device,

        optimizer=None,

    )


    print()


    print(
        "="
        *
        60
    )


    print(
        "Best epoch:",
        best_epoch
    )


    print(
        "Test metrics"
    )


    for key, value in test_metrics.items():

        if key == "ConfusionMatrix":

            print(
                "ConfusionMatrix:"
            )


            print(
                value
            )


        else:

            print(
                f"{key}: "
                f"{value:.6f}"
            )


    print(
        "="
        *
        60
    )


    with open(

        TEST_METRICS_PATH,

        "w",

        encoding=
            "utf-8",

    ) as file:

        file.write(
            f"best_epoch: "
            f"{best_epoch}\n"
        )


        file.write(
            f"prediction_horizon: "
            f"{PREDICTION_HORIZON}\n"
        )


        for key, value in test_metrics.items():

            if key == "ConfusionMatrix":

                file.write(
                    "ConfusionMatrix:\n"
                )


                file.write(
                    np.array2string(
                        value
                    )
                )


                file.write(
                    "\n"
                )


            else:

                file.write(
                    f"{key}: "
                    f"{value:.8f}\n"
                )


    (
        full_reliability,
        full_state,
        full_probability,
    ) = predict_full_timeline(

        model=

            model,

        normalized_features=

            normalized_features,

        target_frame_ids=

            target_frame_ids,

        raw_frame_count=

            raw_frame_count,

        device=

            device,

    )


    np.savetxt(

        PREDICTION_PATH,

        full_reliability,

        fmt="%.8f",

    )


    np.savetxt(

        STATE_PATH,

        full_state,

        fmt="%d",

    )


    np.savetxt(

        STATE_PROBABILITY_PATH,

        full_probability,

        fmt="%.8f",

    )


    np.savetxt(

        SOURCE_FRAME_PATH,

        source_frame_ids,

        fmt="%d",

    )


    np.savetxt(

        TARGET_FRAME_PATH,

        target_frame_ids,

        fmt="%d",

    )


    print()


    print(
        "Saved:",
        BEST_MODEL_PATH
    )


    print(
        "Saved reliability:",
        PREDICTION_PATH
    )


    print(
        "Saved state:",
        STATE_PATH
    )


    print(
        "Saved probabilities:",
        STATE_PROBABILITY_PATH
    )


if __name__ == "__main__":

    main()
