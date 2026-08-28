"""
Train the same Shared-Mamba model on balanced V2 data.

Important
---------
Model architecture is unchanged.

Only:
    data plan
    training-data path
    evaluation discipline

are changed.
"""

from __future__ import annotations

import os
import sys
import random
import csv

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


from src.dataset.multisensor_predictive_dataset import (
    SENSORS,
    load_multisensor_npz,
    compute_normalization,
    apply_normalization,
    MultiSensorPredictiveDataset,
)

from src.model.multisensor_predictive_reliability_model import (
    MultiSensorPredictiveReliabilityModel,
)


DATA_PATH = os.path.join(
    ROOT,
    "results",
    "multisensor_reliability_v2",
    "multisensor_reliability_data_v2.npz",
)

OUTPUT_DIR = os.path.join(
    ROOT,
    "results",
    "multisensor_predictive_reliability_v2",
)


SEED = 20260826
SEQUENCE_LENGTH = 64
BATCH_SIZE = 64
EPOCHS = 100
PATIENCE = 15
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4

CURRENT_LOSS_WEIGHT = 0.35
FUTURE_LOSS_WEIGHT = 1.00

# Stronger emphasis on unreliable frames than V1.
LOW_RELIABILITY_EMPHASIS = 3.0


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


def metrics(
    y,
    p,
):
    y = np.asarray(
        y,
        dtype=np.float64,
    )

    p = np.asarray(
        p,
        dtype=np.float64,
    )

    mae = float(
        np.mean(
            np.abs(
                y
                -
                p
            )
        )
    )

    rmse = float(
        np.sqrt(
            np.mean(
                (
                    y
                    -
                    p
                )
                ** 2
            )
        )
    )

    if (
        np.std(y) > 1e-12
        and
        np.std(p) > 1e-12
    ):
        corr = float(
            np.corrcoef(
                y,
                p,
            )[
                0,
                1
            ]
        )
    else:
        corr = 0.0

    acc = float(
        np.mean(
            (
                p
                >=
                0.5
            )
            ==
            (
                y
                >=
                0.5
            )
        )
    )

    bad_mask = (
        y
        <
        0.5
    )

    if np.any(
        bad_mask
    ):
        bad_recall = float(
            np.mean(
                p[
                    bad_mask
                ]
                <
                0.5
            )
        )
    else:
        bad_recall = 1.0

    severe_mask = (
        y
        <
        0.2
    )

    if np.any(
        severe_mask
    ):
        severe_recall = float(
            np.mean(
                p[
                    severe_mask
                ]
                <
                0.2
            )
        )
    else:
        severe_recall = 1.0

    return {
        "MAE":
            mae,

        "RMSE":
            rmse,

        "Correlation":
            corr,

        "BinaryAccuracy@0.5":
            acc,

        "BadRecall@0.5":
            bad_recall,

        "SevereRecall@0.2":
            severe_recall,
    }


def to_device(
    batch,
    device,
):
    return {
        sensor:
            batch[
                f"{sensor}_feature"
            ].to(
                device
            )
        for sensor in SENSORS
    }


@torch.no_grad()
def collect(
    model,
    loader,
    device,
):
    model.eval()

    ids = []

    output = {
        sensor:
            {
                key:
                    []
                for key in (
                    "current_y",
                    "current_p",
                    "future_y",
                    "future_p",
                )
            }
        for sensor in SENSORS
    }

    for batch in loader:
        x = to_device(
            batch,
            device,
        )

        pred = model(
            x[
                "gps"
            ],
            x[
                "imu"
            ],
            x[
                "lidar"
            ],
            x[
                "camera"
            ],
        )

        ids.extend(
            batch[
                "frame_id"
            ].numpy().tolist()
        )

        for sensor in SENSORS:
            output[
                sensor
            ][
                "current_y"
            ].extend(
                batch[
                    f"{sensor}_current_label"
                ].numpy().tolist()
            )

            output[
                sensor
            ][
                "current_p"
            ].extend(
                pred[
                    sensor
                ][
                    "current"
                ]
                .cpu()
                .numpy()
                .tolist()
            )

            output[
                sensor
            ][
                "future_y"
            ].extend(
                batch[
                    f"{sensor}_future_label"
                ].numpy().tolist()
            )

            output[
                sensor
            ][
                "future_p"
            ].extend(
                pred[
                    sensor
                ][
                    "future"
                ]
                .cpu()
                .numpy()
                .tolist()
            )

    ids = np.asarray(
        ids,
        dtype=np.int64,
    )

    for sensor in SENSORS:
        for key in output[
            sensor
        ]:
            output[
                sensor
            ][
                key
            ] = np.asarray(
                output[
                    sensor
                ][
                    key
                ],
                dtype=np.float64,
            )

    return ids, output


def main():
    set_seed()

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    raw = load_multisensor_npz(
        DATA_PATH
    )

    features = raw[
        "features"
    ]

    current = raw[
        "current_labels"
    ]

    future = raw[
        "future_labels"
    ]

    horizon = int(
        raw[
            "horizon"
        ]
    )

    n = int(
        raw[
            "length"
        ]
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

    normalization = (
        compute_normalization(
            features,
            train_end,
        )
    )

    normalized = (
        apply_normalization(
            features,
            normalization,
        )
    )

    valid = np.arange(
        SEQUENCE_LENGTH - 1,
        n,
        dtype=np.int64,
    )

    train_ids = valid[
        valid
        <
        train_end
    ]

    val_ids = valid[
        (
            valid
            >=
            train_end
        )
        &
        (
            valid
            <
            val_end
        )
    ]

    test_ids = valid[
        valid
        >=
        val_end
    ]

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else
        "cpu"
    )

    print("=" * 96)
    print(
        "BALANCED MULTI-SENSOR "
        "PREDICTIVE RELIABILITY V2"
    )
    print("=" * 96)

    print(
        "Device:",
        device,
    )

    print(
        "Frames:",
        n,
    )

    print(
        "Horizon:",
        horizon,
    )

    print(
        "Train/Val/Test:",
        len(
            train_ids
        ),
        len(
            val_ids
        ),
        len(
            test_ids
        ),
    )

    def make_loader(
        ids,
        shuffle,
    ):
        return DataLoader(
            MultiSensorPredictiveDataset(
                normalized,
                current,
                future,
                ids,
                sequence_length=
                    SEQUENCE_LENGTH,
            ),
            batch_size=
                BATCH_SIZE,
            shuffle=
                shuffle,
        )

    train_loader = (
        make_loader(
            train_ids,
            True,
        )
    )

    val_loader = (
        make_loader(
            val_ids,
            False,
        )
    )

    test_loader = (
        make_loader(
            test_ids,
            False,
        )
    )

    model = (
        MultiSensorPredictiveReliabilityModel(
            gps_dim=
                normalized[
                    "gps"
                ].shape[1],

            imu_dim=
                normalized[
                    "imu"
                ].shape[1],

            lidar_dim=
                normalized[
                    "lidar"
                ].shape[1],

            camera_dim=
                normalized[
                    "camera"
                ].shape[1],

            sensor_embed_dim=32,
            hidden_dim=128,
            num_layers=2,
            dropout=0.10,
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

        losses = []

        for batch in train_loader:
            x = to_device(
                batch,
                device,
            )

            pred = model(
                x[
                    "gps"
                ],
                x[
                    "imu"
                ],
                x[
                    "lidar"
                ],
                x[
                    "camera"
                ],
            )

            total = torch.zeros(
                (),
                device=device,
            )

            for sensor in SENSORS:
                yc = batch[
                    f"{sensor}_current_label"
                ].to(
                    device
                )

                yf = batch[
                    f"{sensor}_future_label"
                ].to(
                    device
                )

                current_loss = (
                    torch.mean(
                        (
                            pred[
                                sensor
                            ][
                                "current"
                            ]
                            -
                            yc
                        )
                        ** 2
                    )
                )

                # Strongly weight poor reliability regions.
                weight = (
                    1.0
                    +
                    LOW_RELIABILITY_EMPHASIS
                    *
                    (
                        1.0
                        -
                        yf
                    )
                    ** 2
                )

                future_loss = (
                    torch.mean(
                        weight
                        *
                        (
                            pred[
                                sensor
                            ][
                                "future"
                            ]
                            -
                            yf
                        )
                        ** 2
                    )
                )

                total = (
                    total
                    +
                    CURRENT_LOSS_WEIGHT
                    *
                    current_loss
                    +
                    FUTURE_LOSS_WEIGHT
                    *
                    future_loss
                )

            total = (
                total
                /
                len(
                    SENSORS
                )
            )

            optimizer.zero_grad()

            total.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                5.0,
            )

            optimizer.step()

            losses.append(
                float(
                    total.item()
                )
            )

        _, val_out = collect(
            model,
            val_loader,
            device,
        )

        val_mse = []

        text = []

        for sensor in SENSORS:
            y = val_out[
                sensor
            ][
                "future_y"
            ]

            p = val_out[
                sensor
            ][
                "future_p"
            ]

            val_mse.append(
                float(
                    np.mean(
                        (
                            y
                            -
                            p
                        )
                        ** 2
                    )
                )
            )

            m = metrics(
                y,
                p,
            )

            text.append(
                f"{sensor}:"
                f"{m['Correlation']:.3f}/"
                f"{m['BadRecall@0.5']:.2f}"
            )

        score = float(
            np.mean(
                val_mse
            )
        )

        print(
            f"Epoch {epoch:03d} | "
            f"train {np.mean(losses):.6f} | "
            f"val {score:.6f} | "
            +
            " ".join(
                text
            )
        )

        if score < best_val:
            best_val = score
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
            map_location=device,
        )
    )

    test_frame_ids, test_out = (
        collect(
            model,
            test_loader,
            device,
        )
    )

    print()
    print("=" * 96)
    print(
        "BEST EPOCH:",
        best_epoch,
    )
    print("=" * 96)

    rows = []

    for sensor in SENSORS:
        cm = metrics(
            test_out[
                sensor
            ][
                "current_y"
            ],
            test_out[
                sensor
            ][
                "current_p"
            ],
        )

        fm = metrics(
            test_out[
                sensor
            ][
                "future_y"
            ],
            test_out[
                sensor
            ][
                "future_p"
            ],
        )

        print()
        print(
            sensor.upper()
        )

        print(
            "Current:",
            cm,
        )

        print(
            "Future :",
            fm,
        )

        rows.append(
            {
                "sensor":
                    sensor,

                "head":
                    "current",

                **cm,
            }
        )

        rows.append(
            {
                "sensor":
                    sensor,

                "head":
                    "future",

                **fm,
            }
        )

    with open(
        os.path.join(
            OUTPUT_DIR,
            "test_metrics.csv",
        ),
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "sensor",
                "head",
                "MAE",
                "RMSE",
                "Correlation",
                "BinaryAccuracy@0.5",
                "BadRecall@0.5",
                "SevereRecall@0.2",
            ],
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

    # Save normalization.
    norm_kwargs = {
        "sequence_length":
            np.asarray(
                [
                    SEQUENCE_LENGTH
                ],
                dtype=np.int64,
            ),

        "horizon":
            np.asarray(
                [
                    horizon
                ],
                dtype=np.int64,
            ),
    }

    for sensor in SENSORS:
        norm_kwargs[
            f"{sensor}_mean"
        ] = normalization[
            sensor
        ][
            "mean"
        ]

        norm_kwargs[
            f"{sensor}_std"
        ] = normalization[
            sensor
        ][
            "std"
        ]

    np.savez(
        os.path.join(
            OUTPUT_DIR,
            "normalization.npz",
        ),
        **norm_kwargs,
    )

    # Full sequence inference is saved for later FG use,
    # but paper metrics must use test metrics above.
    all_loader = (
        make_loader(
            valid,
            False,
        )
    )

    all_ids, all_out = (
        collect(
            model,
            all_loader,
            device,
        )
    )

    for sensor in SENSORS:
        current_full = np.ones(
            n,
            dtype=np.float64,
        )

        source_future = np.ones(
            n,
            dtype=np.float64,
        )

        target_aligned = np.ones(
            n,
            dtype=np.float64,
        )

        source_frame = np.full(
            n,
            -1,
            dtype=np.int64,
        )

        current_full[
            all_ids
        ] = all_out[
            sensor
        ][
            "current_p"
        ]

        source_future[
            all_ids
        ] = all_out[
            sensor
        ][
            "future_p"
        ]

        for source, pred in zip(
            all_ids,
            all_out[
                sensor
            ][
                "future_p"
            ],
        ):
            target = (
                int(
                    source
                )
                +
                horizon
            )

            if target < n:
                target_aligned[
                    target
                ] = float(
                    pred
                )

                source_frame[
                    target
                ] = int(
                    source
                )

        for i in range(n):
            if source_frame[
                i
            ] < 0:
                target_aligned[
                    i
                ] = current_full[
                    i
                ]

        np.savetxt(
            os.path.join(
                OUTPUT_DIR,
                f"{sensor}_predictive_prior_target_aligned.txt",
            ),
            target_aligned,
            fmt="%.8f",
        )

        np.savetxt(
            os.path.join(
                OUTPUT_DIR,
                f"{sensor}_prediction_source_frame.txt",
            ),
            source_frame,
            fmt="%d",
        )

        np.savetxt(
            os.path.join(
                OUTPUT_DIR,
                f"{sensor}_current_prediction.txt",
            ),
            current_full,
            fmt="%.8f",
        )

        np.savetxt(
            os.path.join(
                OUTPUT_DIR,
                f"{sensor}_future_prediction_source_aligned.txt",
            ),
            source_future,
            fmt="%.8f",
        )

    np.savetxt(
        os.path.join(
            OUTPUT_DIR,
            "test_frame_ids.txt",
        ),
        test_frame_ids,
        fmt="%d",
    )

    print()
    print(
        "Saved:",
        OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
