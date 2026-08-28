from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


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


from src.reliability.lidar_factor_error_predictor import (
    LidarFactorErrorModelConfig,
    LidarFactorErrorPredictor,
    make_windows,
    robust_standardize_apply,
    robust_standardize_fit,
)


RESULT_DIR = (
    Path(ROOT)
    /
    "results"
    /
    "lidar_factor_error_predictive_v1"
)

DATASET_PATH = (
    RESULT_DIR
    /
    "lidar_factor_error_dataset.npz"
)

MODEL_PATH = (
    RESULT_DIR
    /
    "best_model.pt"
)

NORM_PATH = (
    RESULT_DIR
    /
    "normalization.npz"
)

HORIZON = 3
WINDOW = 20
BATCH_SIZE = 64
MAX_EPOCHS = 60
PATIENCE = 12
LR = 1e-3


def split_by_target_index(
    target_index,
    n_pairs,
):
    """
    Chronological 70/15/15 split.
    """

    train_end = int(
        round(
            n_pairs
            *
            0.70
        )
    )

    val_end = int(
        round(
            n_pairs
            *
            0.85
        )
    )

    train_mask = (
        target_index
        <
        train_end
    )

    val_mask = (
        (target_index >= train_end)
        &
        (target_index < val_end)
    )

    test_mask = (
        target_index
        >=
        val_end
    )

    return (
        train_mask,
        val_mask,
        test_mask,
    )


def weighted_loss(
    prediction,
    target,
):
    """
    Emphasize rare harmful factors.

    Translation and rotation are normalized by practical scales, then
    high-error targets receive higher weight.
    """

    pred_t = prediction[
        :,
        0
    ]

    pred_r = prediction[
        :,
        1
    ]

    true_t = target[
        :,
        0
    ]

    true_r = target[
        :,
        1
    ]

    t_scale = 0.15
    r_scale = 0.50

    severe = torch.maximum(
        true_t / t_scale,
        true_r / r_scale,
    )

    weight = (
        1.0
        +
        2.0
        *
        torch.clamp(
            severe,
            0.0,
            3.0,
        )
    )

    loss_t = torch.abs(
        pred_t
        -
        true_t
    ) / t_scale

    loss_r = torch.abs(
        pred_r
        -
        true_r
    ) / r_scale

    return torch.mean(
        weight
        *
        (
            loss_t
            +
            loss_r
        )
    )


def evaluate(
    model,
    loader,
    device,
):
    model.eval()

    predictions = []
    targets = []

    with torch.no_grad():
        for X, Y in loader:
            X = X.to(
                device
            )

            Y = Y.to(
                device
            )

            pred = model(
                X
            )

            predictions.append(
                pred.cpu().numpy()
            )

            targets.append(
                Y.cpu().numpy()
            )

    pred = np.concatenate(
        predictions,
        axis=0,
    )

    true = np.concatenate(
        targets,
        axis=0,
    )

    t_mae = float(
        np.mean(
            np.abs(
                pred[
                    :,
                    0
                ]
                -
                true[
                    :,
                    0
                ]
            )
        )
    )

    r_mae = float(
        np.mean(
            np.abs(
                pred[
                    :,
                    1
                ]
                -
                true[
                    :,
                    1
                ]
            )
        )
    )

    t_rmse = float(
        np.sqrt(
            np.mean(
                (
                    pred[
                        :,
                        0
                    ]
                    -
                    true[
                        :,
                        0
                    ]
                )
                **
                2
            )
        )
    )

    r_rmse = float(
        np.sqrt(
            np.mean(
                (
                    pred[
                        :,
                        1
                    ]
                    -
                    true[
                        :,
                        1
                    ]
                )
                **
                2
            )
        )
    )

    return {
        "translation_mae":
            t_mae,
        "translation_rmse":
            t_rmse,
        "rotation_mae_deg":
            r_mae,
        "rotation_rmse_deg":
            r_rmse,
    }


def main():
    print(
        "=" * 120
    )

    print(
        "TRAIN LIDAR FACTOR-ERROR-AWARE PREDICTIVE MODEL"
    )

    print(
        "=" * 120
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else
        "cpu"
    )

    print(
        "Device:",
        device,
    )

    data = np.load(
        DATASET_PATH,
        allow_pickle=True,
    )

    features = np.asarray(
        data[
            "features"
        ],
        dtype=np.float32,
    )

    translation_error = np.asarray(
        data[
            "translation_error"
        ],
        dtype=np.float32,
    )

    rotation_error = np.asarray(
        data[
            "rotation_error_deg"
        ],
        dtype=np.float32,
    )

    X, Y, target_index = make_windows(
        features,
        translation_error,
        rotation_error,
        window=
            WINDOW,
        horizon=
            HORIZON,
    )

    train_mask, val_mask, test_mask = (
        split_by_target_index(
            target_index,
            len(
                features
            ),
        )
    )

    X_train = X[
        train_mask
    ]

    Y_train = Y[
        train_mask
    ]

    X_val = X[
        val_mask
    ]

    Y_val = Y[
        val_mask
    ]

    X_test = X[
        test_mask
    ]

    Y_test = Y[
        test_mask
    ]

    test_target_index = target_index[
        test_mask
    ]

    norm = robust_standardize_fit(
        X_train
    )

    X_train = robust_standardize_apply(
        X_train,
        norm,
    )

    X_val = robust_standardize_apply(
        X_val,
        norm,
    )

    X_test = robust_standardize_apply(
        X_test,
        norm,
    )

    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(
                X_train
            ),
            torch.from_numpy(
                Y_train
            ),
        ),
        batch_size=
            BATCH_SIZE,
        shuffle=True,
    )

    val_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(
                X_val
            ),
            torch.from_numpy(
                Y_val
            ),
        ),
        batch_size=
            BATCH_SIZE,
        shuffle=False,
    )

    test_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(
                X_test
            ),
            torch.from_numpy(
                Y_test
            ),
        ),
        batch_size=
            BATCH_SIZE,
        shuffle=False,
    )

    config = LidarFactorErrorModelConfig(
        input_dim=
            X_train.shape[
                -1
            ],
        hidden_dim=
            64,
        num_layers=
            2,
        dropout=
            0.10,
        horizon=
            HORIZON,
        window=
            WINDOW,
    )

    model = LidarFactorErrorPredictor(
        config
    ).to(
        device
    )

    print(
        "Temporal backend:",
        model.backend_name,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=1e-4,
    )

    best_val = float(
        "inf"
    )

    best_epoch = 0
    wait = 0

    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):
        model.train()

        train_losses = []

        for Xb, Yb in train_loader:
            Xb = Xb.to(
                device
            )

            Yb = Yb.to(
                device
            )

            optimizer.zero_grad()

            pred = model(
                Xb
            )

            loss = weighted_loss(
                pred,
                Yb,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                5.0,
            )

            optimizer.step()

            train_losses.append(
                float(
                    loss.detach().cpu()
                )
            )

        val_metric = evaluate(
            model,
            val_loader,
            device,
        )

        val_score = (
            val_metric[
                "translation_mae"
            ]
            /
            0.15
            +
            val_metric[
                "rotation_mae_deg"
            ]
            /
            0.50
        )

        print(
            f"Epoch {epoch:03d} | "
            f"train {np.mean(train_losses):.6f} | "
            f"val score {val_score:.6f} | "
            f"tMAE {val_metric['translation_mae']:.4f} | "
            f"rMAE {val_metric['rotation_mae_deg']:.4f}"
        )

        if val_score < best_val:
            best_val = val_score
            best_epoch = epoch
            wait = 0

            torch.save(
                {
                    "model_state":
                        model.state_dict(),
                    "config":
                        config.__dict__,
                    "backend":
                        model.backend_name,
                },
                MODEL_PATH,
            )

        else:
            wait += 1

        if wait >= PATIENCE:
            print(
                "Early stopping."
            )

            break

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=
            device,
    )

    model.load_state_dict(
        checkpoint[
            "model_state"
        ]
    )

    test_metric = evaluate(
        model,
        test_loader,
        device,
    )

    print()
    print(
        "=" * 120
    )

    print(
        "BEST EPOCH:",
        best_epoch,
    )

    for key, value in test_metric.items():
        print(
            f"{key:24s}: "
            f"{value:.6f}"
        )

    print(
        "=" * 120
    )

    np.savez_compressed(
        NORM_PATH,
        median=
            norm[
                "median"
            ],
        scale=
            norm[
                "scale"
            ],
        horizon=
            np.asarray(
                [HORIZON],
                dtype=np.int32,
            ),
        window=
            np.asarray(
                [WINDOW],
                dtype=np.int32,
            ),
    )

    np.savez_compressed(
        RESULT_DIR
        /
        "test_split_info.npz",
        target_index=
            test_target_index,
        true_targets=
            Y_test,
    )

    (
        RESULT_DIR
        /
        "training_summary.json"
    ).write_text(
        json.dumps(
            {
                "best_epoch":
                    best_epoch,
                "backend":
                    model.backend_name,
                "test_metrics":
                    test_metric,
                "train_windows":
                    int(
                        len(
                            X_train
                        )
                    ),
                "val_windows":
                    int(
                        len(
                            X_val
                        )
                    ),
                "test_windows":
                    int(
                        len(
                            X_test
                        )
                    ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "Saved:",
        MODEL_PATH,
    )


if __name__ == "__main__":
    main()
