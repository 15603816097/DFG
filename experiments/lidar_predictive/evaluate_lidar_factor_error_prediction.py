from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


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
    error_to_reliability,
    make_windows,
    robust_standardize_apply,
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

ORACLE_PATH = (
    Path(ROOT)
    /
    "results"
    /
    "lidar_physical_oracle_benchmark"
    /
    "lidar_gt_oracle_reliability.npz"
)

OUTPUT_PATH = (
    RESULT_DIR
    /
    "lidar_factor_error_prediction_target_aligned.npz"
)


def corr(
    a,
    b,
):
    a = np.asarray(
        a,
        dtype=np.float64,
    )

    b = np.asarray(
        b,
        dtype=np.float64,
    )

    if (
        np.std(
            a
        )
        <
        1e-12
        or
        np.std(
            b
        )
        <
        1e-12
    ):
        return 0.0

    return float(
        np.corrcoef(
            a,
            b,
        )[
            0,
            1
        ]
    )


def main():
    print(
        "=" * 120
    )

    print(
        "EVALUATE LIDAR FACTOR-ERROR PREDICTION"
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

    te = np.asarray(
        data[
            "translation_error"
        ],
        dtype=np.float32,
    )

    re = np.asarray(
        data[
            "rotation_error_deg"
        ],
        dtype=np.float32,
    )

    norm = np.load(
        NORM_PATH,
        allow_pickle=False,
    )

    horizon = int(
        norm[
            "horizon"
        ][
            0
        ]
    )

    window = int(
        norm[
            "window"
        ][
            0
        ]
    )

    X, Y, target_index = make_windows(
        features,
        te,
        re,
        window=
            window,
        horizon=
            horizon,
    )

    X = robust_standardize_apply(
        X,
        {
            "median":
                norm[
                    "median"
                ],
            "scale":
                norm[
                    "scale"
                ],
        },
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=
            device,
    )

    cfg = LidarFactorErrorModelConfig(
        **checkpoint[
            "config"
        ]
    )

    model = LidarFactorErrorPredictor(
        cfg
    ).to(
        device
    )

    model.load_state_dict(
        checkpoint[
            "model_state"
        ]
    )

    model.eval()

    pred_all = []

    batch = 256

    with torch.no_grad():
        for start in range(
            0,
            len(
                X
            ),
            batch,
        ):
            xb = torch.from_numpy(
                X[
                    start:
                    start + batch
                ]
            ).to(
                device
            )

            pred_all.append(
                model(
                    xb
                ).cpu().numpy()
            )

    prediction = np.concatenate(
        pred_all,
        axis=0,
    )

    oracle = np.load(
        ORACLE_PATH,
        allow_pickle=False,
    )

    trans_normal = float(
        oracle[
            "trans_normal"
        ][
            0
        ]
    )

    trans_severe = float(
        oracle[
            "trans_severe"
        ][
            0
        ]
    )

    rot_normal = float(
        oracle[
            "rot_normal"
        ][
            0
        ]
    )

    rot_severe = float(
        oracle[
            "rot_severe"
        ][
            0
        ]
    )

    predicted_reliability = (
        error_to_reliability(
            prediction[
                :,
                0
            ],
            prediction[
                :,
                1
            ],
            trans_normal=
                trans_normal,
            trans_severe=
                trans_severe,
            rot_normal=
                rot_normal,
            rot_severe=
                rot_severe,
        )
    )

    n_pairs = len(
        features
    )

    # Pair-level default before the first valid prediction:
    # keep LiDAR fully reliable rather than inventing future estimates.
    pair_reliability = np.ones(
        n_pairs,
        dtype=np.float64,
    )

    pair_reliability[
        target_index
    ] = (
        predicted_reliability
    )

    frame_reliability = np.ones(
        n_pairs
        +
        1,
        dtype=np.float64,
    )

    frame_reliability[
        1:
    ] = pair_reliability

    np.savez_compressed(
        OUTPUT_PATH,
        target_index=
            target_index,
        prediction=
            prediction,
        target=
            Y,
        predicted_reliability_windows=
            predicted_reliability,
        lidar_reliability_pair=
            pair_reliability,
        lidar_reliability=
            frame_reliability,
    )

    np.savetxt(
        RESULT_DIR
        /
        "lidar_predictive_prior_target_aligned.txt",
        frame_reliability,
        fmt="%.10f",
    )

    print(
        "Translation MAE:",
        float(
            np.mean(
                np.abs(
                    prediction[
                        :,
                        0
                    ]
                    -
                    Y[
                        :,
                        0
                    ]
                )
            )
        ),
    )

    print(
        "Translation Corr:",
        corr(
            prediction[
                :,
                0
            ],
            Y[
                :,
                0
            ],
        ),
    )

    print(
        "Rotation MAE [deg]:",
        float(
            np.mean(
                np.abs(
                    prediction[
                        :,
                        1
                    ]
                    -
                    Y[
                        :,
                        1
                    ]
                )
            )
        ),
    )

    print(
        "Rotation Corr:",
        corr(
            prediction[
                :,
                1
            ],
            Y[
                :,
                1
            ],
        ),
    )

    oracle_frame = np.asarray(
        oracle[
            "lidar_reliability"
        ],
        dtype=np.float64,
    )

    common = min(
        len(
            frame_reliability
        ),
        len(
            oracle_frame
        ),
    )

    print(
        "Reliability Corr vs GT Oracle:",
        corr(
            frame_reliability[
                :common
            ],
            oracle_frame[
                :common
            ],
        ),
    )

    print(
        "Predictive reliability mean:",
        float(
            np.mean(
                frame_reliability
            )
        ),
    )

    print(
        "GT oracle reliability mean:",
        float(
            np.mean(
                oracle_frame
            )
        ),
    )

    print(
        "Saved:",
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()
