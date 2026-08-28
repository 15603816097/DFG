from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np


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


BENCHMARK_DIR = (
    Path(ROOT)
    /
    "results"
    /
    "lidar_physical_oracle_benchmark"
)

INPUT_PATH = (
    BENCHMARK_DIR
    /
    "physical_lidar_factor_data.npz"
)

OUTPUT_DIR = (
    Path(ROOT)
    /
    "results"
    /
    "lidar_factor_error_predictive_v1"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    /
    "lidar_factor_error_dataset.npz"
)


def safe_diff(
    x,
    order=1,
):
    x = np.asarray(
        x,
        dtype=np.float64,
    )

    result = x.copy()

    for _ in range(order):
        d = np.diff(
            result,
            prepend=result[0],
        )

        result = d

    return result


def rolling_mean(
    x,
    window,
):
    x = np.asarray(
        x,
        dtype=np.float64,
    )

    out = np.zeros_like(
        x
    )

    for i in range(
        len(x)
    ):
        start = max(
            0,
            i - window + 1,
        )

        out[
            i
        ] = np.mean(
            x[
                start:
                i + 1
            ]
        )

    return out


def rolling_std(
    x,
    window,
):
    x = np.asarray(
        x,
        dtype=np.float64,
    )

    out = np.zeros_like(
        x
    )

    for i in range(
        len(x)
    ):
        start = max(
            0,
            i - window + 1,
        )

        out[
            i
        ] = np.std(
            x[
                start:
                i + 1
            ]
        )

    return out


def rotation_angle_deg(
    transforms,
):
    result = []

    for T in transforms:
        c = (
            np.trace(
                T[
                    :3,
                    :3
                ]
            )
            -
            1.0
        ) * 0.5

        result.append(
            np.degrees(
                np.arccos(
                    np.clip(
                        c,
                        -1.0,
                        1.0,
                    )
                )
            )
        )

    return np.asarray(
        result,
        dtype=np.float64,
    )


def main():
    print(
        "=" * 120
    )

    print(
        "BUILD LIDAR FACTOR-ERROR-AWARE PREDICTIVE DATASET"
    )

    print(
        "=" * 120
    )

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            INPUT_PATH
        )

    data = np.load(
        INPUT_PATH,
        allow_pickle=False,
    )

    fitness = np.asarray(
        data[
            "fitness"
        ],
        dtype=np.float64,
    )

    rmse = np.asarray(
        data[
            "rmse"
        ],
        dtype=np.float64,
    )

    corr = np.asarray(
        data[
            "corr"
        ],
        dtype=np.float64,
    )

    quality = np.asarray(
        data[
            "quality"
        ],
        dtype=np.float64,
    )

    body_between = np.asarray(
        data[
            "body_between"
        ],
        dtype=np.float64,
    )

    translation_error = np.asarray(
        data[
            "te"
        ],
        dtype=np.float64,
    )

    rotation_error_deg = np.asarray(
        data[
            "re"
        ],
        dtype=np.float64,
    )

    type_id = np.asarray(
        data[
            "type"
        ],
        dtype=np.float64,
    )

    level_id = np.asarray(
        data[
            "level"
        ],
        dtype=np.float64,
    )

    translation_norm = np.linalg.norm(
        body_between[
            :,
            :3,
            3
        ],
        axis=1,
    )

    rotation_norm_deg = rotation_angle_deg(
        body_between
    )

    feature_dict = {
        "fitness":
            fitness,
        "rmse":
            rmse,
        "corr_norm":
            np.clip(
                corr / 6000.0,
                0.0,
                2.0,
            ),
        "quality":
            quality,

        "translation_norm":
            translation_norm,
        "rotation_norm_deg":
            rotation_norm_deg,

        "d_fitness":
            safe_diff(
                fitness
            ),
        "d_rmse":
            safe_diff(
                rmse
            ),
        "d_corr_norm":
            safe_diff(
                corr / 6000.0
            ),
        "d_quality":
            safe_diff(
                quality
            ),

        "d_translation":
            safe_diff(
                translation_norm
            ),
        "d_rotation":
            safe_diff(
                rotation_norm_deg
            ),

        "dd_translation":
            safe_diff(
                translation_norm,
                order=2,
            ),
        "dd_rotation":
            safe_diff(
                rotation_norm_deg,
                order=2,
            ),

        "fitness_mean_5":
            rolling_mean(
                fitness,
                5,
            ),
        "fitness_std_5":
            rolling_std(
                fitness,
                5,
            ),

        "rmse_mean_5":
            rolling_mean(
                rmse,
                5,
            ),
        "rmse_std_5":
            rolling_std(
                rmse,
                5,
            ),

        "translation_mean_5":
            rolling_mean(
                translation_norm,
                5,
            ),
        "translation_std_5":
            rolling_std(
                translation_norm,
                5,
            ),

        "rotation_mean_5":
            rolling_mean(
                rotation_norm_deg,
                5,
            ),
        "rotation_std_5":
            rolling_std(
                rotation_norm_deg,
                5,
            ),

        "fitness_mean_20":
            rolling_mean(
                fitness,
                20,
            ),
        "rmse_mean_20":
            rolling_mean(
                rmse,
                20,
            ),
        "translation_mean_20":
            rolling_mean(
                translation_norm,
                20,
            ),
        "rotation_mean_20":
            rolling_mean(
                rotation_norm_deg,
                20,
            ),

        # These two are included only as known synthetic-environment descriptors.
        # They must NOT be used for a real deployment model. The training script
        # excludes them by default.
        "degradation_type_id":
            type_id,
        "degradation_level_id":
            level_id,
    }

    deployable_feature_names = [
        name
        for name in feature_dict
        if not name.startswith(
            "degradation_"
        )
    ]

    features = np.column_stack(
        [
            feature_dict[
                name
            ]
            for name in deployable_feature_names
        ]
    )

    features = np.nan_to_num(
        features,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        OUTPUT_PATH,
        features=
            features.astype(
                np.float32
            ),
        feature_names=
            np.asarray(
                deployable_feature_names
            ),
        translation_error=
            translation_error.astype(
                np.float32
            ),
        rotation_error_deg=
            rotation_error_deg.astype(
                np.float32
            ),
        pair_type_id=
            type_id.astype(
                np.int32
            ),
        pair_level_id=
            level_id.astype(
                np.int32
            ),
    )

    metadata = {
        "pairs":
            int(
                len(
                    features
                )
            ),
        "feature_dim":
            int(
                features.shape[
                    1
                ]
            ),
        "feature_names":
            deployable_feature_names,
        "target":
            [
                "future_translation_factor_error_m",
                "future_rotation_factor_error_deg",
            ],
    }

    (
        OUTPUT_DIR
        /
        "dataset_metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "Features:",
        features.shape,
    )

    print(
        "Translation target mean:",
        float(
            np.mean(
                translation_error
            )
        ),
    )

    print(
        "Rotation target mean [deg]:",
        float(
            np.mean(
                rotation_error_deg
            )
        ),
    )

    print(
        "Saved:",
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()
