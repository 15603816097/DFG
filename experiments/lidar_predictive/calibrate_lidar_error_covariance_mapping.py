from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = (
    ROOT
    /
    "results"
)

PREDICTION_PATH = (
    RESULTS_DIR
    /
    "lidar_factor_error_predictive_v1"
    /
    "lidar_factor_error_prediction_target_aligned.npz"
)

BEST_SIGMA_PATH = (
    RESULTS_DIR
    /
    "lidar_covariance_calibration_v1"
    /
    "best_fixed_sigma.txt"
)

OUTPUT_DIR = (
    RESULTS_DIR
    /
    "lidar_covariance_calibration_v1"
    /
    "mapping"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    /
    "lidar_dynamic_mapping.npz"
)

CONFIG_PATH = (
    OUTPUT_DIR
    /
    "mapping_config.json"
)


def read_best_sigma() -> float:
    """
    Read empirically selected LiDAR fixed baseline sigma.

    File should contain only one float, e.g.:

        0.15
    """

    if not BEST_SIGMA_PATH.exists():
        raise FileNotFoundError(
            "\n"
            "best_fixed_sigma.txt 不存在。\n"
            "你现在还没有完成 fixed covariance sweep。\n\n"
            f"Expected:\n{BEST_SIGMA_PATH}\n\n"
            "请先确定最优 LiDAR fixed sigma，"
            "然后把一个数字写入该文件，例如：\n"
            "0.15\n"
        )

    text = (
        BEST_SIGMA_PATH
        .read_text(
            encoding="utf-8",
        )
        .strip()
    )

    try:
        sigma = float(
            text
        )

    except ValueError as exc:
        raise ValueError(
            "\n"
            "best_fixed_sigma.txt 应该只包含一个数字，例如：\n"
            "0.15\n"
        ) from exc

    if sigma <= 0:
        raise ValueError(
            f"Invalid sigma: {sigma}"
        )

    return sigma


def percentile_score(
    value: np.ndarray,
    p50: float,
    p90: float,
) -> np.ndarray:
    """
    Map predicted factor error to [0, 1].

    score = 0:
        normal / below median prediction.

    score = 1:
        predicted factor error reaches p90 or higher.
    """

    value = np.asarray(
        value,
        dtype=np.float64,
    )

    denominator = max(
        float(
            p90
            -
            p50
        ),
        1e-9,
    )

    score = (
        value
        -
        float(
            p50
        )
    ) / denominator

    return np.clip(
        score,
        0.0,
        1.0,
    )


def main():

    print(
        "=" * 120
    )

    print(
        "CALIBRATE LIDAR FACTOR-ERROR -> DYNAMIC COVARIANCE"
    )

    print(
        "=" * 120
    )

    # ============================================================
    # 1. Load predicted factor errors
    # ============================================================

    if not PREDICTION_PATH.exists():
        raise FileNotFoundError(
            PREDICTION_PATH
        )

    data = np.load(
        PREDICTION_PATH,
        allow_pickle=False,
    )

    print()
    print(
        "Prediction file:"
    )
    print(
        PREDICTION_PATH
    )

    print()
    print(
        "Available arrays:"
    )

    for key in data.files:
        print(
            f"  {key:32s}",
            data[
                key
            ].shape,
        )

    if "prediction" not in data.files:
        raise KeyError(
            "prediction array not found."
        )

    prediction = np.asarray(
        data[
            "prediction"
        ],
        dtype=np.float64,
    )

    if (
        prediction.ndim != 2
        or
        prediction.shape[
            1
        ] < 2
    ):
        raise ValueError(
            "prediction should have shape (N, 2). "
            f"Actual: {prediction.shape}"
        )

    predicted_translation_error = (
        prediction[
            :,
            0
        ]
    )

    predicted_rotation_error_deg = (
        prediction[
            :,
            1
        ]
    )

    target_index = np.asarray(
        data[
            "target_index"
        ],
        dtype=np.int64,
    )

    if len(
        target_index
    ) != len(
        prediction
    ):
        raise ValueError(
            "target_index / prediction length mismatch."
        )

    # ============================================================
    # 2. Read best fixed covariance baseline
    # ============================================================

    sigma0_translation = (
        read_best_sigma()
    )

    # Keep the existing project's translation / rotation ratio.
    sigma0_rotation = (
        sigma0_translation
        *
        (
            0.03
            /
            0.35
        )
    )

    print()
    print(
        "Best fixed LiDAR sigma:"
    )

    print(
        f"  translation: "
        f"{sigma0_translation:.6f}"
    )

    print(
        f"  rotation   : "
        f"{sigma0_rotation:.6f}"
    )

    # ============================================================
    # 3. Derive error calibration thresholds
    #
    # Important:
    # These thresholds come from PREDICTED factor errors,
    # not GT factor errors.
    #
    # This prevents the previous reliability=1 saturation.
    # ============================================================

    finite_t = (
        predicted_translation_error[
            np.isfinite(
                predicted_translation_error
            )
        ]
    )

    finite_r = (
        predicted_rotation_error_deg[
            np.isfinite(
                predicted_rotation_error_deg
            )
        ]
    )

    if len(
        finite_t
    ) == 0:
        raise ValueError(
            "No finite translation predictions."
        )

    if len(
        finite_r
    ) == 0:
        raise ValueError(
            "No finite rotation predictions."
        )

    t_p50 = float(
        np.percentile(
            finite_t,
            50,
        )
    )

    t_p90 = float(
        np.percentile(
            finite_t,
            90,
        )
    )

    r_p50 = float(
        np.percentile(
            finite_r,
            50,
        )
    )

    r_p90 = float(
        np.percentile(
            finite_r,
            90,
        )
    )

    print()
    print(
        "Prediction calibration thresholds:"
    )

    print(
        f"  translation p50/p90: "
        f"{t_p50:.6f} / "
        f"{t_p90:.6f} m"
    )

    print(
        f"  rotation    p50/p90: "
        f"{r_p50:.6f} / "
        f"{r_p90:.6f} deg"
    )

    # ============================================================
    # 4. Convert translation / rotation predictions to risk score
    # ============================================================

    translation_score = (
        percentile_score(
            predicted_translation_error,
            t_p50,
            t_p90,
        )
    )

    rotation_score = (
        percentile_score(
            predicted_rotation_error_deg,
            r_p50,
            r_p90,
        )
    )

    # LiDAR factor becomes risky if either translation OR rotation
    # is predicted unreliable.
    risk_score = np.maximum(
        translation_score,
        rotation_score,
    )

    # ============================================================
    # 5. Conservative dynamic covariance mapping
    #
    # sigma = sigma0 * [1 + (max_scale-1) * risk^gamma]
    #
    # Normal:
    #   sigma ~= sigma0
    #
    # High predicted factor error:
    #   sigma increases progressively.
    # ============================================================

    MAX_SCALE = 3.0

    GAMMA = 2.0

    covariance_scale = (
        1.0
        +
        (
            MAX_SCALE
            -
            1.0
        )
        *
        (
            risk_score
            **
            GAMMA
        )
    )

    predicted_translation_sigma_windows = (
        sigma0_translation
        *
        covariance_scale
    )

    predicted_rotation_sigma_windows = (
        sigma0_rotation
        *
        covariance_scale
    )

    # Reliability here is diagnostic only.
    predicted_reliability_windows = (
        1.0
        /
        covariance_scale
    )

    # ============================================================
    # 6. Convert window-target aligned result to complete pair array
    #
    # Existing model:
    #   prediction[k] belongs to factor target_index[k].
    #
    # For factors before the first valid target:
    #   use sigma0, i.e. normal fixed LiDAR confidence.
    # ============================================================

    if "lidar_reliability_pair" in data.files:

        pair_count = len(
            data[
                "lidar_reliability_pair"
            ]
        )

    else:

        pair_count = int(
            np.max(
                target_index
            )
            +
            1
        )

    translation_sigma_pair = np.full(
        pair_count,
        sigma0_translation,
        dtype=np.float64,
    )

    rotation_sigma_pair = np.full(
        pair_count,
        sigma0_rotation,
        dtype=np.float64,
    )

    reliability_pair = np.ones(
        pair_count,
        dtype=np.float64,
    )

    risk_score_pair = np.zeros(
        pair_count,
        dtype=np.float64,
    )

    valid_index_mask = (
        (target_index >= 0)
        &
        (target_index < pair_count)
    )

    valid_target_index = (
        target_index[
            valid_index_mask
        ]
    )

    translation_sigma_pair[
        valid_target_index
    ] = (
        predicted_translation_sigma_windows[
            valid_index_mask
        ]
    )

    rotation_sigma_pair[
        valid_target_index
    ] = (
        predicted_rotation_sigma_windows[
            valid_index_mask
        ]
    )

    reliability_pair[
        valid_target_index
    ] = (
        predicted_reliability_windows[
            valid_index_mask
        ]
    )

    risk_score_pair[
        valid_target_index
    ] = (
        risk_score[
            valid_index_mask
        ]
    )

    # ============================================================
    # 7. Frame-aligned arrays
    #
    # LiDAR relative factor i:
    #     frame i -> frame i+1
    #
    # target-frame reliability convention:
    #     frame index = pair index + 1
    # ============================================================

    frame_count = (
        pair_count
        +
        1
    )

    translation_sigma_frame = np.full(
        frame_count,
        sigma0_translation,
        dtype=np.float64,
    )

    rotation_sigma_frame = np.full(
        frame_count,
        sigma0_rotation,
        dtype=np.float64,
    )

    reliability_frame = np.ones(
        frame_count,
        dtype=np.float64,
    )

    risk_score_frame = np.zeros(
        frame_count,
        dtype=np.float64,
    )

    translation_sigma_frame[
        1:
    ] = translation_sigma_pair

    rotation_sigma_frame[
        1:
    ] = rotation_sigma_pair

    reliability_frame[
        1:
    ] = reliability_pair

    risk_score_frame[
        1:
    ] = risk_score_pair

    # ============================================================
    # 8. Save
    # ============================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        OUTPUT_PATH,

        target_index=
            target_index,

        predicted_translation_error=
            predicted_translation_error,

        predicted_rotation_error_deg=
            predicted_rotation_error_deg,

        translation_score_windows=
            translation_score,

        rotation_score_windows=
            rotation_score,

        risk_score_windows=
            risk_score,

        predicted_reliability_windows=
            predicted_reliability_windows,

        translation_sigma_windows=
            predicted_translation_sigma_windows,

        rotation_sigma_windows=
            predicted_rotation_sigma_windows,

        risk_score_pair=
            risk_score_pair,

        lidar_reliability_pair=
            reliability_pair,

        translation_sigma_pair=
            translation_sigma_pair,

        rotation_sigma_pair=
            rotation_sigma_pair,

        risk_score_frame=
            risk_score_frame,

        lidar_reliability=
            reliability_frame,

        translation_sigma=
            translation_sigma_frame,

        rotation_sigma=
            rotation_sigma_frame,
    )

    np.savetxt(
        OUTPUT_DIR
        /
        "lidar_translation_sigma_pair.txt",
        translation_sigma_pair,
        fmt="%.10f",
    )

    np.savetxt(
        OUTPUT_DIR
        /
        "lidar_rotation_sigma_pair.txt",
        rotation_sigma_pair,
        fmt="%.10f",
    )

    np.savetxt(
        OUTPUT_DIR
        /
        "lidar_reliability_pair.txt",
        reliability_pair,
        fmt="%.10f",
    )

    config = {
        "sigma0_translation":
            float(
                sigma0_translation
            ),

        "sigma0_rotation":
            float(
                sigma0_rotation
            ),

        "translation_prediction_p50":
            float(
                t_p50
            ),

        "translation_prediction_p90":
            float(
                t_p90
            ),

        "rotation_prediction_p50_deg":
            float(
                r_p50
            ),

        "rotation_prediction_p90_deg":
            float(
                r_p90
            ),

        "max_scale":
            float(
                MAX_SCALE
            ),

        "gamma":
            float(
                GAMMA
            ),

        "pair_count":
            int(
                pair_count
            ),

        "frame_count":
            int(
                frame_count
            ),
    }

    CONFIG_PATH.write_text(
        json.dumps(
            config,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ============================================================
    # 9. Diagnostics
    # ============================================================

    dynamic_pair_mask = (
        risk_score_pair
        >
        1e-8
    )

    high_risk_mask = (
        risk_score_pair
        >=
        0.5
    )

    severe_risk_mask = (
        risk_score_pair
        >=
        0.9
    )

    print()
    print(
        "=" * 120
    )

    print(
        "CALIBRATED DYNAMIC MAPPING STATISTICS"
    )

    print(
        "=" * 120
    )

    print(
        "Risk score min/max/mean:"
    )

    print(
        f"  "
        f"{risk_score_pair.min():.6f} / "
        f"{risk_score_pair.max():.6f} / "
        f"{risk_score_pair.mean():.6f}"
    )

    print()
    print(
        "Reliability min/max/mean:"
    )

    print(
        f"  "
        f"{reliability_pair.min():.6f} / "
        f"{reliability_pair.max():.6f} / "
        f"{reliability_pair.mean():.6f}"
    )

    print()
    print(
        "Translation sigma min/max/mean:"
    )

    print(
        f"  "
        f"{translation_sigma_pair.min():.6f} / "
        f"{translation_sigma_pair.max():.6f} / "
        f"{translation_sigma_pair.mean():.6f}"
    )

    print()
    print(
        "Rotation sigma min/max/mean:"
    )

    print(
        f"  "
        f"{rotation_sigma_pair.min():.6f} / "
        f"{rotation_sigma_pair.max():.6f} / "
        f"{rotation_sigma_pair.mean():.6f}"
    )

    print()
    print(
        "Dynamic factors:",
        int(
            np.sum(
                dynamic_pair_mask
            )
        ),
        "/",
        pair_count,
    )

    print(
        "High-risk factors (score >= 0.5):",
        int(
            np.sum(
                high_risk_mask
            )
        ),
    )

    print(
        "Severe-risk factors (score >= 0.9):",
        int(
            np.sum(
                severe_risk_mask
            )
        ),
    )

    print()
    print(
        "Saved:"
    )

    print(
        OUTPUT_PATH
    )

    print()
    print(
        "IMPORTANT:"
    )

    if np.ptp(
        translation_sigma_pair
    ) < 1e-10:

        print(
            "WARNING: LiDAR covariance is still constant."
        )

    else:

        print(
            "SUCCESS: LiDAR covariance is genuinely dynamic."
        )


if __name__ == "__main__":
    main()
