import os
import sys
import numpy as np

HERE = os.path.dirname(
    os.path.abspath(__file__)
)

if HERE not in sys.path:
    sys.path.insert(
        0,
        HERE,
    )

from src.reliability.feedback_corrector import (
    adaptive_fuse_reliability,
)

from calibrated_fg_common import (
    ROOT,
    load_data,
    calibrate_prediction,
    map_sigma,
    optimize_graph,
    factor_graph_feedback,
)

STAGE1_OUTPUT = os.path.join(
    ROOT,
    "results",
    "calibrated_feedback_stage1",
)

FINAL_OUTPUT = os.path.join(
    ROOT,
    "results",
    "calibrated_predictive_feedback_fg",
)


def main():
    data = load_data(
        require_prediction=True
    )

    raw_prediction = np.clip(
        data["prediction"],
        0.0,
        1.0,
    )

    calibrated = (
        calibrate_prediction(
            raw_prediction
        )
    )

    # ---------------------------------------------------------
    # Stage 1
    # Predictive feed-forward covariance
    # ---------------------------------------------------------

    stage1_sigma = map_sigma(
        calibrated
    )

    print()
    print(
        "Stage 1: predictive "
        "feed-forward optimization"
    )

    stage1_trajectory = (
        optimize_graph(
            gps=data["gps"],
            acceleration=data[
                "acceleration"
            ],
            gyro=data["gyro"],
            sigma=stage1_sigma,
            output_dir=
                STAGE1_OUTPUT,
            title=(
                "Stage 1 "
                "Calibrated Predictive FG"
            ),
            reliability=
                calibrated,
            extra_outputs={
                "raw_prediction.txt":
                    raw_prediction,
                "calibrated_prediction.txt":
                    calibrated,
            },
        )
    )

    # ---------------------------------------------------------
    # Stage 2
    # Factor-graph innovation feedback
    # ---------------------------------------------------------

    feedback, innovation = (
        factor_graph_feedback(
            data["gps"],
            stage1_trajectory,
            scale=8.0,
        )
    )

    final_reliability, (
        prediction_weight
    ) = adaptive_fuse_reliability(
        calibrated,
        feedback,
        base_prediction_weight=0.75,
        min_prediction_weight=0.55,
        max_prediction_weight=0.90,
    )

    final_sigma = map_sigma(
        final_reliability
    )

    print()
    print(
        "Raw prediction mean:",
        float(
            raw_prediction.mean()
        ),
    )

    print(
        "Calibrated mean:",
        float(
            calibrated.mean()
        ),
    )

    print(
        "Feedback mean:",
        float(
            feedback.mean()
        ),
    )

    print(
        "Final reliability mean:",
        float(
            final_reliability.mean()
        ),
    )

    print(
        "Prediction weight "
        "min/max/mean:",
        float(
            prediction_weight.min()
        ),
        float(
            prediction_weight.max()
        ),
        float(
            prediction_weight.mean()
        ),
    )

    print()
    print(
        "Stage 2: feedback-corrected "
        "optimization"
    )

    optimize_graph(
        gps=data["gps"],
        acceleration=data[
            "acceleration"
        ],
        gyro=data["gyro"],
        sigma=final_sigma,
        output_dir=
            FINAL_OUTPUT,
        title=(
            "Calibrated Predictive + "
            "FG Feedback"
        ),
        reliability=
            final_reliability,
        extra_outputs={
            "raw_prediction.txt":
                raw_prediction,
            "calibrated_prediction.txt":
                calibrated,
            "feedback_reliability.txt":
                feedback,
            "fg_innovation.txt":
                innovation,
            "prediction_weight.txt":
                prediction_weight,
            "final_reliability.txt":
                final_reliability,
        },
    )


if __name__ == "__main__":
    main()
