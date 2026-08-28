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

from calibrated_fg_common import (
    ROOT,
    load_data,
    calibrate_prediction,
    map_sigma,
    optimize_graph,
)

OUTPUT = os.path.join(
    ROOT,
    "results",
    "calibrated_predictive_only_fg",
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

    sigma = map_sigma(
        calibrated
    )

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

    optimize_graph(
        gps=data["gps"],
        acceleration=data[
            "acceleration"
        ],
        gyro=data["gyro"],
        sigma=sigma,
        output_dir=OUTPUT,
        title=(
            "Calibrated Predictive "
            "Reliability Only FG"
        ),
        reliability=calibrated,
        extra_outputs={
            "raw_prediction.txt":
                raw_prediction,
            "calibrated_prediction.txt":
                calibrated,
        },
    )


if __name__ == "__main__":
    main()
