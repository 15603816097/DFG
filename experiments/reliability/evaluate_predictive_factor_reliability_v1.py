from __future__ import annotations

import os
import sys

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


SENSORS = (
    "gps",
    "imu",
    "lidar",
    "camera",
)


ORACLE_PATH = os.path.join(
    ROOT,
    "results",
    "oracle_factor_reliability",
    "oracle_factor_reliability.npz",
)

PRED_DIR = os.path.join(
    ROOT,
    "results",
    "predictive_factor_reliability_v1",
)

HORIZON = 3
SEQUENCE_LENGTH = 64


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
        np.std(
            y
        )
        >
        1e-12
        and
        np.std(
            p
        )
        >
        1e-12
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

    bad = (
        y
        <
        0.5
    )

    bad_recall = (
        float(
            np.mean(
                p[
                    bad
                ]
                <
                0.5
            )
        )
        if
        np.any(
            bad
        )
        else
        1.0
    )

    return (
        mae,
        rmse,
        corr,
        bad_recall,
    )


def main():
    oracle = np.load(
        ORACLE_PATH,
        allow_pickle=False,
    )

    n = len(
        oracle[
            "gps_reliability"
        ]
    )

    test_start = int(
        n
        *
        0.85
    )

    start = max(
        test_start,
        SEQUENCE_LENGTH
        -
        1
        +
        HORIZON,
    )

    print("=" * 108)
    print(
        "STRICT TEST-ONLY PREDICTIVE FACTOR RELIABILITY V1"
    )
    print("=" * 108)

    print(
        "Test target-frame range:",
        start,
        "->",
        n - 1,
    )

    for sensor in SENSORS:
        target = np.asarray(
            oracle[
                f"{sensor}_reliability"
            ],
            dtype=np.float64,
        )

        prediction = np.loadtxt(
            os.path.join(
                PRED_DIR,
                f"{sensor}_predictive_prior_target_aligned.txt",
            ),
            dtype=np.float64,
        ).reshape(
            -1
        )

        end = min(
            len(
                target
            ),
            len(
                prediction
            ),
        )

        mae, rmse, corr, bad_recall = metrics(
            target[
                start:end
            ],
            prediction[
                start:end
            ],
        )

        print()
        print(
            sensor.upper()
        )

        print(
            f"MAE             : {mae:.6f}"
        )

        print(
            f"RMSE            : {rmse:.6f}"
        )

        print(
            f"Correlation     : {corr:.6f}"
        )

        print(
            f"BadRecall@0.5   : {bad_recall:.6f}"
        )


if __name__ == "__main__":
    main()
