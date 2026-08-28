"""
Strict test-only evaluation.

This script intentionally evaluates ONLY the final 15% split.
It does not report full-sequence metrics as paper test results.
"""

from __future__ import annotations

import csv
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


from src.dataset.multisensor_predictive_dataset import (
    SENSORS,
    load_multisensor_npz,
)


DATA_PATH = os.path.join(
    ROOT,
    "results",
    "multisensor_reliability_v2",
    "multisensor_reliability_data_v2.npz",
)

PRED_DIR = os.path.join(
    ROOT,
    "results",
    "multisensor_predictive_reliability_v2",
)

OUTPUT_CSV = os.path.join(
    ROOT,
    "results",
    "multisensor_predictive_reliability_v2",
    "target_aligned_test_metrics.csv",
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
        if np.any(
            bad
        )
        else 1.0
    )

    severe = (
        y
        <
        0.2
    )

    severe_recall = (
        float(
            np.mean(
                p[
                    severe
                ]
                <
                0.2
            )
        )
        if np.any(
            severe
        )
        else 1.0
    )

    return {
        "MAE":
            mae,

        "RMSE":
            rmse,

        "Correlation":
            corr,

        "BadRecall@0.5":
            bad_recall,

        "SevereRecall@0.2":
            severe_recall,
    }


def main():
    raw = load_multisensor_npz(
        DATA_PATH
    )

    n = int(
        raw[
            "length"
        ]
    )

    horizon = int(
        raw[
            "horizon"
        ]
    )

    test_start = int(
        n
        *
        0.85
    )

    # Need enough history and a valid predictive source.
    start = max(
        test_start,
        63
        +
        horizon,
    )

    print("=" * 100)
    print(
        "STRICT TEST-ONLY "
        "TARGET-ALIGNED EVALUATION V2"
    )
    print("=" * 100)

    print(
        "Test frame range:",
        start,
        "->",
        n - 1,
    )

    rows = []

    for sensor in SENSORS:
        pred = np.loadtxt(
            os.path.join(
                PRED_DIR,
                f"{sensor}_predictive_prior_target_aligned.txt",
            ),
            dtype=np.float64,
        ).reshape(-1)

        target = raw[
            "current_labels"
        ][
            sensor
        ]

        end = min(
            n,
            len(
                pred
            ),
            len(
                target
            ),
        )

        result = metrics(
            target[
                start:end
            ],
            pred[
                start:end
            ],
        )

        row = {
            "sensor":
                sensor,

            **result,
        }

        rows.append(
            row
        )

        print()
        print(
            sensor.upper()
        )

        for key, value in result.items():
            print(
                f"{key:18s}: "
                f"{value:.6f}"
            )

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "sensor",
                "MAE",
                "RMSE",
                "Correlation",
                "BadRecall@0.5",
                "SevereRecall@0.2",
            ],
        )

        writer.writeheader()
        writer.writerows(
            rows
        )

    print()
    print(
        "Saved:",
        OUTPUT_CSV,
    )


if __name__ == "__main__":
    main()
