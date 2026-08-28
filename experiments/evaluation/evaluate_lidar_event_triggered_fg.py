from __future__ import annotations

import os
from pathlib import Path

import numpy as np


ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)


GROUND_TRUTH = os.path.join(
    ROOT,
    "results",
    "ground_truth",
    "trajectory.txt",
)


METHODS = [
    (
        "Degraded Fixed Four-Sensor",
        os.path.join(
            ROOT,
            "results",
            "degraded_four_sensor_fixed_fg",
            "trajectory.txt",
        ),
    ),

    (
        "All Predictive V1",
        os.path.join(
            ROOT,
            "results",
            "predicted_factor_reliability_fg",
            "trajectory.txt",
        ),
    ),

    (
        "GPS+IMU+Camera Predictive + LiDAR Fixed",
        os.path.join(
            ROOT,
            "results",
            "sensorwise_predictive_ablation",
            "gps_imu_camera",
            "trajectory.txt",
        ),
    ),

    (
        "LiDAR Event-Triggered Dynamic",
        os.path.join(
            ROOT,
            "results",
            "lidar_event_triggered_fg",
            "trajectory.txt",
        ),
    ),

    (
        "Error-derived Oracle All",
        os.path.join(
            ROOT,
            "results",
            "oracle_four_sensor_fg",
            "trajectory.txt",
        ),
    ),
]


def metrics(
    reference,
    trajectory,
):
    n = min(
        len(
            reference
        ),
        len(
            trajectory
        ),
    )

    reference = np.asarray(
        reference[
            :n,
            :3
        ],
        dtype=np.float64,
    )

    trajectory = np.asarray(
        trajectory[
            :n,
            :3
        ],
        dtype=np.float64,
    )

    error = (
        trajectory
        -
        reference
    )

    e3 = np.linalg.norm(
        error,
        axis=1,
    )

    e2 = np.linalg.norm(
        error[
            :,
            :2
        ],
        axis=1,
    )

    return {
        "ATE3D":
            float(
                np.sqrt(
                    np.mean(
                        e3
                        **
                        2
                    )
                )
            ),

        "ATE2D":
            float(
                np.sqrt(
                    np.mean(
                        e2
                        **
                        2
                    )
                )
            ),

        "Mean3D":
            float(
                np.mean(
                    e3
                )
            ),

        "Max3D":
            float(
                np.max(
                    e3
                )
            ),

        "ZRMSE":
            float(
                np.sqrt(
                    np.mean(
                        error[
                            :,
                            2
                        ]
                        **
                        2
                    )
                )
            ),
    }


def main():
    if not Path(
        GROUND_TRUTH
    ).exists():
        raise FileNotFoundError(
            GROUND_TRUTH
        )

    reference = np.loadtxt(
        GROUND_TRUTH,
        dtype=np.float64,
    )

    print(
        "=" * 124
    )

    print(
        "LIDAR EVENT-TRIGGERED DYNAMIC COVARIANCE EVALUATION"
    )

    print(
        "=" * 124
    )

    results = {}

    for name, path in METHODS:
        if not Path(
            path
        ).exists():
            print(
                f"{name:44s}: NOT RUN"
            )

            continue

        trajectory = np.loadtxt(
            path,
            dtype=np.float64,
        )

        result = metrics(
            reference,
            trajectory,
        )

        results[
            name
        ] = result

        print(
            f"{name:44s} "
            f"ATE3D={result['ATE3D']:.6f} "
            f"ATE2D={result['ATE2D']:.6f} "
            f"Mean3D={result['Mean3D']:.6f} "
            f"Max3D={result['Max3D']:.6f} "
            f"ZRMSE={result['ZRMSE']:.6f}"
        )

    baseline_name = (
        "GPS+IMU+Camera Predictive + LiDAR Fixed"
    )

    event_name = (
        "LiDAR Event-Triggered Dynamic"
    )

    baseline = results.get(
        baseline_name
    )

    event = results.get(
        event_name
    )

    print()

    if (
        baseline
        and
        event
    ):
        ate_gain = (
            baseline[
                "ATE3D"
            ]
            -
            event[
                "ATE3D"
            ]
        )

        ate_percent = (
            ate_gain
            /
            baseline[
                "ATE3D"
            ]
            *
            100.0
        )

        z_gain = (
            baseline[
                "ZRMSE"
            ]
            -
            event[
                "ZRMSE"
            ]
        )

        print(
            "Event-triggered improvement "
            "vs current best:"
        )

        print(
            f"ATE3D gain: "
            f"{ate_gain:+.6f} m "
            f"({ate_percent:+.2f}%)"
        )

        print(
            f"ZRMSE gain: "
            f"{z_gain:+.6f} m"
        )

    output = os.path.join(
        ROOT,
        "results",
        "lidar_event_triggered_evaluation.txt",
    )

    lines = []

    for name, result in results.items():
        lines.append(
            f"{name},"
            f"{result['ATE3D']:.10f},"
            f"{result['ATE2D']:.10f},"
            f"{result['Mean3D']:.10f},"
            f"{result['Max3D']:.10f},"
            f"{result['ZRMSE']:.10f}"
        )

    Path(
        output
    ).write_text(
        "\n".join(
            lines
        )
        +
        "\n",
        encoding="utf-8",
    )

    print()

    print(
        "Saved:",
        output,
    )


if __name__ == "__main__":
    main()
