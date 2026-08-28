from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

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


from src.factor_graph.degraded_measurements import (
    load_degraded_four_sensor_measurements,
)

from src.factor_graph.sensorwise_reliability_graph import (
    ReliabilitySource,
    run_sensorwise_reliability_graph,
)


BASE_DIR = os.path.join(
    ROOT,
    "results",
    "degraded_four_sensor_measurements",
)

PREDICTIVE_DIR = os.path.join(
    ROOT,
    "results",
    "predictive_factor_reliability_v1",
)

LIDAR_PRED_DIR = os.path.join(
    ROOT,
    "results",
    "lidar_factor_error_predictive_v1",
)

PHYSICAL_FACTOR_PATH = os.path.join(
    ROOT,
    "results",
    "lidar_physical_oracle_benchmark",
    "physical_lidar_factor_data.npz",
)

OUTPUT_DIR = os.path.join(
    ROOT,
    "results",
    "lidar_factor_error_predictive_fg_v1",
)


def make_measurements():
    base = (
        load_degraded_four_sensor_measurements(
            BASE_DIR
        )
    )

    physical = np.load(
        PHYSICAL_FACTOR_PATH,
        allow_pickle=False,
    )

    m = SimpleNamespace(
        **dict(
            vars(
                base
            )
        )
    )

    m.lidar_between = np.asarray(
        physical[
            "body_between"
        ],
        dtype=np.float64,
    )

    m.lidar_quality = np.asarray(
        physical[
            "quality"
        ],
        dtype=np.float64,
    )

    m.lidar_valid = np.asarray(
        physical[
            "converged"
        ],
        dtype=bool,
    )

    return m


def make_prediction_dir():
    """
    Build a temporary compatibility directory:
        GPS/IMU/Camera -> existing V1 prediction
        LiDAR          -> new factor-error-aware prediction
    """

    out = (
        Path(ROOT)
        /
        "results"
        /
        "lidar_factor_error_predictive_v1"
        /
        "graph_prediction_dir"
    )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    mapping = {
        "gps":
            Path(
                PREDICTIVE_DIR
            )
            /
            "gps_predictive_prior_target_aligned.txt",

        "imu":
            Path(
                PREDICTIVE_DIR
            )
            /
            "imu_predictive_prior_target_aligned.txt",

        "camera":
            Path(
                PREDICTIVE_DIR
            )
            /
            "camera_predictive_prior_target_aligned.txt",

        "lidar":
            Path(
                LIDAR_PRED_DIR
            )
            /
            "lidar_predictive_prior_target_aligned.txt",
    }

    for sensor, src in mapping.items():
        if not src.exists():
            raise FileNotFoundError(
                src
            )

        dst = (
            out
            /
            f"{sensor}_predictive_prior_target_aligned.txt"
        )

        values = np.loadtxt(
            src,
            dtype=np.float64,
        )

        np.savetxt(
            dst,
            values,
            fmt="%.10f",
        )

    return str(
        out
    )


def main():
    print(
        "=" * 124
    )

    print(
        "LIDAR FACTOR-ERROR-AWARE PREDICTIVE FOUR-SENSOR FG V1"
    )

    print(
        "=" * 124
    )

    measurements = (
        make_measurements()
    )

    graph_prediction_dir = (
        make_prediction_dir()
    )

    sources = {
        "gps":
            ReliabilitySource(
                "predictive"
            ),

        "imu":
            ReliabilitySource(
                "predictive"
            ),

        "lidar":
            ReliabilitySource(
                "predictive"
            ),

        "camera":
            ReliabilitySource(
                "predictive"
            ),
    }

    print(
        "Frames:",
        len(
            measurements.gps_local
        ),
    )

    print(
        "LiDAR factors:",
        len(
            measurements.lidar_between
        ),
    )

    print(
        "Graph prediction dir:",
        graph_prediction_dir,
    )

    run_sensorwise_reliability_graph(
        measurements=
            measurements,
        output_dir=
            OUTPUT_DIR,
        sources=
            sources,
        predictive_dir=
            graph_prediction_dir,
        oracle_path=
            None,
    )

    print(
        "Saved:",
        OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
