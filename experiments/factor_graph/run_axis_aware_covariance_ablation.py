from __future__ import annotations

import os
import sys


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

from src.factor_graph.axis_aware_dynamic_covariance import (
    POLICIES,
    run_axis_aware_factor_graph,
)


DEGRADED_DIR = os.path.join(
    ROOT,
    "results",
    "degraded_four_sensor_measurements",
)

PREDICTION_DIR = os.path.join(
    ROOT,
    "results",
    "predictive_factor_reliability_v1",
)

OUTPUT_ROOT = os.path.join(
    ROOT,
    "results",
    "axis_aware_covariance_ablation",
)


RUN_ORDER = [
    "best_subset_control",
    "gps_z_fixed",
    "camera_z_fixed",
    "gps_camera_z_fixed",
    "camera_rpz_fixed",
    "full_axis_aware",
]


def main():
    print(
        "=" * 120
    )

    print(
        "AXIS-AWARE / DOF-SPECIFIC DYNAMIC COVARIANCE ABLATION"
    )

    print(
        "=" * 120
    )

    measurements = (
        load_degraded_four_sensor_measurements(
            DEGRADED_DIR
        )
    )

    print(
        "Frames:",
        len(
            measurements.gps_local
        ),
    )

    print(
        "Prediction:",
        PREDICTION_DIR,
    )

    print(
        "LiDAR policy: FIXED in every case."
    )

    for index, name in enumerate(
        RUN_ORDER,
        start=1,
    ):
        print()
        print(
            "#" * 120
        )

        print(
            f"[{index}/{len(RUN_ORDER)}] "
            f"{name}"
        )

        print(
            "#" * 120
        )

        output_dir = os.path.join(
            OUTPUT_ROOT,
            name,
        )

        run_axis_aware_factor_graph(
            measurements=
                measurements,
            output_dir=
                output_dir,
            prediction_dir=
                PREDICTION_DIR,
            policy_name=
                name,
        )

    print()
    print(
        "Axis-aware covariance ablation finished."
    )


if __name__ == "__main__":
    main()
