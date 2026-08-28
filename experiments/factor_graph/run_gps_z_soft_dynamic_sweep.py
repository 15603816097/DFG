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
    sys.path.insert(0, ROOT)


from src.factor_graph.degraded_measurements import (
    load_degraded_four_sensor_measurements,
)

from src.factor_graph.gps_z_soft_dynamic_covariance import (
    ALPHA_VALUES,
    alpha_name,
    run_gps_z_soft_factor_graph,
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
    "gps_z_soft_dynamic_sweep",
)


def main():
    print("=" * 120)
    print("GPS Z SOFT-DYNAMIC COVARIANCE SWEEP")
    print("=" * 120)

    measurements = load_degraded_four_sensor_measurements(
        DEGRADED_DIR
    )

    print("Frames:", len(measurements.gps_local))
    print("GPS XY: predictive")
    print("GPS Z : soft dynamic alpha_z")
    print("IMU   : predictive")
    print("LiDAR : fixed")
    print("Camera: predictive")

    for index, alpha in enumerate(
        ALPHA_VALUES,
        start=1,
    ):
        print()
        print("#" * 120)
        print(
            f"[{index}/{len(ALPHA_VALUES)}] "
            f"alpha_z={alpha:.2f}"
        )
        print("#" * 120)

        run_gps_z_soft_factor_graph(
            measurements=measurements,
            output_dir=os.path.join(
                OUTPUT_ROOT,
                alpha_name(alpha),
            ),
            prediction_dir=PREDICTION_DIR,
            alpha_z=alpha,
        )

    print()
    print(
        "GPS Z soft-dynamic sweep finished."
    )


if __name__ == "__main__":
    main()
