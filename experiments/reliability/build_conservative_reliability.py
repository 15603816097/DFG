from __future__ import annotations

import os
import shutil
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


from src.reliability.conservative_fusion import (
    ConservativeFusionConfig,
    build_conservative_target_aligned_files,
)


SOURCE_V1 = os.path.join(
    ROOT,
    "results",
    "predictive_factor_reliability_v1",
)

DEGRADED_DIR = os.path.join(
    ROOT,
    "results",
    "degraded_four_sensor_measurements",
)

OUTPUT_DIR = os.path.join(
    ROOT,
    "results",
    "conservative_predictive_reliability",
)


def main():
    print("=" * 108)
    print(
        "BUILD CONSERVATIVE FOUR-SENSOR RELIABILITY - FIX V2"
    )
    print("=" * 108)

    # Remove the incorrect previous conservative directory.
    # This does NOT touch V1/V2/Oracle or any odometry result.
    if os.path.isdir(
        OUTPUT_DIR
    ):
        shutil.rmtree(
            OUTPUT_DIR
        )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    config = ConservativeFusionConfig()

    summary = build_conservative_target_aligned_files(
        SOURCE_V1,
        DEGRADED_DIR,
        OUTPUT_DIR,
        config=config,
    )

    for sensor in (
        "gps",
        "imu",
        "lidar",
        "camera",
    ):
        s = summary[
            sensor
        ]

        print()
        print(
            sensor.upper()
        )

        print(
            "V1 prediction "
            f"min/max/mean: "
            f"{s['pred_min']:.4f} / "
            f"{s['pred_max']:.4f} / "
            f"{s['pred_mean']:.4f}"
        )

        if (
            "quality_mean"
            in s
        ):
            print(
                "Online quality "
                f"min/max/mean: "
                f"{s['quality_min']:.4f} / "
                f"{s['quality_max']:.4f} / "
                f"{s['quality_mean']:.4f}"
            )

        print(
            "Final reliability "
            f"min/max/mean: "
            f"{s['final_min']:.4f} / "
            f"{s['final_max']:.4f} / "
            f"{s['final_mean']:.4f}"
        )

    print()
    print(
        "Only four target-aligned reliability files were written."
    )

    print(
        "No *_prediction_source_frame.txt file was modified."
    )

    print(
        "Saved:",
        OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
