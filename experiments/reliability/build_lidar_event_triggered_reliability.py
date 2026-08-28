from __future__ import annotations

import json
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


from src.factor_graph.degraded_measurements import (
    load_degraded_four_sensor_measurements,
)

from src.reliability.lidar_event_triggered_reliability import (
    LidarEventTriggerConfig,
    build_lidar_event_trigger,
    load_lidar_predictive_reliability,
)


DEGRADED_DIR = os.path.join(
    ROOT,
    "results",
    "degraded_four_sensor_measurements",
)

PREDICTIVE_DIR = os.path.join(
    ROOT,
    "results",
    "predictive_factor_reliability_v1",
)

OUTPUT_DIR = os.path.join(
    ROOT,
    "results",
    "lidar_event_triggered_reliability",
)

OUTPUT_NPZ = os.path.join(
    OUTPUT_DIR,
    "lidar_event_trigger.npz",
)


def main():
    print(
        "=" * 112
    )

    print(
        "BUILD LIDAR EVENT-TRIGGERED RELIABILITY"
    )

    print(
        "=" * 112
    )

    measurements = (
        load_degraded_four_sensor_measurements(
            DEGRADED_DIR
        )
    )

    n = len(
        measurements.gps_local
    )

    prediction = (
        load_lidar_predictive_reliability(
            PREDICTIVE_DIR,
            n,
        )
    )

    config = (
        LidarEventTriggerConfig()
    )

    result = (
        build_lidar_event_trigger(
            lidar_between=
                measurements.lidar_between,
            lidar_quality=
                measurements.lidar_quality,
            camera_between=
                measurements.camera_between,
            camera_valid=
                measurements.camera_valid,
            predictive_reliability=
                prediction,
            n_frames=
                n,
            config=
                config,
        )
    )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    np.savez_compressed(
        OUTPUT_NPZ,
        **result,
    )

    trigger_mask = np.asarray(
        result[
            "trigger_mask"
        ],
        dtype=bool,
    )

    severe_mask = np.asarray(
        result[
            "severe_mask"
        ],
        dtype=bool,
    )

    print(
        "Frames:",
        n,
    )

    print(
        "Triggered frames:",
        int(
            trigger_mask.sum()
        ),
        "/",
        n,
        f"({100.0 * trigger_mask.mean():.2f}%)",
    )

    print(
        "Severe frames:",
        int(
            severe_mask.sum()
        ),
        "/",
        n,
        f"({100.0 * severe_mask.mean():.2f}%)",
    )

    for key in (
        "event_score",
        "predictive_reliability",
        "quality_frame",
        "translation_disagreement",
        "rotation_disagreement_deg",
        "temporal_translation_z",
        "temporal_rotation_z",
    ):
        values = np.asarray(
            result[
                key
            ],
            dtype=np.float64,
        )

        print(
            f"{key:30s} "
            f"min={values.min():.6f} "
            f"max={values.max():.6f} "
            f"mean={values.mean():.6f}"
        )

    metadata = {
        "config":
            config.__dict__,
        "frames":
            n,
        "triggered_frames":
            int(
                trigger_mask.sum()
            ),
        "severe_frames":
            int(
                severe_mask.sum()
            ),
    }

    with open(
        os.path.join(
            OUTPUT_DIR,
            "metadata.json",
        ),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

    print(
        "Saved:",
        OUTPUT_NPZ,
    )


if __name__ == "__main__":
    main()
