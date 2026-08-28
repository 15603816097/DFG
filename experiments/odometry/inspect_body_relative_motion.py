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


from src.odometry.body_frame_converter import (
    transform_angle,
)
from src.odometry.relative_factor_data import (
    RelativeFactorData,
    motion_disagreement,
)


OUTPUT_DIR = os.path.join(
    ROOT,
    "results",
    "body_relative_odometry",
)


def safe_stats(values):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    values = values[
        np.isfinite(
            values
        )
    ]

    if len(values) == 0:
        return (
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
        )

    return (
        float(
            np.mean(
                values
            )
        ),
        float(
            np.median(
                values
            )
        ),
        float(
            np.percentile(
                values,
                90.0,
            )
        ),
        float(
            np.max(
                values
            )
        ),
    )


def main():
    lidar = RelativeFactorData.load(
        os.path.join(
            OUTPUT_DIR,
            "lidar_factor_data.npz",
        )
    )

    camera = RelativeFactorData.load(
        os.path.join(
            OUTPUT_DIR,
            "camera_factor_data.npz",
        )
    )

    n = min(
        len(
            lidar
        ),
        len(
            camera
        ),
    )

    both = (
        lidar.valid[:n]
        &
        camera.valid[:n]
    )

    indices = np.flatnonzero(
        both
    )

    translation_error = []
    rotation_error_deg = []

    lidar_step = []
    camera_step = []

    rows = []

    for i in indices:
        lt = lidar.between_measurements[
            i
        ]

        ct = camera.between_measurements[
            i
        ]

        te, re = motion_disagreement(
            lt,
            ct,
        )

        ls = float(
            np.linalg.norm(
                lt[
                    :3,
                    3
                ]
            )
        )

        cs = float(
            np.linalg.norm(
                ct[
                    :3,
                    3
                ]
            )
        )

        translation_error.append(
            te
        )

        rotation_error_deg.append(
            re
        )

        lidar_step.append(
            ls
        )

        camera_step.append(
            cs
        )

        rows.append(
            [
                int(i),
                ls,
                cs,
                te,
                float(
                    np.degrees(
                        transform_angle(
                            lt
                        )
                    )
                ),
                float(
                    np.degrees(
                        transform_angle(
                            ct
                        )
                    )
                ),
                re,
                float(
                    lidar.quality[
                        i
                    ]
                ),
                float(
                    camera.quality[
                        i
                    ]
                ),
            ]
        )

    t_stats = safe_stats(
        translation_error
    )

    r_stats = safe_stats(
        rotation_error_deg
    )

    lidar_step_stats = safe_stats(
        lidar_step
    )

    camera_step_stats = safe_stats(
        camera_step
    )

    report = []

    report.append(
        "=" * 100
    )
    report.append(
        "BODY-FRAME RELATIVE ODOMETRY CONSISTENCY REPORT"
    )
    report.append(
        "=" * 100
    )
    report.append(
        f"Relative pairs: {n}"
    )
    report.append(
        f"LiDAR valid: {int(np.sum(lidar.valid[:n]))}/{n}"
    )
    report.append(
        f"Camera valid: {int(np.sum(camera.valid[:n]))}/{n}"
    )
    report.append(
        f"Both valid: {len(indices)}/{n}"
    )
    report.append(
        ""
    )
    report.append(
        "Translation disagreement [m] "
        "(mean / median / p90 / max): "
        f"{t_stats[0]:.6f} / "
        f"{t_stats[1]:.6f} / "
        f"{t_stats[2]:.6f} / "
        f"{t_stats[3]:.6f}"
    )
    report.append(
        "Rotation disagreement [deg] "
        "(mean / median / p90 / max): "
        f"{r_stats[0]:.6f} / "
        f"{r_stats[1]:.6f} / "
        f"{r_stats[2]:.6f} / "
        f"{r_stats[3]:.6f}"
    )
    report.append(
        ""
    )
    report.append(
        "LiDAR translation step [m] "
        "(mean / median / p90 / max): "
        f"{lidar_step_stats[0]:.6f} / "
        f"{lidar_step_stats[1]:.6f} / "
        f"{lidar_step_stats[2]:.6f} / "
        f"{lidar_step_stats[3]:.6f}"
    )
    report.append(
        "Camera translation step [m] "
        "(mean / median / p90 / max): "
        f"{camera_step_stats[0]:.6f} / "
        f"{camera_step_stats[1]:.6f} / "
        f"{camera_step_stats[2]:.6f} / "
        f"{camera_step_stats[3]:.6f}"
    )
    report.append(
        ""
    )
    report.append(
        "NOTE:"
    )
    report.append(
        "Large LiDAR-vs-camera disagreement does not automatically mean "
        "the frame conversion is wrong; the two odometry estimators have "
        "different failure modes. Use this report to identify outliers "
        "before inserting BetweenFactors."
    )

    report_text = "\n".join(
        report
    )

    print(
        report_text
    )

    with open(
        os.path.join(
            OUTPUT_DIR,
            "consistency_report.txt",
        ),
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            report_text
            +
            "\n"
        )

    with open(
        os.path.join(
            OUTPUT_DIR,
            "consistency_per_frame.csv",
        ),
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(
            file
        )

        writer.writerow(
            [
                "frame_t",
                "lidar_translation_m",
                "camera_translation_m",
                "translation_disagreement_m",
                "lidar_rotation_deg",
                "camera_rotation_deg",
                "rotation_disagreement_deg",
                "lidar_quality",
                "camera_quality",
            ]
        )

        writer.writerows(
            rows
        )

    print()
    print(
        "Saved:",
        os.path.join(
            OUTPUT_DIR,
            "consistency_report.txt",
        )
    )
    print(
        "Saved:",
        os.path.join(
            OUTPUT_DIR,
            "consistency_per_frame.csv",
        )
    )


if __name__ == "__main__":
    main()
