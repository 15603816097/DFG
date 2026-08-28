from pathlib import Path

import numpy as np

from src.odometry.stereo_visual_odometry import (
    _parse_kitti_calibration,
)


def test_parse_kitti_calibration(
    tmp_path,
):
    path = Path(
        tmp_path
    ) / "calib.txt"

    path.write_text(
        "P_rect_02: "
        "700 0 600 0 "
        "0 700 180 0 "
        "0 0 1 0\n"
        "P_rect_03: "
        "700 0 600 -378 "
        "0 700 180 0 "
        "0 0 1 0\n",
        encoding="utf-8",
    )

    calib = (
        _parse_kitti_calibration(
            path
        )
    )

    assert (
        "P_rect_02"
        in calib
    )

    assert (
        calib[
            "P_rect_02"
        ].shape
        ==
        (
            12,
        )
    )
