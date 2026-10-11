from pathlib import Path

import numpy as np

from src.common.time_utils import load_kitti_timestamps


class IMULoader:
    """
    KITTI Raw OXTS IMU Loader.

    Data source
    -----------
    oxts/data/*.txt

    KITTI OXTS fields used here
    ---------------------------
    0  : latitude
    1  : longitude
    2  : altitude

    3  : roll
    4  : pitch
    5  : yaw

    11 : ax
    12 : ay
    13 : az

    17 : wx
    18 : wy
    19 : wz
    """

    def __init__(self, sequence_path):
        self.sequence_path = Path(sequence_path)

        self.oxts_path = (
            self.sequence_path
            / "oxts"
            / "data"
        )

        self.timestamp_path = (
            self.sequence_path
            / "oxts"
            / "timestamps.txt"
        )

        if not self.oxts_path.exists():
            raise FileNotFoundError(
                self.oxts_path
            )

        if not self.timestamp_path.exists():
            raise FileNotFoundError(
                self.timestamp_path
            )

        self.files = sorted(
            self.oxts_path.glob("*.txt")
        )

        self.timestamps = (
            load_kitti_timestamps(
                self.timestamp_path
            )
        )

        if len(self.files) != len(self.timestamps):
            raise RuntimeError(
                f"IMU files {len(self.files)} "
                f"!= timestamps {len(self.timestamps)}"
            )

        if len(self.files) == 0:
            raise RuntimeError(
                f"No OXTS files found in {self.oxts_path}"
            )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        return self.load(index)

    def load(self, index):
        if index < 0 or index >= len(self.files):
            raise IndexError(
                f"IMU frame index out of range: "
                f"{index}, total={len(self.files)}"
            )

        file = self.files[index]

        data = np.asarray(
            np.loadtxt(file),
            dtype=np.float64,
        ).reshape(-1)

        if data.size < 20:
            raise ValueError(
                f"Invalid KITTI OXTS frame: {file}. "
                f"Expected at least 20 values, got {data.size}"
            )

        result = {
            "frame_id": index,

            "timestamp":
                self.timestamps[index],

            # -------------------------------------------------
            # GPS / geodetic position
            # -------------------------------------------------
            "latitude":
                float(data[0]),

            "longitude":
                float(data[1]),

            "altitude":
                float(data[2]),

            # -------------------------------------------------
            # Vehicle attitude in KITTI OXTS navigation frame
            #
            # KITTI stores these values in radians.
            # -------------------------------------------------
            "roll":
                float(data[3]),

            "pitch":
                float(data[4]),

            "yaw":
                float(data[5]),

            "orientation":
                np.array(
                    [
                        data[3],
                        data[4],
                        data[5],
                    ],
                    dtype=np.float64,
                ),

            # -------------------------------------------------
            # IMU acceleration
            # -------------------------------------------------
            "acceleration":
                np.array(
                    [
                        data[11],
                        data[12],
                        data[13],
                    ],
                    dtype=np.float32,
                ),

            # -------------------------------------------------
            # IMU angular velocity
            # -------------------------------------------------
            "angular_velocity":
                np.array(
                    [
                        data[17],
                        data[18],
                        data[19],
                    ],
                    dtype=np.float32,
                ),
        }

        return result
