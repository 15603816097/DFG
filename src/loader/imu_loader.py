from pathlib import Path
import numpy as np

from src.common.time_utils import load_kitti_timestamps


class IMULoader:
    """
    KITTI OXTS IMU Loader

    数据来源:

    oxts/data/*.txt

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


    def __len__(self):

        return len(self.files)



    def __getitem__(self, index):

        return self.load(index)



    def load(self, index):

        file = self.files[index]


        data = np.loadtxt(
            file
        )


        result = {

            "frame_id": index,

            "timestamp":
                self.timestamps[index],


            # GPS
            "latitude":
                data[0],

            "longitude":
                data[1],

            "altitude":
                data[2],


            # IMU
            "acceleration":
                np.array(
                    [
                        data[11],
                        data[12],
                        data[13]
                    ],
                    dtype=np.float32
                ),


            "angular_velocity":
                np.array(
                    [
                        data[17],
                        data[18],
                        data[19]
                    ],
                    dtype=np.float32
                )

        }


        return result
