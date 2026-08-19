from pathlib import Path
import numpy as np

from src.common.time_utils import load_kitti_timestamps



class GPSLoader:
    """
    KITTI GPS Loader

    数据来源:
    oxts/data/*.txt

    """


    def __init__(self, sequence_path):

        self.sequence_path = Path(sequence_path)


        self.oxts_path = (
            self.sequence_path
            /
            "oxts"
            /
            "data"
        )


        self.timestamp_path = (
            self.sequence_path
            /
            "oxts"
            /
            "timestamps.txt"
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
                "GPS timestamp mismatch"
            )


    def __len__(self):

        return len(self.files)



    def __getitem__(self,index):

        return self.load(index)



    def load(self,index):

        data = np.loadtxt(
            self.files[index]
        )


        return {


            "frame_id":
                index,


            "timestamp":
                self.timestamps[index],



            "position":
                np.array(
                    [
                        data[0],
                        data[1],
                        data[2]
                    ],
                    dtype=np.float64
                ),


            "velocity":
                np.array(
                    [
                        data[6],
                        data[7],
                        data[8]
                    ],
                    dtype=np.float32
                )

        }
