"""
IMU Dead Reckoning Baseline

KITTI OXTS IMU integration baseline

Input:
    IMU acceleration

Output:
    Estimated trajectory
"""

import os
import sys

import numpy as np


# ============================
# Project root
# ============================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.append(PROJECT_ROOT)


from src.loader.imu_loader import IMULoader



# ============================
# Configuration
# ============================

DATASET_ROOT = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "kitti",
    "2011_10_03",
    "2011_10_03_drive_0027_sync"
)


RESULT_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "imu_baseline"
)


RESULT_FILE = os.path.join(
    RESULT_DIR,
    "trajectory.txt"
)



# KITTI IMU acceleration
# vehicle coordinate
GRAVITY = np.array(
    [
        0.0,
        0.0,
        9.81
    ],
    dtype=np.float64
)




class IMUDeadReckoning:
    """
    Simple IMU integration

    state:
        position
        velocity

    """

    def __init__(self):

        self.position = np.zeros(
            3,
            dtype=np.float64
        )

        self.velocity = np.zeros(
            3,
            dtype=np.float64
        )


        self.trajectory = []



    def update(
        self,
        acceleration,
        dt
    ):

        acceleration = np.asarray(
            acceleration,
            dtype=np.float64
        )


        # remove gravity
        acc = acceleration - GRAVITY


        self.position += (
            self.velocity * dt
            +
            0.5 * acc * dt * dt
        )


        self.velocity += (
            acc * dt
        )


        self.trajectory.append(
            self.position.copy()
        )



    def get_result(self):

        return np.asarray(
            self.trajectory
        )





def save_trajectory(
    trajectory,
    path
):

    os.makedirs(
        os.path.dirname(path),
        exist_ok=True
    )


    np.savetxt(
        path,
        trajectory,
        fmt="%.6f"
    )





def main():

    print("=" * 60)
    print("IMU Dead Reckoning Baseline")
    print("=" * 60)


    print(
        "Dataset:",
        DATASET_ROOT
    )


    imu_loader = IMULoader(
        DATASET_ROOT
    )


    num_frames = len(
        imu_loader
    )


    print(
        "IMU Frames:",
        num_frames
    )


    estimator = IMUDeadReckoning()


    last_time = None



    for i in range(num_frames):

        imu = imu_loader.load(i)


        timestamp = imu["timestamp"]


        if last_time is None:

            dt = 0.1

        else:

            delta = (
                timestamp
                -
                last_time
            )


            dt = delta.total_seconds()


            if dt <= 0:

                dt = 0.1


        last_time = timestamp



        acceleration = imu[
            "acceleration"
        ]


        estimator.update(
            acceleration,
            dt
        )



        if i % 500 == 0:

            print(
                f"Processing {i}/{num_frames}"
            )



    trajectory = estimator.get_result()



    save_trajectory(
        trajectory,
        RESULT_FILE
    )



    print()
    print("=" * 60)
    print("Finished")
    print("=" * 60)


    print(
        "Trajectory:",
        trajectory.shape
    )


    print(
        "Final position:",
        trajectory[-1]
    )


    print(
        "Saved:",
        RESULT_FILE
    )




if __name__ == "__main__":

    main()
