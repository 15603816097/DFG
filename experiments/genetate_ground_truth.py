"""
Generate KITTI Ground Truth trajectory

GPS WGS84
    |
    ↓
ENU coordinate

Output:
results/ground_truth/trajectory.txt
"""


import os
import sys

import numpy as np


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.append(
    PROJECT_ROOT
)


from src.loader.imu_loader import IMULoader

from src.preprocessing.coordinate import (
    GPSCoordinateConverter
)



DATASET_ROOT = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "kitti",
    "2011_10_03",
    "2011_10_03_drive_0027_sync"
)



OUTPUT_FILE = os.path.join(
    PROJECT_ROOT,
    "results",
    "ground_truth",
    "trajectory.txt"
)





def main():


    print("="*60)
    print("Generate Ground Truth")
    print("="*60)



    imu_loader = IMULoader(
        DATASET_ROOT
    )


    converter = GPSCoordinateConverter()


    trajectory=[]



    for i in range(
        len(imu_loader)
    ):


        imu = imu_loader[i]


        enu = converter.gps_to_enu(
            imu["latitude"],
            imu["longitude"],
            imu["altitude"]
        )


        trajectory.append(
            enu
        )



        if i % 500 == 0:

            print(
                f"Processing {i}/{len(imu_loader)}"
            )



    trajectory=np.asarray(
        trajectory
    )



    os.makedirs(
        os.path.dirname(
            OUTPUT_FILE
        ),
        exist_ok=True
    )


    np.savetxt(
        OUTPUT_FILE,
        trajectory,
        fmt="%.6f"
    )


    print()
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
        OUTPUT_FILE
    )




if __name__=="__main__":

    main()
