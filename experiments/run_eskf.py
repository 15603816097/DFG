import os
import sys

import numpy as np


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.append(PROJECT_ROOT)



from src.loader.imu_loader import IMULoader

from src.preprocessing.coordinate import (
    GPSCoordinateConverter
)

from src.estimation.eskf import ESKF



DATASET_ROOT = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "kitti",
    "2011_10_03",
    "2011_10_03_drive_0027_sync"
)


RESULT_FILE = os.path.join(
    PROJECT_ROOT,
    "results",
    "eskf",
    "trajectory.txt"
)



def save_result(
    trajectory
):

    os.makedirs(
        os.path.dirname(
            RESULT_FILE
        ),
        exist_ok=True
    )


    np.savetxt(
        RESULT_FILE,
        trajectory,
        fmt="%.6f"
    )





def main():


    print("="*60)
    print("ESKF Baseline")
    print("="*60)


    imu_loader = IMULoader(
        DATASET_ROOT
    )


    converter = GPSCoordinateConverter()


    eskf = ESKF()


    trajectory=[]


    last_time=None



    for i in range(
        len(imu_loader)
    ):


        imu = imu_loader[i]


        timestamp = imu["timestamp"]



        if last_time is None:

            dt = 0.1

        else:

            dt = (
                timestamp-last_time
            ).total_seconds()


            if dt<=0:

                dt=0.1


        last_time=timestamp



        # prediction

        eskf.predict(
            imu["acceleration"],
            dt
        )


        # GPS

        gps = converter.gps_to_enu(
            imu["latitude"],
            imu["longitude"],
            imu["altitude"]
        )


        eskf.update(
            gps
        )


        trajectory.append(
            eskf.get_position()
        )


        if i%500==0:

            print(
                f"Processing {i}/{len(imu_loader)}"
            )



    trajectory=np.asarray(
        trajectory
    )


    save_result(
        trajectory
    )


    print()
    print("Finished")

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



if __name__=="__main__":

    main()
