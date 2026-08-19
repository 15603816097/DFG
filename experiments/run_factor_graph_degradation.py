"""
Factor Graph Baseline with GPS degradation

Experiments:

GPS noise:
    5m
    10m
    20m
    50m

Output:

results/
    factor_graph_noise_x/

"""


import os
import sys
import numpy as np


import gtsam

from gtsam import (
    symbol,
    Values,
    NonlinearFactorGraph,
    LevenbergMarquardtOptimizer,

    Pose3,
    Point3,
    Rot3,

    PriorFactorPose3,
    PriorFactorVector,

    GPSFactor,
    ImuFactor,

    PreintegratedImuMeasurements,
    PreintegrationParams,

    BetweenFactorConstantBias,

    noiseModel
)



PROJECT_ROOT=os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


sys.path.append(
    PROJECT_ROOT
)


from src.loader.imu_loader import IMULoader
from src.preprocessing.coordinate import GPSCoordinateConverter





DATASET=os.path.join(
    PROJECT_ROOT,
    "dataset",
    "kitti",
    "2011_10_03",
    "2011_10_03_drive_0027_sync"
)





def run(
    gps_sigma
):


    print()
    print("="*60)
    print(
        f"GPS Noise Sigma = {gps_sigma}"
    )
    print("="*60)



    loader=IMULoader(
        DATASET
    )



    converter=GPSCoordinateConverter()



    gps=[]
    acc=[]
    gyro=[]



    for i in range(len(loader)):


        data=loader[i]


        gps.append(

            converter.gps_to_enu(
                data["latitude"],
                data["longitude"],
                data["altitude"]
            )

        )


        acc.append(
            data["acceleration"]
        )


        gyro.append(
            data["angular_velocity"]
        )



    gps=np.asarray(gps)
    acc=np.asarray(acc)
    gyro=np.asarray(gyro)



    N=len(gps)



    graph=NonlinearFactorGraph()

    initial=Values()



    params=PreintegrationParams.MakeSharedU(
        9.81
    )


    params.setAccelerometerCovariance(
        np.eye(3)*0.1
    )

    params.setGyroscopeCovariance(
        np.eye(3)*0.01
    )

    params.setIntegrationCovariance(
        np.eye(3)*0.001
    )



    bias0=gtsam.imuBias.ConstantBias()



    pose_noise=noiseModel.Isotropic.Sigma(
        6,
        0.1
    )


    velocity_noise=noiseModel.Isotropic.Sigma(
        3,
        1
    )


    bias_noise=noiseModel.Isotropic.Sigma(
        6,
        1e-3
    )



    gps_noise=noiseModel.Isotropic.Sigma(
        3,
        gps_sigma
    )



    # ==============================
    # initial
    # ==============================


    pose0=Pose3(
        Rot3(),
        Point3(
            *gps[0]
        )
    )



    graph.add(

        PriorFactorPose3(
            symbol('x',0),
            pose0,
            pose_noise
        )
    )



    graph.add(

        PriorFactorVector(
            symbol('v',0),
            np.zeros(3),
            velocity_noise
        )
    )



    graph.add(

        gtsam.PriorFactorConstantBias(
            symbol('b',0),
            bias0,
            bias_noise
        )
    )



    initial.insert(
        symbol('x',0),
        pose0
    )

    initial.insert(
        symbol('v',0),
        np.zeros(3)
    )


    initial.insert(
        symbol('b',0),
        bias0
    )



    # ==============================
    # factors
    # ==============================


    for i in range(N-1):


        pim=PreintegratedImuMeasurements(
            params,
            bias0
        )


        pim.integrateMeasurement(
            acc[i],
            gyro[i],
            0.1
        )



        graph.add(

            ImuFactor(

                symbol('x',i),
                symbol('v',i),

                symbol('x',i+1),
                symbol('v',i+1),

                symbol('b',i),

                pim

            )
        )



        graph.add(

            BetweenFactorConstantBias(

                symbol('b',i),
                symbol('b',i+1),

                gtsam.imuBias.ConstantBias(),

                bias_noise
            )

        )



        initial.insert(

            symbol('x',i+1),

            Pose3(
                Rot3(),
                Point3(
                    *gps[i+1]
                )
            )

        )


        initial.insert(
            symbol('v',i+1),
            np.zeros(3)
        )


        initial.insert(
            symbol('b',i+1),
            bias0
        )



        # GPS every 10 frame

        if i%10==0:


            graph.add(

                GPSFactor(

                    symbol('x',i),

                    Point3(
                        *gps[i]
                    ),

                    gps_noise

                )

            )



    print(
        "Optimizing..."
    )


    optimizer=LevenbergMarquardtOptimizer(
        graph,
        initial
    )


    result=optimizer.optimize()



    trajectory=[]


    for i in range(N):


        t=result.atPose3(
            symbol('x',i)
        ).translation()


        trajectory.append(
            [
                t[0],
                t[1],
                t[2]
            ]
        )


    trajectory=np.asarray(
        trajectory
    )



    out=os.path.join(

        PROJECT_ROOT,

        "results",

        f"factor_graph_noise_{gps_sigma}",

        "trajectory.txt"

    )


    os.makedirs(
        os.path.dirname(out),
        exist_ok=True
    )


    np.savetxt(
        out,
        trajectory,
        fmt="%.6f"
    )



    print(
        "Saved:",
        out
    )





if __name__=="__main__":


    for sigma in [

        5,
        10,
        20,
        50

    ]:

        run(
            sigma
        )
