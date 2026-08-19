"""
GTSAM IMU Preintegration Factor Graph Baseline

State:

Pose3
Velocity3
Bias


Factors:

1. IMU Preintegration Factor
2. GPS Position Factor
3. Bias Random Walk


Output:

results/factor_graph/trajectory.txt

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
    BetweenFactorConstantBias,

    PreintegratedImuMeasurements,
    PreintegrationParams,

    noiseModel
)



PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


sys.path.append(
    PROJECT_ROOT
)



from src.loader.imu_loader import IMULoader
from src.preprocessing.coordinate import GPSCoordinateConverter





DATASET = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "kitti",
    "2011_10_03",
    "2011_10_03_drive_0027_sync"
)



OUTPUT = os.path.join(
    PROJECT_ROOT,
    "results",
    "factor_graph",
    "trajectory.txt"
)






def main():


    print("="*60)
    print("IMU Preintegration Factor Graph Baseline")
    print("="*60)



    loader = IMULoader(
        DATASET
    )



    converter = GPSCoordinateConverter()



    N=len(loader)



    print(
        "Frames:",
        N
    )



    # =====================================================
    # Load sensors
    # =====================================================


    gps=[]
    acc=[]
    gyro=[]



    for i in range(N):


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



    # =====================================================
    # GTSAM setup
    # =====================================================


    graph=NonlinearFactorGraph()

    initial=Values()



    # Gravity

    gravity=9.81



    params = PreintegrationParams.MakeSharedU(
        gravity
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



    # =====================================================
    # Noise
    # =====================================================


    pose_noise=noiseModel.Diagonal.Sigmas(
        np.array(
            [
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
                0.1
            ]
        )
    )



    velocity_noise=noiseModel.Isotropic.Sigma(
        3,
        1.0
    )



    bias_noise=noiseModel.Isotropic.Sigma(
        6,
        1e-3
    )



    gps_noise=noiseModel.Isotropic.Sigma(
        3,
        5.0
    )





    # =====================================================
    # Initial state
    # =====================================================


    pose0 = Pose3(
        Rot3(),
        Point3(
            gps[0,0],
            gps[0,1],
            gps[0,2]
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





    # =====================================================
    # IMU preintegration
    # =====================================================


    for i in range(N-1):


        dt=0.1



        pim = PreintegratedImuMeasurements(
            params,
            bias0
        )



        pim.integrateMeasurement(
            acc[i],
            gyro[i],
            dt
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



        # initial guess

        initial.insert(

            symbol('x',i+1),

            Pose3(
                Rot3(),
                Point3(
                    gps[i+1,0],
                    gps[i+1,1],
                    gps[i+1,2]
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




        # GPS every 10 frames

        if i % 10 ==0:


            graph.add(

                GPSFactor(

                    symbol('x',i),

                    Point3(
                        gps[i,0],
                        gps[i,1],
                        gps[i,2]
                    ),

                    gps_noise

                )

            )



        if i%500==0:

            print(
                "Add factors:",
                i
            )





    print()

    print(
        "Graph size:",
        graph.size()
    )



    print(
        "Optimizing..."
    )



    optimizer=LevenbergMarquardtOptimizer(
        graph,
        initial
    )



    result=optimizer.optimize()





    # =====================================================
    # Save trajectory
    # =====================================================


    trajectory=[]


    for i in range(N):


        pose=result.atPose3(
            symbol('x',i)
        )


        t=pose.translation()



        trajectory.append(
            [
                float(t[0]),
                float(t[1]),
               	float(t[2])
            ]
        )



    trajectory=np.asarray(
        trajectory
    )



    os.makedirs(
        os.path.dirname(OUTPUT),
        exist_ok=True
    )



    np.savetxt(
        OUTPUT,
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
        OUTPUT
    )





if __name__=="__main__":

    main()
