"""
Fixed Covariance Factor Graph
under GPS degradation


GPS:

results/gps_degradation/gps_corrupted.txt


Covariance:

fixed sigma = 5m


Used for comparison with:

Dynamic Covariance Factor Graph


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





# =====================================================
# Path
# =====================================================


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



GPS_PATH=os.path.join(

    PROJECT_ROOT,

    "results",

    "gps_degradation",

    "gps_corrupted.txt"

)




OUTPUT=os.path.join(

    PROJECT_ROOT,

    "results",

    "factor_graph_degradation",

    "trajectory.txt"

)








def main():



    print("="*60)

    print(
        "Fixed Covariance Factor Graph"
    )

    print(
        "GPS degradation experiment"
    )

    print("="*60)





    # ===============================================
    # Load IMU
    # ===============================================


    imu_loader=IMULoader(

        DATASET

    )



    N=len(imu_loader)



    print(

        "Frames:",

        N

    )





    # ===============================================
    # Load corrupted GPS
    # ===============================================


    if not os.path.exists(GPS_PATH):

        raise FileNotFoundError(

            GPS_PATH

        )



    gps=np.loadtxt(

        GPS_PATH

    )



    print(

        "GPS:",
        gps.shape

    )





    # ===============================================
    # Prepare IMU
    # ===============================================


    acc=[]

    gyro=[]



    for i in range(N):


        data=imu_loader[i]


        acc.append(

            data["acceleration"]

        )


        gyro.append(

            data["angular_velocity"]

        )



    acc=np.asarray(acc)

    gyro=np.asarray(gyro)







    # ===============================================
    # GTSAM graph
    # ===============================================


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

        1.0

    )


    bias_noise=noiseModel.Isotropic.Sigma(

        6,

        1e-3

    )





    # ===============================================
    # Initial
    # ===============================================


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






    # ===============================================
    # Build graph
    # ===============================================



    for i in range(N-1):



        dt=0.1



        pim=PreintegratedImuMeasurements(

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






        # ============================
        # Fixed GPS covariance
        # ============================


        gps_noise=noiseModel.Isotropic.Sigma(

            3,

            5.0

        )



        graph.add(

            GPSFactor(

                symbol('x',i),

                Point3(

                    *gps[i]

                ),

                gps_noise

            )

        )





        # initial guess


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





    # ===============================================
    # Save trajectory
    # ===============================================


    trajectory=[]



    for i in range(N):


        pose=result.atPose3(

            symbol('x',i)

        )


        t=pose.translation()



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
