"""
Trajectory Evaluation

Metrics:

ATE 3D RMSE
ATE 2D RMSE
Mean Error
Max Error
Z RMSE


Compare:

IMU
ESKF
Factor Graph
Factor Graph under GPS degradation

"""


import os
import numpy as np




PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)



RESULT_ROOT = os.path.join(
    PROJECT_ROOT,
    "results"
)



GT_PATH = os.path.join(
    RESULT_ROOT,
    "ground_truth",
    "trajectory.txt"
)




METHODS = {


    "IMU Dead Reckoning":

        os.path.join(
            RESULT_ROOT,
            "imu_baseline",
            "trajectory.txt"
        ),



    "ESKF":

        os.path.join(
            RESULT_ROOT,
            "eskf",
            "trajectory.txt"
        ),



    "Factor Graph":

        os.path.join(
            RESULT_ROOT,
            "factor_graph",
            "trajectory.txt"
        ),



    "FG GPS Noise 5m":

        os.path.join(
            RESULT_ROOT,
            "factor_graph_noise_5",
            "trajectory.txt"
        ),



    "FG GPS Noise 10m":

        os.path.join(
            RESULT_ROOT,
            "factor_graph_noise_10",
            "trajectory.txt"
        ),



    "FG GPS Noise 20m":

        os.path.join(
            RESULT_ROOT,
            "factor_graph_noise_20",
            "trajectory.txt"
        ),



    "FG GPS Noise 50m":

        os.path.join(
            RESULT_ROOT,
            "factor_graph_noise_50",
            "trajectory.txt"
        ),

}






def load_trajectory(path):

    if not os.path.exists(path):

        raise FileNotFoundError(
            path
        )


    traj=np.loadtxt(
        path
    )


    if traj.ndim==1:

        traj=traj.reshape(
            1,-1
        )


    return traj[:,0:3]








def align_origin(traj):

    """
    Remove initial position offset

    """

    return traj-traj[0]









def compute_metrics(
    pred,
    gt
):


    # same length

    N=min(
        len(pred),
        len(gt)
    )


    pred=pred[:N]

    gt=gt[:N]



    # origin alignment

    pred=align_origin(
        pred
    )


    gt=align_origin(
        gt
    )



    error=pred-gt



    # =========================
    # 3D error
    # =========================

    error_3d=np.linalg.norm(
        error,
        axis=1
    )


    ate_3d_rmse=np.sqrt(
        np.mean(
            error_3d**2
        )
    )



    mean_3d=np.mean(
        error_3d
    )


    max_3d=np.max(
        error_3d
    )



    # =========================
    # 2D xy error
    # =========================


    error_2d=np.linalg.norm(
        error[:,:2],
        axis=1
    )


    ate_2d_rmse=np.sqrt(
        np.mean(
            error_2d**2
        )
    )



    mean_2d=np.mean(
        error_2d
    )


    max_2d=np.max(
        error_2d
    )



    # =========================
    # height
    # =========================


    z_rmse=np.sqrt(
        np.mean(
            error[:,2]**2
        )
    )



    return {


        "ATE_3D_RMSE":

            ate_3d_rmse,



        "ATE_2D_RMSE":

            ate_2d_rmse,



        "Mean_3D_Error":

            mean_3d,



        "Max_3D_Error":

            max_3d,



        "Mean_2D_Error":

            mean_2d,



        "Max_2D_Error":

            max_2d,



        "Z_RMSE":

            z_rmse

    }








def main():


    print("="*60)

    print(
        "Trajectory Evaluation"
    )

    print("="*60)



    gt=load_trajectory(
        GT_PATH
    )



    print(
        "Ground Truth:",
        gt.shape
    )



    all_results={}



    for name,path in METHODS.items():


        if not os.path.exists(path):

            print()

            print(
                "Skip:",
                name
            )

            continue



        traj=load_trajectory(
            path
        )



        metrics=compute_metrics(
            traj,
            gt
        )


        all_results[name]=metrics



        print()

        print("-"*60)

        print(
            name
        )

        print(
            "Trajectory:",
            traj.shape
        )


        for k,v in metrics.items():

            print(
                f"{k:<25}: {v:.6f}"
            )





    # ===============================
    # save
    # ===============================


    output=os.path.join(
        RESULT_ROOT,
        "evaluation_all.txt"
    )


    with open(
        output,
        "w"
    ) as f:


        for name,metrics in all_results.items():


            f.write(
                "\n"
            )

            f.write(
                "-"*60+"\n"
            )


            f.write(
                name+"\n"
            )


            for k,v in metrics.items():

                f.write(
                    f"{k}: {v:.6f}\n"
                )



    print()

    print("="*60)

    print(
        "Saved:"
    )

    print(
        output
    )

    print("="*60)






if __name__=="__main__":

    main()
