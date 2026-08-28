"""
Trajectory Evaluation

Metrics:

ATE 3D RMSE
ATE 2D RMSE
Mean Error
Max Error
Z RMSE


Methods:

1. IMU Dead Reckoning
2. ESKF
3. Factor Graph
4. Dynamic Covariance FG
5. FG GPS Noise
6. FG GPS Degradation

"""



import os
import numpy as np



# =====================================================
# Path
# =====================================================


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




# =====================================================
# Trajectory files
# =====================================================


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



    "Dynamic Covariance FG":

    os.path.join(
        RESULT_ROOT,
        "dynamic_covariance_fg",
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



    "FG GPS Degradation":

    os.path.join(
        RESULT_ROOT,
        "factor_graph_degradation",
        "trajectory.txt"
    )

}





# =====================================================
# Load
# =====================================================


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


    return traj[:,:3]







# =====================================================
# Align origin
# =====================================================


def align_origin(traj):


    return traj-traj[0]








# =====================================================
# Metrics
# =====================================================


def evaluate(
    pred,
    gt
):


    N=min(
        len(pred),
        len(gt)
    )



    pred=pred[:N]

    gt=gt[:N]



    pred=align_origin(
        pred
    )


    gt=align_origin(
        gt
    )



    error=pred-gt




    # =========================
    # 3D
    # =========================


    error3d=np.linalg.norm(
        error,
        axis=1
    )



    ate3d=np.sqrt(
        np.mean(
            error3d**2
        )
    )


    mean3d=np.mean(
        error3d
    )


    max3d=np.max(
        error3d
    )





    # =========================
    # 2D
    # =========================


    error2d=np.linalg.norm(
        error[:,:2],
        axis=1
    )


    ate2d=np.sqrt(
        np.mean(
            error2d**2
        )
    )


    mean2d=np.mean(
        error2d
    )


    max2d=np.max(
        error2d
    )





    # =========================
    # Z
    # =========================


    z_rmse=np.sqrt(
        np.mean(
            error[:,2]**2
        )
    )




    return {


        "ATE_3D_RMSE(m)": ate3d,


        "ATE_2D_RMSE(m)": ate2d,


        "Mean_3D_Error(m)": mean3d,


        "Max_3D_Error(m)": max3d,


        "Mean_2D_Error(m)": mean2d,


        "Max_2D_Error(m)": max2d,


        "Z_RMSE(m)": z_rmse

    }








# =====================================================
# Main
# =====================================================


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



        metrics=evaluate(
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





    # =================================================
    # Save
    # =================================================


    save_path=os.path.join(
        RESULT_ROOT,
        "evaluation_all.txt"
    )



    with open(
        save_path,
        "w"
    ) as f:


        f.write(
            "Trajectory Evaluation\n"
        )



        for name,result in all_results.items():


            f.write(
                "\n"
            )


            f.write(
                "-"*60+"\n"
            )


            f.write(
                name+"\n"
            )



            for k,v in result.items():

                f.write(
                    f"{k}: {v:.6f}\n"
                )





    print()

    print("="*60)

    print(
        "Saved:"
    )

    print(
        save_path
    )

    print("="*60)






if __name__=="__main__":

    main()
