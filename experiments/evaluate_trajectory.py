"""
Trajectory Evaluation

Metrics:

1. ATE RMSE 3D
2. ATE RMSE 2D (XY)
3. Mean Error
4. Maximum Error
5. Z Error


Reference:
Ground Truth trajectory
"""


import os
import numpy as np





# ==================================================
# Path
# ==================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)



GT_FILE = os.path.join(
    PROJECT_ROOT,
    "results",
    "ground_truth",
    "trajectory.txt"
)



METHODS = {

    "IMU":
    os.path.join(
        PROJECT_ROOT,
        "results",
        "imu_baseline",
        "trajectory.txt"
    ),


    "ESKF":
    os.path.join(
        PROJECT_ROOT,
        "results",
        "eskf",
        "trajectory.txt"
    )

}



OUTPUT_FILE = os.path.join(
    PROJECT_ROOT,
    "results",
    "evaluation.txt"
)





# ==================================================
# Load
# ==================================================

def load_trajectory(path):

    if not os.path.exists(path):

        raise FileNotFoundError(
            path
        )


    return np.loadtxt(
        path
    )





# ==================================================
# Metrics
# ==================================================

def evaluate(
    estimate,
    gt
):


    error = (
        estimate
        -
        gt
    )


    # ------------------
    # 3D error
    # ------------------

    error_3d = np.linalg.norm(
        error,
        axis=1
    )


    ate_3d = np.sqrt(
        np.mean(
            error_3d**2
        )
    )



    mean_3d = np.mean(
        error_3d
    )



    max_3d = np.max(
        error_3d
    )



    # ------------------
    # 2D XY error
    # ------------------

    error_xy = np.linalg.norm(
        error[:,:2],
        axis=1
    )


    ate_2d = np.sqrt(
        np.mean(
            error_xy**2
        )
    )



    mean_2d = np.mean(
        error_xy
    )


    max_2d = np.max(
        error_xy
    )



    # ------------------
    # Z error
    # ------------------

    z_error = np.abs(
        error[:,2]
    )


    z_rmse = np.sqrt(
        np.mean(
            z_error**2
        )
    )


    return {

        "ATE_3D_RMSE": ate_3d,

        "ATE_2D_RMSE": ate_2d,

        "Mean_3D": mean_3d,

        "Max_3D": max_3d,

        "Mean_2D": mean_2d,

        "Max_2D": max_2d,

        "Z_RMSE": z_rmse

    }





# ==================================================
# Main
# ==================================================

def main():


    print("="*60)

    print(
        "Trajectory Evaluation"
    )

    print("="*60)



    gt = load_trajectory(
        GT_FILE
    )



    results={}



    for name,path in METHODS.items():


        trajectory = load_trajectory(
            path
        )


        if trajectory.shape != gt.shape:

            raise RuntimeError(
                f"{name} trajectory mismatch"
            )



        metrics = evaluate(
            trajectory,
            gt
        )


        results[name]=metrics



        print()
        print(
            name
        )

        for k,v in metrics.items():

            print(
                f"{k}: {v:.4f} m"
            )





    # save

    with open(
        OUTPUT_FILE,
        "w"
    ) as f:


        f.write(
            "Trajectory Evaluation\n"
        )

        f.write(
            "="*60+"\n\n"
        )


        for name,metrics in results.items():

            f.write(
                name+"\n"
            )

            f.write(
                "-"*30+"\n"
            )


            for k,v in metrics.items():

                f.write(
                    f"{k}: {v:.6f} m\n"
                )


            f.write(
                "\n"
            )



    print()

    print(
        "Saved:"
    )

    print(
        OUTPUT_FILE
    )





if __name__=="__main__":

    main()
