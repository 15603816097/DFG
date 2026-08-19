"""
Trajectory Evaluation

Compare:

1. IMU Dead Reckoning
2. ESKF
3. Factor Graph


Metrics:

- ATE 3D RMSE
- ATE 2D RMSE
- Mean Error
- Maximum Error
- Z RMSE


Coordinate:
ENU
"""


import os
import numpy as np





# =====================================================
# Project Path
# =====================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)



RESULT_DIR = os.path.join(
    PROJECT_ROOT,
    "results"
)



# Ground Truth

GT_FILE = os.path.join(
    RESULT_DIR,
    "ground_truth",
    "trajectory.txt"
)



# Methods

METHODS = {


    "IMU Dead Reckoning":

    os.path.join(
        RESULT_DIR,
        "imu_baseline",
        "trajectory.txt"
    ),



    "ESKF":

    os.path.join(
        RESULT_DIR,
        "eskf",
        "trajectory.txt"
    ),



    "Factor Graph":

    os.path.join(
        RESULT_DIR,
        "factor_graph",
        "trajectory.txt"
    )

}





OUTPUT_FILE = os.path.join(
    RESULT_DIR,
    "evaluation_all.txt"
)






# =====================================================
# Load trajectory
# =====================================================

def load_trajectory(path):


    if not os.path.exists(path):

        raise FileNotFoundError(
            path
        )


    trajectory = np.loadtxt(
        path
    )


    if trajectory.ndim != 2:

        raise RuntimeError(
            "Trajectory format error"
        )


    if trajectory.shape[1] != 3:

        raise RuntimeError(
            "Trajectory must be Nx3"
        )


    return trajectory







# =====================================================
# Metrics
# =====================================================

def compute_metrics(
    estimate,
    ground_truth
):


    if estimate.shape != ground_truth.shape:

        raise RuntimeError(
            f"Shape mismatch: "
            f"{estimate.shape} vs {ground_truth.shape}"
        )



    # position error

    error = (
        estimate
        -
        ground_truth
    )



    # -----------------------------
    # 3D error
    # -----------------------------

    error_3d = np.linalg.norm(
        error,
        axis=1
    )


    ate_3d_rmse = np.sqrt(
        np.mean(
            error_3d ** 2
        )
    )


    mean_3d = np.mean(
        error_3d
    )


    max_3d = np.max(
        error_3d
    )




    # -----------------------------
    # 2D XY error
    # -----------------------------

    error_2d = np.linalg.norm(
        error[:, :2],
        axis=1
    )


    ate_2d_rmse = np.sqrt(
        np.mean(
            error_2d ** 2
        )
    )


    mean_2d = np.mean(
        error_2d
    )


    max_2d = np.max(
        error_2d
    )




    # -----------------------------
    # Z error
    # -----------------------------

    z_error = np.abs(
        error[:,2]
    )


    z_rmse = np.sqrt(
        np.mean(
            z_error ** 2
        )
    )


    return {


        "ATE_3D_RMSE(m)":
        ate_3d_rmse,


        "ATE_2D_RMSE(m)":
        ate_2d_rmse,


        "Mean_3D_Error(m)":
        mean_3d,


        "Max_3D_Error(m)":
        max_3d,


        "Mean_2D_Error(m)":
        mean_2d,


        "Max_2D_Error(m)":
        max_2d,


        "Z_RMSE(m)":
        z_rmse

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



    gt = load_trajectory(
        GT_FILE
    )



    print(
        "Ground Truth:",
        gt.shape
    )



    all_results = {}



    for name,path in METHODS.items():


        print()
        print(
            "-"*60
        )

        print(
            name
        )


        trajectory = load_trajectory(
            path
        )


        print(
            "Trajectory:",
            trajectory.shape
        )



        metrics = compute_metrics(
            trajectory,
            gt
        )


        all_results[name]=metrics



        for key,value in metrics.items():

            print(
                f"{key:<25}: {value:.6f}"
            )






    # =================================================
    # Save
    # =================================================


    with open(
        OUTPUT_FILE,
        "w"
    ) as f:


        f.write(
            "Trajectory Evaluation Results\n"
        )


        f.write(
            "="*70+"\n\n"
        )



        for name,metrics in all_results.items():


            f.write(
                name+"\n"
            )


            f.write(
                "-"*40+"\n"
            )


            for key,value in metrics.items():


                f.write(
                    f"{key}: {value:.6f}\n"
                )


            f.write(
                "\n"
            )



    print()
    print(
        "="*60
    )

    print(
        "Saved:"
    )


    print(
        OUTPUT_FILE
    )

    print(
        "="*60
    )





if __name__ == "__main__":

    main()
