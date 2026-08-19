"""
Trajectory Visualization

Generate:

1. Global trajectory comparison:
    Ground Truth
    IMU Dead Reckoning
    ESKF


2. Local ESKF comparison:
    Ground Truth
    ESKF


Coordinate:
    ENU
"""


import os
import numpy as np
import matplotlib.pyplot as plt



# =====================================================
# Project root
# =====================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)



# =====================================================
# Files
# =====================================================

GT_FILE = os.path.join(
    PROJECT_ROOT,
    "results",
    "ground_truth",
    "trajectory.txt"
)


IMU_FILE = os.path.join(
    PROJECT_ROOT,
    "results",
    "imu_baseline",
    "trajectory.txt"
)


ESKF_FILE = os.path.join(
    PROJECT_ROOT,
    "results",
    "eskf",
    "trajectory.txt"
)



RESULT_DIR = os.path.join(
    PROJECT_ROOT,
    "results"
)


GLOBAL_FIG = os.path.join(
    RESULT_DIR,
    "trajectory_global.png"
)


ESKF_FIG = os.path.join(
    RESULT_DIR,
    "trajectory_eskf_local.png"
)





# =====================================================
# Load
# =====================================================

def load_trajectory(path):

    if not os.path.exists(path):

        raise FileNotFoundError(
            path
        )


    return np.loadtxt(
        path
    )





# =====================================================
# Global Plot
# =====================================================

def plot_global(
    gt,
    imu,
    eskf
):

    plt.figure(
        figsize=(10,8)
    )


    plt.plot(
        gt[:,0],
        gt[:,1],
        label="Ground Truth"
    )


    plt.plot(
        imu[:,0],
        imu[:,1],
        label="IMU Dead Reckoning"
    )


    plt.plot(
        eskf[:,0],
        eskf[:,1],
        label="ESKF"
    )



    plt.scatter(
        gt[0,0],
        gt[0,1],
        marker="o",
        label="Start"
    )


    plt.scatter(
        gt[-1,0],
        gt[-1,1],
        marker="x",
        label="End"
    )



    plt.xlabel(
        "East (m)"
    )


    plt.ylabel(
        "North (m)"
    )


    plt.title(
        "Global Trajectory Comparison"
    )


    plt.grid(
        True
    )


    plt.legend()



    plt.savefig(
        GLOBAL_FIG,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()



# =====================================================
# ESKF Local Plot
# =====================================================

def plot_eskf_local(
    gt,
    eskf
):


    plt.figure(
        figsize=(8,8)
    )


    plt.plot(
        gt[:,0],
        gt[:,1],
        label="Ground Truth"
    )


    plt.plot(
        eskf[:,0],
        eskf[:,1],
        label="ESKF"
    )



    plt.scatter(
        gt[0,0],
        gt[0,1],
        marker="o",
        label="Start"
    )


    plt.scatter(
        gt[-1,0],
        gt[-1,1],
        marker="x",
        label="End"
    )



    plt.xlabel(
        "East (m)"
    )


    plt.ylabel(
        "North (m)"
    )


    plt.title(
        "Ground Truth vs ESKF"
    )



    plt.axis(
        "equal"
    )


    plt.grid(
        True
    )


    plt.legend()



    plt.savefig(
        ESKF_FIG,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()





# =====================================================
# Main
# =====================================================

def main():

    print("="*60)
    print("Trajectory Visualization")
    print("="*60)



    gt = load_trajectory(
        GT_FILE
    )


    imu = load_trajectory(
        IMU_FILE
    )


    eskf = load_trajectory(
        ESKF_FILE
    )



    print(
        "GT:",
        gt.shape
    )

    print(
        "IMU:",
        imu.shape
    )

    print(
        "ESKF:",
        eskf.shape
    )



    plot_global(
        gt,
        imu,
        eskf
    )


    plot_eskf_local(
        gt,
        eskf
    )



    print()
    print(
        "Generated:"
    )


    print(
        GLOBAL_FIG
    )


    print(
        ESKF_FIG
    )





if __name__ == "__main__":

    main()
