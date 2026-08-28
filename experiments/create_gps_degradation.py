"""
Generate GPS degradation data

Simulation:

1. Normal GPS
    small Gaussian noise


2. GPS degradation
    continuous bias


3. Recovery stage
    bias gradually decreases


Used for:

Dynamic Covariance Factor Graph experiment

"""


import os
import sys

import numpy as np



ROOT=os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


sys.path.append(ROOT)



GT_PATH=os.path.join(
    ROOT,
    "results",
    "ground_truth",
    "trajectory.txt"
)



OUTPUT=os.path.join(
    ROOT,
    "results",
    "gps_degradation",
    "gps_corrupted.txt"
)




def main():


    print("="*60)

    print(
        "Generate GPS Degradation"
    )

    print("="*60)



    gps=np.loadtxt(
        GT_PATH
    )


    N=len(gps)



    print(
        "Ground Truth:",
        gps.shape
    )



    corrupted=gps.copy()



    np.random.seed(42)



    print(
        "Inject GPS degradation..."
    )



    for i in range(N):


        # =========================
        # Normal GPS
        # =========================

        noise=np.random.normal(
            0,
            1.0,
            3
        )


        corrupted[i]+=noise



        # =========================
        # GPS degradation stage
        # 1000-2500 frames
        # =========================


        if 1000 <= i < 2500:


            bias=np.array(
                [
                    50.0,
                    50.0,
                    0.0
                ]
            )


            corrupted[i]+=bias



        # =========================
        # Recovery stage
        # 2500-3500
        # =========================


        elif 2500 <= i < 3500:


            ratio=(3500-i)/1000


            bias=np.array(
                [
                    50.0*ratio,
                    50.0*ratio,
                    0.0
                ]
            )


            corrupted[i]+=bias




    os.makedirs(
        os.path.dirname(OUTPUT),
        exist_ok=True
    )



    np.savetxt(
        OUTPUT,
        corrupted,
        fmt="%.6f"
    )



    print()

    print(
        "Generated:",
        corrupted.shape
    )


    print(
        "Saved:",
        OUTPUT
    )



    # =========================
    # statistics
    # =========================


    error=np.linalg.norm(
        corrupted-gps,
        axis=1
    )


    print()

    print(
        "GPS error statistics:"
    )


    print(
        "Mean:",
        np.mean(error),
        "m"
    )


    print(
        "Max:",
        np.max(error),
        "m"
    )


    print(
        "Degradation frames:",
        "1000-3500"
    )




if __name__=="__main__":

    main()
