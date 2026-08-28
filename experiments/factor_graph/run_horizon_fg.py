"""
Run the predictive reliability factor graph for one horizon.

Example
-------
python experiments/factor_graph/run_horizon_fg.py --horizon 5
"""

from __future__ import annotations

import argparse
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
    noiseModel,
)


ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

if ROOT not in sys.path:
    sys.path.insert(
        0,
        ROOT,
    )


from src.loader.imu_loader import IMULoader
from src.reliability.reliability_mapper import reliability_to_sigma
from src.reliability.feedback_corrector import (
    residual_reliability,
    adaptive_fuse_reliability,
)


DATASET = os.path.join(
    ROOT,
    "dataset",
    "kitti",
    "2011_10_03",
    "2011_10_03_drive_0027_sync",
)

GPS_PATH = os.path.join(
    ROOT,
    "results",
    "progressive_gps_degradation",
    "gps_corrupted.txt",
)

DT = 0.1
FEEDBACK_SCALE = 8.0


def load_imu():
    loader = IMULoader(
        DATASET
    )

    acceleration = []
    gyro = []

    for i in range(len(loader)):
        item = loader[i]

        acceleration.append(
            item["acceleration"]
        )

        gyro.append(
            item[
                "angular_velocity"
            ]
        )

    return (
        np.asarray(
            acceleration,
            dtype=np.float64,
        ),
        np.asarray(
            gyro,
            dtype=np.float64,
        ),
    )


def create_params():
    params = (
        PreintegrationParams
        .MakeSharedU(
            9.81
        )
    )

    params.setAccelerometerCovariance(
        np.eye(3)
        *
        0.1
    )

    params.setGyroscopeCovariance(
        np.eye(3)
        *
        0.01
    )

    params.setIntegrationCovariance(
        np.eye(3)
        *
        0.001
    )

    return params


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--horizon",
        type=int,
        required=True,
    )

    args = parser.parse_args()

    horizon = int(
        args.horizon
    )

    result_dir = os.path.join(
        ROOT,
        "results",
        "horizon_sensitivity",
        f"h{horizon}",
    )

    prior_path = os.path.join(
        result_dir,
        "predictive_prior_target_aligned.txt",
    )

    if not os.path.exists(
        prior_path
    ):
        raise FileNotFoundError(
            prior_path
        )

    gps = np.loadtxt(
        GPS_PATH,
        dtype=np.float64,
    )

    predictive_prior = np.loadtxt(
        prior_path,
        dtype=np.float64,
    ).reshape(-1)

    acceleration, gyro = (
        load_imu()
    )

    n = min(
        len(gps),
        len(predictive_prior),
        len(acceleration),
        len(gyro),
    )

    gps = gps[:n]
    predictive_prior = np.clip(
        predictive_prior[:n],
        0.0,
        1.0,
    )
    acceleration = (
        acceleration[:n]
    )
    gyro = gyro[:n]

    # ---------------------------------------------------------
    # Pre-GPS causal innovation feedback.
    # Current GPS(t) is not used to construct the prediction.
    # ---------------------------------------------------------

    imu_prediction = np.zeros_like(
        gps
    )

    innovation = np.zeros(
        n,
        dtype=np.float64,
    )

    imu_prediction[0] = (
        gps[0]
    )

    if n > 1:
        imu_prediction[1] = (
            gps[0]
            +
            0.5
            *
            acceleration[0]
            *
            DT
            *
            DT
        )

        innovation[1] = np.linalg.norm(
            gps[1]
            -
            imu_prediction[1]
        )

    for i in range(
        2,
        n,
    ):
        previous_velocity = (
            gps[i - 1]
            -
            gps[i - 2]
        ) / DT

        imu_prediction[i] = (
            gps[i - 1]
            +
            previous_velocity
            *
            DT
            +
            0.5
            *
            acceleration[i - 1]
            *
            DT
            *
            DT
        )

        innovation[i] = np.linalg.norm(
            gps[i]
            -
            imu_prediction[i]
        )

    feedback = residual_reliability(
        innovation,
        scale=
            FEEDBACK_SCALE,
    )

    (
        final_reliability,
        prediction_weight,
    ) = adaptive_fuse_reliability(
        predictive_prior,
        feedback,
        base_prediction_weight=0.75,
        min_prediction_weight=0.60,
        max_prediction_weight=0.90,
    )

    sigma = reliability_to_sigma(
        final_reliability,
        sigma_min=3.0,
        sigma_max=20.0,
        gamma=3.0,
    )

    print("=" * 72)
    print(
        "Horizon FG:",
        horizon,
        "frames",
    )
    print(
        "Predictive prior mean:",
        float(
            predictive_prior.mean()
        ),
    )
    print(
        "Feedback mean:",
        float(
            feedback.mean()
        ),
    )
    print(
        "Final reliability mean:",
        float(
            final_reliability.mean()
        ),
    )
    print(
        "Sigma min/max/mean:",
        float(
            sigma.min()
        ),
        float(
            sigma.max()
        ),
        float(
            sigma.mean()
        ),
    )
    print("=" * 72)

    graph = (
        NonlinearFactorGraph()
    )

    initial = Values()

    params = create_params()

    bias0 = (
        gtsam.imuBias
        .ConstantBias()
    )

    pose_noise = (
        noiseModel.Isotropic.Sigma(
            6,
            0.1,
        )
    )

    velocity_noise = (
        noiseModel.Isotropic.Sigma(
            3,
            1.0,
        )
    )

    bias_noise = (
        noiseModel.Isotropic.Sigma(
            6,
            1e-3,
        )
    )

    pose0 = Pose3(
        Rot3(),
        Point3(
            *gps[0]
        ),
    )

    graph.add(
        PriorFactorPose3(
            symbol("x", 0),
            pose0,
            pose_noise,
        )
    )

    graph.add(
        PriorFactorVector(
            symbol("v", 0),
            np.zeros(3),
            velocity_noise,
        )
    )

    initial.insert(
        symbol("x", 0),
        pose0,
    )

    initial.insert(
        symbol("v", 0),
        np.zeros(3),
    )

    initial.insert(
        symbol("b", 0),
        bias0,
    )

    for i in range(
        n - 1
    ):
        pim = (
            PreintegratedImuMeasurements(
                params,
                bias0,
            )
        )

        pim.integrateMeasurement(
            acceleration[i],
            gyro[i],
            DT,
        )

        graph.add(
            ImuFactor(
                symbol("x", i),
                symbol("v", i),
                symbol(
                    "x",
                    i + 1,
                ),
                symbol(
                    "v",
                    i + 1,
                ),
                symbol("b", i),
                pim,
            )
        )

        graph.add(
            BetweenFactorConstantBias(
                symbol("b", i),
                symbol(
                    "b",
                    i + 1,
                ),
                gtsam.imuBias
                .ConstantBias(),
                bias_noise,
            )
        )

        base = (
            noiseModel.Isotropic.Sigma(
                3,
                float(
                    sigma[i]
                ),
            )
        )

        robust = (
            noiseModel.Robust.Create(
                noiseModel.mEstimator
                .Huber.Create(
                    1.345
                ),
                base,
            )
        )

        graph.add(
            GPSFactor(
                symbol("x", i),
                Point3(
                    *gps[i]
                ),
                robust,
            )
        )

        initial.insert(
            symbol(
                "x",
                i + 1,
            ),
            Pose3(
                Rot3(),
                Point3(
                    *gps[
                        i + 1
                    ]
                ),
            ),
        )

        initial.insert(
            symbol(
                "v",
                i + 1,
            ),
            np.zeros(3),
        )

        initial.insert(
            symbol(
                "b",
                i + 1,
            ),
            bias0,
        )

        if i % 500 == 0:
            print(
                "Add factors:",
                i,
            )

    last_base = (
        noiseModel.Isotropic.Sigma(
            3,
            float(
                sigma[-1]
            ),
        )
    )

    last_robust = (
        noiseModel.Robust.Create(
            noiseModel.mEstimator
            .Huber.Create(
                1.345
            ),
            last_base,
        )
    )

    graph.add(
        GPSFactor(
            symbol(
                "x",
                n - 1,
            ),
            Point3(
                *gps[-1]
            ),
            last_robust,
        )
    )

    print(
        "Graph size:",
        graph.size(),
    )

    print(
        "Optimizing..."
    )

    result = (
        LevenbergMarquardtOptimizer(
            graph,
            initial,
        ).optimize()
    )

    trajectory = []

    for i in range(n):
        t = (
            result
            .atPose3(
                symbol(
                    "x",
                    i,
                )
            )
            .translation()
        )

        trajectory.append(
            [
                float(t[0]),
                float(t[1]),
                float(t[2]),
            ]
        )

    trajectory = np.asarray(
        trajectory,
        dtype=np.float64,
    )

    np.savetxt(
        os.path.join(
            result_dir,
            "fg_trajectory.txt",
        ),
        trajectory,
        fmt="%.8f",
    )

    np.savetxt(
        os.path.join(
            result_dir,
            "feedback_reliability.txt",
        ),
        feedback,
        fmt="%.8f",
    )

    np.savetxt(
        os.path.join(
            result_dir,
            "final_reliability.txt",
        ),
        final_reliability,
        fmt="%.8f",
    )

    np.savetxt(
        os.path.join(
            result_dir,
            "prediction_weight.txt",
        ),
        prediction_weight,
        fmt="%.8f",
    )

    np.savetxt(
        os.path.join(
            result_dir,
            "innovation.txt",
        ),
        innovation,
        fmt="%.8f",
    )

    np.savetxt(
        os.path.join(
            result_dir,
            "sigma.txt",
        ),
        sigma,
        fmt="%.8f",
    )

    print(
        "Saved:",
        result_dir,
    )


if __name__ == "__main__":
    main()
