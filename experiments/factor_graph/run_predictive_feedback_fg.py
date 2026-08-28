"""
Predictive Reliability Factor Graph V3
======================================

Correct temporal order for factor at frame t:

1. Before using GPS(t), obtain predictive prior:
       R_pred(t | t-H)

2. Obtain an IMU-only one-step predicted position:
       p_imu(t)

3. GPS(t) arrives.

4. Compute innovation BEFORE GPS(t) enters the graph:
       || GPS(t) - p_imu(t) ||

5. Convert innovation to feedback reliability.

6. Fuse:
       predictive prior + innovation feedback

7. Map final reliability to covariance.

8. Add GPS(t) factor with that covariance.

No ground truth is used online.
"""

from __future__ import annotations

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
from src.reliability.reliability_mapper import (
    reliability_to_sigma,
)
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

PREDICTIVE_PRIOR_PATH = os.path.join(
    ROOT,
    "results",
    "predictive_reliability_v3",
    "predictive_prior_target_aligned.txt",
)

OUTPUT_DIR = os.path.join(
    ROOT,
    "results",
    "predictive_feedback_fg_v3",
)

COMPAT_DIR = os.path.join(
    ROOT,
    "results",
    "dynamic_covariance_fg",
)


DT = 0.1

FEEDBACK_SCALE = 8.0


def load_imu():
    loader = IMULoader(
        DATASET
    )

    acceleration = []
    gyro = []

    for i in range(
        len(loader)
    ):
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


def simple_imu_position_prediction(
    previous_position,
    previous_velocity,
    acceleration,
    dt,
):
    """
    Lightweight IMU-only one-step position prediction.

    The current GPS measurement is NOT used.
    This preserves the correct innovation order.
    """
    return (
        previous_position
        +
        previous_velocity
        *
        dt
        +
        0.5
        *
        acceleration
        *
        dt
        *
        dt
    )


def main():
    print("=" * 70)
    print(
        "Predictive Reliability + "
        "Pre-GPS IMU Innovation FG V3"
    )
    print("=" * 70)

    gps = np.loadtxt(
        GPS_PATH,
        dtype=np.float64,
    )

    predictive_prior = np.loadtxt(
        PREDICTIVE_PRIOR_PATH,
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
    predictive_prior = (
        predictive_prior[:n]
    )
    acceleration = (
        acceleration[:n]
    )
    gyro = gyro[:n]

    predictive_prior = np.clip(
        predictive_prior,
        0.0,
        1.0,
    )

    # ---------------------------------------------------------
    # Pre-GPS innovation feedback
    # ---------------------------------------------------------

    imu_prediction = np.zeros_like(
        gps,
        dtype=np.float64,
    )

    innovation = np.zeros(
        n,
        dtype=np.float64,
    )

    velocity_estimate = np.zeros(
        (
            n,
            3,
        ),
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

        velocity_estimate[1] = (
            gps[1]
            -
            gps[0]
        ) / DT

    for i in range(
        2,
        n,
    ):
        # Velocity estimate uses only measurements available
        # strictly before current GPS(i).
        previous_velocity = (
            gps[i - 1]
            -
            gps[i - 2]
        ) / DT

        velocity_estimate[
            i
        ] = previous_velocity

        imu_prediction[
            i
        ] = (
            simple_imu_position_prediction(
                gps[i - 1],
                previous_velocity,
                acceleration[i - 1],
                DT,
            )
        )

        innovation[
            i
        ] = np.linalg.norm(
            gps[i]
            -
            imu_prediction[i]
        )

    feedback = residual_reliability(
        innovation,
        scale=
            FEEDBACK_SCALE,
    )

    final_reliability, (
        prediction_weight
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

    # ---------------------------------------------------------
    # Build graph
    # ---------------------------------------------------------

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

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    os.makedirs(
        COMPAT_DIR,
        exist_ok=True,
    )

    outputs = {
        "trajectory.txt":
            trajectory,

        "predictive_prior.txt":
            predictive_prior,

        "imu_prediction.txt":
            imu_prediction,

        "innovation.txt":
            innovation,

        "feedback_reliability.txt":
            feedback,

        "prediction_weight.txt":
            prediction_weight,

        "final_reliability.txt":
            final_reliability,

        "sigma.txt":
            sigma,
    }

    for filename, values in (
        outputs.items()
    ):
        np.savetxt(
            os.path.join(
                OUTPUT_DIR,
                filename,
            ),
            values,
            fmt="%.8f",
        )

    np.savetxt(
        os.path.join(
            COMPAT_DIR,
            "trajectory.txt",
        ),
        trajectory,
        fmt="%.8f",
    )

    print(
        "Trajectory:",
        trajectory.shape,
    )

    print(
        "Saved:",
        OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
