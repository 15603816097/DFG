"""
Prediction / Feedback Weight Sensitivity Experiment
===================================================

Fixed:
    Horizon H = 3 frames
    Progressive GPS degradation
    Dual-head Mamba prediction
    Pre-GPS innovation feedback
    Covariance mapping:
        sigma_min = 3
        sigma_max = 20
        gamma = 3

Only variable:
    alpha = prediction weight

Final reliability:
    R_final =
        alpha * R_prediction
        +
        (1 - alpha) * R_feedback

Example
-------
python experiments/factor_graph/run_weight_fg.py --alpha 0.8
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
from src.reliability.reliability_mapper import (
    reliability_to_sigma,
)
from src.reliability.feedback_corrector import (
    residual_reliability,
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

HORIZON = 3

PREDICTIVE_PRIOR_PATH = os.path.join(
    ROOT,
    "results",
    "horizon_sensitivity",
    f"h{HORIZON}",
    "predictive_prior_target_aligned.txt",
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


def build_pre_gps_feedback(
    gps,
    acceleration,
):
    """
    Build causal innovation feedback.

    GPS(t) is NOT used to construct the predicted position at t.

    Prediction uses:
        GPS(t-1)
        GPS(t-2)
        IMU acceleration(t-1)

    Then:
        innovation(t)
        =
        ||GPS(t) - p_pred(t)||

    This is the same feedback definition used in the H=3 V3 experiment.
    """
    n = min(
        len(gps),
        len(acceleration),
    )

    gps = gps[:n]
    acceleration = (
        acceleration[:n]
    )

    predicted_position = (
        np.zeros_like(
            gps,
            dtype=np.float64,
        )
    )

    innovation = np.zeros(
        n,
        dtype=np.float64,
    )

    predicted_position[0] = (
        gps[0]
    )

    if n > 1:
        predicted_position[1] = (
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

        innovation[1] = (
            np.linalg.norm(
                gps[1]
                -
                predicted_position[1]
            )
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

        predicted_position[i] = (
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

        innovation[i] = (
            np.linalg.norm(
                gps[i]
                -
                predicted_position[i]
            )
        )

    feedback = (
        residual_reliability(
            innovation,
            scale=
                FEEDBACK_SCALE,
        )
    )

    return (
        predicted_position,
        innovation,
        feedback,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--alpha",
        type=float,
        required=True,
        help=(
            "Prediction weight. "
            "Feedback weight = 1-alpha."
        ),
    )

    args = parser.parse_args()

    alpha = float(
        args.alpha
    )

    if not (
        0.0
        <=
        alpha
        <=
        1.0
    ):
        raise ValueError(
            "alpha must be in [0, 1]"
        )

    if not os.path.exists(
        PREDICTIVE_PRIOR_PATH
    ):
        raise FileNotFoundError(
            "H=3 predictive prior not found:\n"
            f"{PREDICTIVE_PRIOR_PATH}\n\n"
            "Run first:\n"
            "python experiments/"
            "run_horizon_sensitivity.py"
        )

    gps = np.loadtxt(
        GPS_PATH,
        dtype=np.float64,
    )

    predictive_prior = (
        np.loadtxt(
            PREDICTIVE_PRIOR_PATH,
            dtype=np.float64,
        )
        .reshape(-1)
    )

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

    (
        imu_prediction,
        innovation,
        feedback,
    ) = build_pre_gps_feedback(
        gps,
        acceleration,
    )

    final_reliability = np.clip(
        alpha
        *
        predictive_prior
        +
        (
            1.0
            -
            alpha
        )
        *
        feedback,
        0.0,
        1.0,
    )

    sigma = reliability_to_sigma(
        final_reliability,
        sigma_min=3.0,
        sigma_max=20.0,
        gamma=3.0,
    )

    alpha_tag = (
        f"{alpha:.2f}"
        .replace(
            ".",
            "p",
        )
    )

    result_dir = os.path.join(
        ROOT,
        "results",
        "weight_sensitivity",
        f"alpha_{alpha_tag}",
    )

    os.makedirs(
        result_dir,
        exist_ok=True,
    )

    print("=" * 76)
    print(
        "Prediction / Feedback "
        "Weight Sensitivity"
    )
    print("=" * 76)

    print(
        "Horizon:",
        HORIZON,
        "frames"
    )

    print(
        "Prediction weight alpha:",
        alpha,
    )

    print(
        "Feedback weight:",
        1.0 - alpha,
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

    # =========================================================
    # Build factor graph
    # =========================================================

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
            symbol(
                "x",
                0,
            ),
            pose0,
            pose_noise,
        )
    )

    graph.add(
        PriorFactorVector(
            symbol(
                "v",
                0,
            ),
            np.zeros(3),
            velocity_noise,
        )
    )

    initial.insert(
        symbol(
            "x",
            0,
        ),
        pose0,
    )

    initial.insert(
        symbol(
            "v",
            0,
        ),
        np.zeros(3),
    )

    initial.insert(
        symbol(
            "b",
            0,
        ),
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
                symbol(
                    "x",
                    i,
                ),
                symbol(
                    "v",
                    i,
                ),
                symbol(
                    "x",
                    i + 1,
                ),
                symbol(
                    "v",
                    i + 1,
                ),
                symbol(
                    "b",
                    i,
                ),
                pim,
            )
        )

        graph.add(
            BetweenFactorConstantBias(
                symbol(
                    "b",
                    i,
                ),
                symbol(
                    "b",
                    i + 1,
                ),
                gtsam.imuBias
                .ConstantBias(),
                bias_noise,
            )
        )

        base_noise = (
            noiseModel.Isotropic.Sigma(
                3,
                float(
                    sigma[i]
                ),
            )
        )

        robust_noise = (
            noiseModel.Robust.Create(
                noiseModel.mEstimator
                .Huber.Create(
                    1.345
                ),
                base_noise,
            )
        )

        graph.add(
            GPSFactor(
                symbol(
                    "x",
                    i,
                ),
                Point3(
                    *gps[i]
                ),
                robust_noise,
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

    last_base_noise = (
        noiseModel.Isotropic.Sigma(
            3,
            float(
                sigma[-1]
            ),
        )
    )

    last_robust_noise = (
        noiseModel.Robust.Create(
            noiseModel.mEstimator
            .Huber.Create(
                1.345
            ),
            last_base_noise,
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
            last_robust_noise,
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
        )
        .optimize()
    )

    trajectory = []

    for i in range(n):
        translation = (
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
                float(
                    translation[0]
                ),
                float(
                    translation[1]
                ),
                float(
                    translation[2]
                ),
            ]
        )

    trajectory = np.asarray(
        trajectory,
        dtype=np.float64,
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
                result_dir,
                filename,
            ),
            values,
            fmt="%.8f",
        )

    with open(
        os.path.join(
            result_dir,
            "config.txt",
        ),
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            f"horizon={HORIZON}\n"
        )

        file.write(
            f"alpha={alpha:.8f}\n"
        )

        file.write(
            f"feedback_weight="
            f"{1.0-alpha:.8f}\n"
        )

        file.write(
            "sigma_min=3.0\n"
        )

        file.write(
            "sigma_max=20.0\n"
        )

        file.write(
            "gamma=3.0\n"
        )

        file.write(
            f"feedback_scale="
            f"{FEEDBACK_SCALE}\n"
        )

    print(
        "Trajectory:",
        trajectory.shape,
    )

    print(
        "Saved:",
        result_dir,
    )


if __name__ == "__main__":
    main()
