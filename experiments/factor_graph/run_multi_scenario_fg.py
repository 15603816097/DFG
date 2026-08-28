"""
Multi-Degradation Factor Graph Robustness Experiment
====================================================

All proposed-method parameters are FROZEN:

    Horizon H = 3
    prediction weight = 1.0
    feedback weight = 0.0
    sigma_min = 3.0
    sigma_max = 30.0
    gamma = 3.0
    Huber delta = 1.345

For each degradation scenario, run:

1. Fixed FG
   fixed sigma = 5.0
   no robust kernel

2. Robust Fixed FG
   fixed sigma = 5.0
   Huber robust kernel

3. Proposed Predictive FG
   frozen H=3 Mamba predictive reliability
   dynamic covariance
   Huber robust kernel

4. Oracle FG
   GT-derived reliability
   dynamic covariance
   Huber robust kernel
   upper bound only
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


DATASET = os.path.join(
    ROOT,
    "dataset",
    "kitti",
    "2011_10_03",
    "2011_10_03_drive_0027_sync",
)

GT_PATH = os.path.join(
    ROOT,
    "results",
    "ground_truth",
    "trajectory.txt",
)

SCENARIO_ROOT = os.path.join(
    ROOT,
    "results",
    "multi_degradation",
)

SCENARIOS = [
    "mild_progressive",
    "medium_progressive",
    "severe_progressive",
    "sudden",
    "bias_drift",
    "intermittent_outlier",
]

METHODS = [
    "fixed",
    "robust_fixed",
    "proposed",
    "oracle",
]

DT = 0.1

FIXED_SIGMA = 5.0

SIGMA_MIN = 3.0

SIGMA_MAX = 30.0

GAMMA = 3.0

ORACLE_SCALE = 8.0


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
            item[
                "acceleration"
            ]
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


def reliability_to_sigma(
    reliability,
):
    r = np.clip(
        np.asarray(
            reliability,
            dtype=np.float64,
        ),
        0.0,
        1.0,
    )

    sigma = (
        SIGMA_MIN
        +
        (
            1.0
            -
            r
        )
        ** GAMMA
        *
        (
            SIGMA_MAX
            -
            SIGMA_MIN
        )
    )

    return np.clip(
        sigma,
        SIGMA_MIN,
        SIGMA_MAX,
    )


def oracle_reliability(
    gps,
    gt,
):
    error = np.linalg.norm(
        gps
        -
        gt,
        axis=1,
    )

    r = np.exp(
        -0.5
        *
        (
            error
            /
            ORACLE_SCALE
        )
        ** 2
    )

    return (
        np.clip(
            r,
            0.0,
            1.0,
        ),
        error,
    )


def optimize(
    gps,
    acceleration,
    gyro,
    sigma,
    robust,
    output_dir,
    title,
    extra=None,
):
    n = min(
        len(gps),
        len(acceleration),
        len(gyro),
        len(sigma),
    )

    gps = gps[:n]

    acceleration = (
        acceleration[:n]
    )

    gyro = gyro[:n]

    sigma = np.asarray(
        sigma,
        dtype=np.float64,
    )[:n]

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

    def make_gps_noise(
        s,
    ):
        base = (
            noiseModel.Isotropic.Sigma(
                3,
                float(s),
            )
        )

        if not robust:
            return base

        return (
            noiseModel.Robust.Create(
                noiseModel.mEstimator
                .Huber.Create(
                    1.345
                ),
                base,
            )
        )

    print("-" * 80)
    print(title)
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

        graph.add(
            GPSFactor(
                symbol(
                    "x",
                    i,
                ),
                Point3(
                    *gps[i]
                ),
                make_gps_noise(
                    sigma[i]
                ),
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

        if i % 1000 == 0:
            print(
                "Add factors:",
                i,
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
            make_gps_noise(
                sigma[-1]
            ),
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

    trajectory = np.asarray(
        [
            [
                *result
                .atPose3(
                    symbol(
                        "x",
                        i,
                    )
                )
                .translation()
            ]
            for i in range(n)
        ],
        dtype=np.float64,
    )

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    np.savetxt(
        os.path.join(
            output_dir,
            "trajectory.txt",
        ),
        trajectory,
        fmt="%.8f",
    )

    np.savetxt(
        os.path.join(
            output_dir,
            "sigma.txt",
        ),
        sigma,
        fmt="%.8f",
    )

    for filename, values in (
        extra or {}
    ).items():
        np.savetxt(
            os.path.join(
                output_dir,
                filename,
            ),
            np.asarray(
                values
            )[:n],
            fmt="%.8f",
        )

    print(
        "Saved:",
        output_dir,
    )


def main():
    acceleration, gyro = (
        load_imu()
    )

    gt = np.loadtxt(
        GT_PATH,
        dtype=np.float64,
    )

    print("=" * 96)
    print(
        "MULTI-DEGRADATION FACTOR GRAPH "
        "ROBUSTNESS EXPERIMENT"
    )
    print("=" * 96)

    print(
        "Frozen proposed parameters:"
    )

    print(
        "H=3, alpha=1.0, "
        "sigma_min=3, sigma_max=30, gamma=3"
    )

    for scenario in SCENARIOS:
        folder = os.path.join(
            SCENARIO_ROOT,
            scenario,
        )

        gps = np.loadtxt(
            os.path.join(
                folder,
                "gps_corrupted.txt",
            ),
            dtype=np.float64,
        )

        prior = np.loadtxt(
            os.path.join(
                folder,
                "predictive_prior.txt",
            ),
            dtype=np.float64,
        ).reshape(-1)

        n = min(
            len(gps),
            len(prior),
            len(acceleration),
            len(gyro),
            len(gt),
        )

        gps_s = gps[:n]

        acc_s = (
            acceleration[:n]
        )

        gyro_s = gyro[:n]

        gt_s = gt[:n]

        prior_s = np.clip(
            prior[:n],
            0.0,
            1.0,
        )

        print()
        print("=" * 96)
        print(
            "SCENARIO:",
            scenario,
        )
        print("=" * 96)

        # Fixed covariance baseline.
        optimize(
            gps_s,
            acc_s,
            gyro_s,
            np.full(
                n,
                FIXED_SIGMA,
                dtype=np.float64,
            ),
            robust=False,
            output_dir=os.path.join(
                folder,
                "fixed_fg",
            ),
            title=
                "Fixed FG",
        )

        # Robust fixed baseline.
        optimize(
            gps_s,
            acc_s,
            gyro_s,
            np.full(
                n,
                FIXED_SIGMA,
                dtype=np.float64,
            ),
            robust=True,
            output_dir=os.path.join(
                folder,
                "robust_fixed_fg",
            ),
            title=
                "Robust Fixed FG",
        )

        # Proposed method.
        proposed_sigma = (
            reliability_to_sigma(
                prior_s
            )
        )

        optimize(
            gps_s,
            acc_s,
            gyro_s,
            proposed_sigma,
            robust=True,
            output_dir=os.path.join(
                folder,
                "proposed_fg",
            ),
            title=
                "Proposed Predictive FG",
            extra={
                "predictive_reliability.txt":
                    prior_s,
            },
        )

        # Oracle upper bound.
        (
            oracle_r,
            gps_error,
        ) = oracle_reliability(
            gps_s,
            gt_s,
        )

        oracle_sigma = (
            reliability_to_sigma(
                oracle_r
            )
        )

        optimize(
            gps_s,
            acc_s,
            gyro_s,
            oracle_sigma,
            robust=True,
            output_dir=os.path.join(
                folder,
                "oracle_fg",
            ),
            title=
                "Oracle FG",
            extra={
                "oracle_reliability.txt":
                    oracle_r,
                "gps_error.txt":
                    gps_error,
            },
        )


if __name__ == "__main__":
    main()
