"""
Predictive Mamba Dynamic Factor Graph V2
========================================

Uses both outputs from the multi-task predictive Mamba model:

    predicted_state_h5.txt
        0 = Unreliable
        1 = Degrading
        2 = Reliable

    predicted_reliability_h5.txt
        continuous future reliability

The discrete state decides graph topology.

The continuous reliability controls covariance inside the
Degrading state.

No Ground Truth is used here.

Primary output:
    results/predictive_mamba_dynamic_fg_v2/trajectory.txt

Evaluator compatibility:
    results/dynamic_covariance_fg/trajectory.txt
"""

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
    BetweenFactorPose3,
    BetweenFactorConstantBias,
    PreintegratedImuMeasurements,
    PreintegrationParams,
    noiseModel,
)


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(
            __file__
        )
    )
)


if PROJECT_ROOT not in sys.path:

    sys.path.insert(
        0,
        PROJECT_ROOT,
    )


from src.loader.imu_loader import IMULoader


# ============================================================
# Paths
# ============================================================

DATASET = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "kitti",
    "2011_10_03",
    "2011_10_03_drive_0027_sync",
)


GPS_PATH = os.path.join(
    PROJECT_ROOT,
    "results",
    "gps_degradation",
    "gps_corrupted.txt",
)


MODEL_RESULT_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "predictive_reliability_model_v2",
)


PREDICTED_RELIABILITY_PATH = os.path.join(
    MODEL_RESULT_DIR,
    "predicted_reliability_h5.txt",
)


PREDICTED_STATE_PATH = os.path.join(
    MODEL_RESULT_DIR,
    "predicted_state_h5.txt",
)


PREDICTED_PROBABILITY_PATH = os.path.join(
    MODEL_RESULT_DIR,
    "predicted_state_probability_h5.txt",
)


RESULT_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "predictive_mamba_dynamic_fg_v2",
)


TRAJECTORY_PATH = os.path.join(
    RESULT_DIR,
    "trajectory.txt",
)


RELIABILITY_PATH = os.path.join(
    RESULT_DIR,
    "predictive_reliability.txt",
)


STATE_PATH = os.path.join(
    RESULT_DIR,
    "predictive_state.txt",
)


PROBABILITY_PATH = os.path.join(
    RESULT_DIR,
    "predictive_state_probability.txt",
)


SIGMA_PATH = os.path.join(
    RESULT_DIR,
    "sigma.txt",
)


GPS_MASK_PATH = os.path.join(
    RESULT_DIR,
    "gps_mask.txt",
)


SOFT_ANCHOR_MASK_PATH = os.path.join(
    RESULT_DIR,
    "soft_anchor_mask.txt",
)


INITIAL_TRAJECTORY_PATH = os.path.join(
    RESULT_DIR,
    "initial_trajectory.txt",
)


COMPAT_RESULT_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "dynamic_covariance_fg",
)


COMPAT_TRAJECTORY_PATH = os.path.join(
    COMPAT_RESULT_DIR,
    "trajectory.txt",
)


# ============================================================
# Parameters
# ============================================================

DT = 0.1


GPS_SIGMA_HIGH = 3.0


GPS_SIGMA_MEDIUM_MIN = 5.0


GPS_SIGMA_MEDIUM_MAX = 20.0


HUBER_K = 1.345


SOFT_ANCHOR_SIGMA = 8.0


SOFT_ANCHOR_INTERVAL = 5


RELATIVE_FACTOR_INTERVAL = 5


RELATIVE_TRANSLATION_SIGMA_XY = 2.0


RELATIVE_TRANSLATION_SIGMA_Z = 3.0


RELATIVE_ROTATION_SIGMA = 1000.0


# ============================================================
# Load data
# ============================================================

def load_data():

    required_files = [

        DATASET,

        GPS_PATH,

        PREDICTED_RELIABILITY_PATH,

        PREDICTED_STATE_PATH,

        PREDICTED_PROBABILITY_PATH,

    ]


    for path in required_files:

        if not os.path.exists(
            path
        ):

            raise FileNotFoundError(
                path
            )


    imu_loader = IMULoader(
        DATASET
    )


    N = len(
        imu_loader
    )


    acc = []

    gyro = []


    for i in range(
        N
    ):

        item = imu_loader[
            i
        ]


        acc.append(
            item[
                "acceleration"
            ]
        )


        gyro.append(
            item[
                "angular_velocity"
            ]
        )


    acc = np.asarray(
        acc,
        dtype=np.float64,
    )


    gyro = np.asarray(
        gyro,
        dtype=np.float64,
    )


    gps = np.asarray(

        np.loadtxt(
            GPS_PATH,
            dtype=np.float64,
        ),

        dtype=np.float64,

    ).reshape(
        -1,
        3,
    )


    reliability = np.asarray(

        np.loadtxt(
            PREDICTED_RELIABILITY_PATH,
            dtype=np.float64,
        ),

        dtype=np.float64,

    ).reshape(
        -1
    )


    state = np.asarray(

        np.loadtxt(
            PREDICTED_STATE_PATH,
            dtype=np.int64,
        ),

        dtype=np.int64,

    ).reshape(
        -1
    )


    probability = np.asarray(

        np.loadtxt(
            PREDICTED_PROBABILITY_PATH,
            dtype=np.float64,
        ),

        dtype=np.float64,

    ).reshape(
        -1,
        3,
    )


    if len(
        gps
    ) != N:

        raise RuntimeError(
            "GPS frame count mismatch"
        )


    if len(
        reliability
    ) != N:

        raise RuntimeError(
            "Reliability frame count mismatch"
        )


    if len(
        state
    ) != N:

        raise RuntimeError(
            "State frame count mismatch"
        )


    if len(
        probability
    ) != N:

        raise RuntimeError(
            "Probability frame count mismatch"
        )


    reliability = np.clip(

        np.nan_to_num(
            reliability,
            nan=0.5,
            posinf=1.0,
            neginf=0.0,
        ),

        0.0,

        1.0,

    )


    state = np.clip(
        state,
        0,
        2,
    ).astype(
        np.int32
    )


    return (

        acc,

        gyro,

        gps,

        reliability,

        state,

        probability,

    )


# ============================================================
# State -> covariance
# ============================================================

def calculate_factor_parameters(
    reliability,
    state,
):

    N = len(
        reliability
    )


    sigma = np.zeros(
        N,
        dtype=np.float64,
    )


    mask = np.asarray(
        state,
        dtype=np.int32,
    ).copy()


    for i in range(
        N
    ):

        current_state = int(
            mask[
                i
            ]
        )


        current_reliability = float(
            reliability[
                i
            ]
        )


        if current_state == 2:

            sigma[
                i
            ] = (
                GPS_SIGMA_HIGH
            )


        elif current_state == 1:

            # Continuous dynamic covariance only inside
            # the Degrading state.

            normalized = np.clip(

                (
                    0.80
                    -
                    current_reliability
                )
                /
                0.60,

                0.0,

                1.0,

            )


            sigma[
                i
            ] = (

                GPS_SIGMA_MEDIUM_MIN

                +

                normalized

                *

                (
                    GPS_SIGMA_MEDIUM_MAX
                    -
                    GPS_SIGMA_MEDIUM_MIN
                )

            )


        else:

            sigma[
                i
            ] = (
                GPS_SIGMA_MEDIUM_MAX
            )


    return (
        sigma,
        mask,
    )


# ============================================================
# Initial trajectory
# ============================================================

def build_gated_initial_trajectory(
    gps,
    mask,
):

    gps = np.asarray(
        gps,
        dtype=np.float64,
    )


    N = len(
        gps
    )


    initial = gps.copy()


    bad = (
        mask
        ==
        0
    )


    i = 0


    while i < N:

        if not bad[
            i
        ]:

            i += 1

            continue


        start = i


        while (
            i + 1 < N
            and
            bad[
                i + 1
            ]
        ):

            i += 1


        end = i


        left = (
            start
            -
            1
        )


        right = (
            end
            +
            1
        )


        if (
            left >= 0
            and
            right < N
        ):

            segment_length = (
                end
                -
                start
                +
                1
            )


            if left > 0:

                start_delta = (
                    gps[
                        left
                    ]
                    -
                    gps[
                        left - 1
                    ]
                )


            else:

                start_delta = np.zeros(
                    3,
                    dtype=np.float64,
                )


            start_position = (
                initial[
                    left
                ]
                +
                start_delta
            )


            relative = (
                gps[
                    start:
                    end + 1
                ]
                -
                gps[
                    start
                ]
            )


            base = (
                start_position[
                    None,
                    :
                ]
                +
                relative
            )


            if right + 1 < N:

                end_delta = (
                    gps[
                        right + 1
                    ]
                    -
                    gps[
                        right
                    ]
                )


            else:

                end_delta = start_delta


            desired_end = (
                gps[
                    right
                ]
                -
                end_delta
            )


            correction = (
                desired_end
                -
                base[
                    -1
                ]
            )


            alpha = np.linspace(

                0.0,

                1.0,

                segment_length,

            )[
                :,
                None
            ]


            initial[
                start:
                end + 1
            ] = (

                base

                +

                alpha

                *

                correction[
                    None,
                    :
                ]

            )


        elif left >= 0:

            if left > 0:

                delta = (
                    initial[
                        left
                    ]
                    -
                    initial[
                        left - 1
                    ]
                )


            else:

                delta = np.zeros(
                    3,
                    dtype=np.float64,
                )


            for k in range(
                start,
                end + 1,
            ):

                initial[
                    k
                ] = (
                    initial[
                        k - 1
                    ]
                    +
                    delta
                )


        i += 1


    return initial


# ============================================================
# Soft anchors
# ============================================================

def build_soft_anchor_mask(
    mask,
):

    N = len(
        mask
    )


    soft = np.zeros(
        N,
        dtype=np.int32,
    )


    i = 0


    while i < N:

        if mask[
            i
        ] != 0:

            i += 1

            continue


        start = i


        while (
            i + 1 < N
            and
            mask[
                i + 1
            ] == 0
        ):

            i += 1


        end = i


        soft[
            start
        ] = 1


        k = (
            start
            +
            SOFT_ANCHOR_INTERVAL
        )


        while k <= end:

            soft[
                k
            ] = 1


            k += (
                SOFT_ANCHOR_INTERVAL
            )


        soft[
            end
        ] = 1


        i += 1


    return soft


# ============================================================
# Noise
# ============================================================

def create_robust_gps_noise(
    sigma,
):

    base = (
        noiseModel
        .Isotropic
        .Sigma(
            3,
            float(
                sigma
            ),
        )
    )


    huber = (
        noiseModel
        .mEstimator
        .Huber
        .Create(
            HUBER_K
        )
    )


    return (
        noiseModel
        .Robust
        .Create(
            huber,
            base,
        )
    )


def create_imu_parameters():

    params = (
        PreintegrationParams
        .MakeSharedU(
            9.81
        )
    )


    params.setAccelerometerCovariance(
        np.eye(
            3
        )
        *
        0.1
    )


    params.setGyroscopeCovariance(
        np.eye(
            3
        )
        *
        0.01
    )


    params.setIntegrationCovariance(
        np.eye(
            3
        )
        *
        0.001
    )


    return params


def create_relative_noise():

    return (
        noiseModel
        .Diagonal
        .Sigmas(
            np.asarray(
                [
                    RELATIVE_ROTATION_SIGMA,
                    RELATIVE_ROTATION_SIGMA,
                    RELATIVE_ROTATION_SIGMA,
                    RELATIVE_TRANSLATION_SIGMA_XY,
                    RELATIVE_TRANSLATION_SIGMA_XY,
                    RELATIVE_TRANSLATION_SIGMA_Z,
                ],
                dtype=np.float64,
            )
        )
    )


# ============================================================
# Relative factors
# ============================================================

def add_relative_motion_factors(
    graph,
    mask,
    initial_positions,
):

    N = len(
        mask
    )


    noise = create_relative_noise()


    count = 0


    i = 0


    while (
        i
        +
        RELATIVE_FACTOR_INTERVAL
        <
        N
    ):

        j = (
            i
            +
            RELATIVE_FACTOR_INTERVAL
        )


        if np.all(
            mask[
                i:
                j + 1
            ]
            ==
            0
        ):

            delta = (
                initial_positions[
                    j
                ]
                -
                initial_positions[
                    i
                ]
            )


            graph.add(
                BetweenFactorPose3(
                    symbol(
                        "x",
                        i,
                    ),
                    symbol(
                        "x",
                        j,
                    ),
                    Pose3(
                        Rot3(),
                        Point3(
                            *delta
                        ),
                    ),
                    noise,
                )
            )


            count += 1


            i = j


        else:

            i += 1


    return count


# ============================================================
# Graph
# ============================================================

def build_factor_graph(
    acc,
    gyro,
    gps,
    sigma,
    mask,
    soft_anchor_mask,
    initial_positions,
):

    N = len(
        gps
    )


    graph = NonlinearFactorGraph()


    initial = Values()


    params = create_imu_parameters()


    bias0 = gtsam.imuBias.ConstantBias()


    pose_noise = noiseModel.Isotropic.Sigma(
        6,
        0.1,
    )


    velocity_noise = noiseModel.Isotropic.Sigma(
        3,
        1.0,
    )


    bias_noise = noiseModel.Isotropic.Sigma(
        6,
        1e-3,
    )


    pose0 = Pose3(
        Rot3(),
        Point3(
            *initial_positions[
                0
            ]
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
            np.zeros(
                3
            ),
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
        np.zeros(
            3
        ),
    )


    initial.insert(
        symbol(
            "b",
            0,
        ),
        bias0,
    )


    high_count = 0

    medium_count = 0

    soft_count = 0

    removed_count = 0


    for i in range(
        N - 1
    ):

        pim = PreintegratedImuMeasurements(
            params,
            bias0,
        )


        pim.integrateMeasurement(
            acc[
                i
            ],
            gyro[
                i
            ],
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
                gtsam
                .imuBias
                .ConstantBias(),
                bias_noise,
            )
        )


        state = int(
            mask[
                i
            ]
        )


        if state == 2:

            graph.add(
                GPSFactor(
                    symbol(
                        "x",
                        i,
                    ),
                    Point3(
                        *gps[
                            i
                        ]
                    ),
                    noiseModel
                    .Isotropic
                    .Sigma(
                        3,
                        float(
                            sigma[
                                i
                            ]
                        ),
                    ),
                )
            )


            high_count += 1


        elif state == 1:

            graph.add(
                GPSFactor(
                    symbol(
                        "x",
                        i,
                    ),
                    Point3(
                        *gps[
                            i
                        ]
                    ),
                    create_robust_gps_noise(
                        sigma[
                            i
                        ]
                    ),
                )
            )


            medium_count += 1


        else:

            if soft_anchor_mask[
                i
            ] == 1:

                graph.add(
                    GPSFactor(
                        symbol(
                            "x",
                            i,
                        ),
                        Point3(
                            *initial_positions[
                                i
                            ]
                        ),
                        create_robust_gps_noise(
                            SOFT_ANCHOR_SIGMA
                        ),
                    )
                )


                soft_count += 1


            else:

                removed_count += 1


        next_pose = Pose3(
            Rot3(),
            Point3(
                *initial_positions[
                    i + 1
                ]
            ),
        )


        initial.insert(
            symbol(
                "x",
                i + 1,
            ),
            next_pose,
        )


        initial.insert(
            symbol(
                "v",
                i + 1,
            ),
            np.zeros(
                3
            ),
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
                i
            )


    last = (
        N
        -
        1
    )


    if mask[
        last
    ] == 2:

        graph.add(
            GPSFactor(
                symbol(
                    "x",
                    last,
                ),
                Point3(
                    *gps[
                        last
                    ]
                ),
                noiseModel
                .Isotropic
                .Sigma(
                    3,
                    float(
                        sigma[
                            last
                        ]
                    ),
                ),
            )
        )


        high_count += 1


    elif mask[
        last
    ] == 1:

        graph.add(
            GPSFactor(
                symbol(
                    "x",
                    last,
                ),
                Point3(
                    *gps[
                        last
                    ]
                ),
                create_robust_gps_noise(
                    sigma[
                        last
                    ]
                ),
            )
        )


        medium_count += 1


    elif soft_anchor_mask[
        last
    ] == 1:

        graph.add(
            GPSFactor(
                symbol(
                    "x",
                    last,
                ),
                Point3(
                    *initial_positions[
                        last
                    ]
                ),
                create_robust_gps_noise(
                    SOFT_ANCHOR_SIGMA
                ),
            )
        )


        soft_count += 1


    else:

        removed_count += 1


    relative_count = (
        add_relative_motion_factors(
            graph,
            mask,
            initial_positions,
        )
    )


    return (
        graph,
        initial,
        high_count,
        medium_count,
        soft_count,
        relative_count,
        removed_count,
    )


# ============================================================
# Extract trajectory
# ============================================================

def extract_trajectory(
    result,
    N,
):

    trajectory = []


    for i in range(
        N
    ):

        pose = result.atPose3(
            symbol(
                "x",
                i,
            )
        )


        translation = pose.translation()


        trajectory.append(
            [
                float(
                    translation[
                        0
                    ]
                ),
                float(
                    translation[
                        1
                    ]
                ),
                float(
                    translation[
                        2
                    ]
                ),
            ]
        )


    return np.asarray(
        trajectory,
        dtype=np.float64,
    )


# ============================================================
# Main
# ============================================================

def main():

    print(
        "="
        *
        60
    )


    print(
        "Predictive Mamba Dynamic FG V2"
    )


    print(
        "3-State Prediction + Dynamic Covariance"
    )


    print(
        "="
        *
        60
    )


    (
        acc,
        gyro,
        gps,
        reliability,
        state,
        probability,
    ) = load_data()


    N = len(
        gps
    )


    print(
        "Frames:",
        N
    )


    print()


    print(
        "Reliability:"
    )


    print(
        "min:",
        float(
            reliability.min()
        )
    )


    print(
        "max:",
        float(
            reliability.max()
        )
    )


    print(
        "mean:",
        float(
            reliability.mean()
        )
    )


    print()


    print(
        "Predicted state:"
    )


    print(
        "Reliable:",
        int(
            np.sum(
                state
                ==
                2
            )
        )
    )


    print(
        "Degrading:",
        int(
            np.sum(
                state
                ==
                1
            )
        )
    )


    print(
        "Unreliable:",
        int(
            np.sum(
                state
                ==
                0
            )
        )
    )


    (
        sigma,
        mask,
    ) = calculate_factor_parameters(
        reliability,
        state,
    )


    initial_positions = build_gated_initial_trajectory(
        gps,
        mask,
    )


    soft_anchor_mask = build_soft_anchor_mask(
        mask
    )


    print(
        "Soft anchors selected:",
        int(
            np.sum(
                soft_anchor_mask
            )
        )
    )


    (
        graph,
        initial,
        high_count,
        medium_count,
        soft_count,
        relative_count,
        removed_count,
    ) = build_factor_graph(
        acc,
        gyro,
        gps,
        sigma,
        mask,
        soft_anchor_mask,
        initial_positions,
    )


    print()


    print(
        "Graph size:",
        graph.size()
    )


    print(
        "Reliable GPS factors:",
        high_count
    )


    print(
        "Degrading GPS factors:",
        medium_count
    )


    print(
        "Soft anchors:",
        soft_count
    )


    print(
        "Relative factors:",
        relative_count
    )


    print(
        "Removed frames:",
        removed_count
    )


    print()


    print(
        "Optimizing..."
    )


    optimizer = LevenbergMarquardtOptimizer(
        graph,
        initial,
    )


    result = optimizer.optimize()


    trajectory = extract_trajectory(
        result,
        N,
    )


    os.makedirs(
        RESULT_DIR,
        exist_ok=True,
    )


    np.savetxt(
        TRAJECTORY_PATH,
        trajectory,
        fmt="%.6f",
    )


    np.savetxt(
        RELIABILITY_PATH,
        reliability,
        fmt="%.8f",
    )


    np.savetxt(
        STATE_PATH,
        state,
        fmt="%d",
    )


    np.savetxt(
        PROBABILITY_PATH,
        probability,
        fmt="%.8f",
    )


    np.savetxt(
        SIGMA_PATH,
        sigma,
        fmt="%.8f",
    )


    np.savetxt(
        GPS_MASK_PATH,
        mask,
        fmt="%d",
    )


    np.savetxt(
        SOFT_ANCHOR_MASK_PATH,
        soft_anchor_mask,
        fmt="%d",
    )


    np.savetxt(
        INITIAL_TRAJECTORY_PATH,
        initial_positions,
        fmt="%.6f",
    )


    os.makedirs(
        COMPAT_RESULT_DIR,
        exist_ok=True,
    )


    np.savetxt(
        COMPAT_TRAJECTORY_PATH,
        trajectory,
        fmt="%.6f",
    )


    print()


    print(
        "Trajectory:",
        trajectory.shape
    )


    print(
        "Final:",
        trajectory[
            -1
        ]
    )


    print(
        "Saved:",
        TRAJECTORY_PATH
    )


    print(
        "Evaluator compatibility:",
        COMPAT_TRAJECTORY_PATH
    )


    print()


    print(
        "="
        *
        60
    )


    print(
        "Predictive Mamba Dynamic FG V2 finished"
    )


    print(
        "="
        *
        60
    )


if __name__ == "__main__":

    main()
