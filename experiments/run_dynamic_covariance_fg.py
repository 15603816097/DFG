"""
Dynamic Covariance Factor Graph V11
==================================

Oracle Reliability + Dynamic Covariance + Soft Anchor Factor Gating

Purpose
-------
This version is used to VERIFY whether reliability-aware
soft-anchor factor gating can improve the factor graph under
synthetic GPS degradation.

IMPORTANT
---------
Ground Truth is used ONLY to generate oracle GPS reliability
for mechanism validation.

This is NOT the final proposed method.

Later:

    Oracle Reliability
            ↓
    replaced by
            ↓
    Mamba Reliability Predictor


Pipeline
--------

Corrupted GPS
      |
      |
Compare with Ground Truth
      |
      v
Oracle GPS Error
      |
      v
Reliability Score
      |
      +----------------------------+
      |                            |
      | high reliability           |
      |   normal GPS factor        |
      |                            |
      | medium reliability         |
      |   dynamic covariance       |
      |   + Huber robust kernel    |
      |                            |
      | low reliability            |
      |   reject corrupted GPS      |
      |   sparse soft anchor factor |
      +----------------------------+
      |
      v
IMU Preintegration Factor Graph
      |
      v
Optimization


Outputs
-------

results/dynamic_covariance_fg/

    trajectory.txt
    reliability.txt
    sigma.txt
    gps_error.txt
    gps_mask.txt
    initial_trajectory.txt
    soft_anchor_mask.txt
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


# ============================================================
# Project path
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(
        0,
        PROJECT_ROOT
    )


# ============================================================
# Project imports
# ============================================================

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


GT_PATH = os.path.join(
    PROJECT_ROOT,
    "results",
    "ground_truth",
    "trajectory.txt",
)


RESULT_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "dynamic_covariance_fg",
)


TRAJECTORY_PATH = os.path.join(
    RESULT_DIR,
    "trajectory.txt",
)


RELIABILITY_PATH = os.path.join(
    RESULT_DIR,
    "reliability.txt",
)


SIGMA_PATH = os.path.join(
    RESULT_DIR,
    "sigma.txt",
)


GPS_ERROR_PATH = os.path.join(
    RESULT_DIR,
    "gps_error.txt",
)


GPS_MASK_PATH = os.path.join(
    RESULT_DIR,
    "gps_mask.txt",
)


INITIAL_TRAJECTORY_PATH = os.path.join(
    RESULT_DIR,
    "initial_trajectory.txt",
)


SOFT_ANCHOR_MASK_PATH = os.path.join(
    RESULT_DIR,
    "soft_anchor_mask.txt",
)


# ============================================================
# Experiment parameters
# ============================================================

DT = 0.1


# Reliability model:
#
# reliability =
#
# exp(
#   -0.5 * (gps_error / RELIABILITY_SCALE)^2
# )

RELIABILITY_SCALE = 5.0


# Factor selection thresholds

HIGH_RELIABILITY_THRESHOLD = 0.80

LOW_RELIABILITY_THRESHOLD = 0.20


# GPS covariance

GPS_SIGMA_HIGH = 3.0

GPS_SIGMA_MEDIUM_MIN = 5.0

GPS_SIGMA_MEDIUM_MAX = 20.0


# Huber robust kernel

HUBER_K = 1.345


# V11 soft-anchor parameters
#
# Low-reliability corrupted GPS measurements are not used
# directly. Instead, sparse pseudo-position anchors are added
# from the corrected initialization trajectory.
#
# Soft-anchor settings retained from V10:
#   SOFT_ANCHOR_SIGMA    : 12.0 -> 8.0
#   SOFT_ANCHOR_INTERVAL : 10   -> 5
#
# The purpose is to reduce IMU drift inside long degraded GPS
# intervals while still avoiding direct use of corrupted GPS.

SOFT_ANCHOR_SIGMA = 8.0

SOFT_ANCHOR_INTERVAL = 5


# V11 relative-motion factors
#
# Inside low-reliability GPS regions, corrupted absolute GPS
# measurements are rejected.  However, short-window relative
# displacement is much less sensitive to a slowly varying or
# approximately constant position bias.
#
# A sparse BetweenFactorPose3 is therefore added between
# low-reliability states.  Rotation is intentionally assigned
# very large uncertainty so the factor mainly constrains
# translation.

RELATIVE_FACTOR_INTERVAL = 5

RELATIVE_TRANSLATION_SIGMA_XY = 2.0

RELATIVE_TRANSLATION_SIGMA_Z = 3.0

RELATIVE_ROTATION_SIGMA = 1000.0


# ============================================================
# Load data
# ============================================================

def load_data():

    print("Loading data...")


    if not os.path.exists(
        DATASET
    ):
        raise FileNotFoundError(
            f"Dataset not found: {DATASET}"
        )


    if not os.path.exists(
        GPS_PATH
    ):
        raise FileNotFoundError(
            f"Corrupted GPS not found: {GPS_PATH}"
        )


    if not os.path.exists(
        GT_PATH
    ):
        raise FileNotFoundError(
            f"Ground Truth not found: {GT_PATH}"
        )


    imu_loader = IMULoader(
        DATASET
    )


    N = len(
        imu_loader
    )


    acc = []

    gyro = []


    for i in range(N):

        data = imu_loader[i]


        acc.append(
            data["acceleration"]
        )


        gyro.append(
            data["angular_velocity"]
        )


    acc = np.asarray(
        acc,
        dtype=np.float64,
    )


    gyro = np.asarray(
        gyro,
        dtype=np.float64,
    )


    gps = np.loadtxt(
        GPS_PATH,
        dtype=np.float64,
    )


    gt = np.loadtxt(
        GT_PATH,
        dtype=np.float64,
    )


    if gps.ndim == 1:
        gps = gps.reshape(
            1,
            -1,
        )


    if gt.ndim == 1:
        gt = gt.reshape(
            1,
            -1,
        )


    gps = gps[:, :3]

    gt = gt[:, :3]


    if len(gps) != N:

        raise RuntimeError(
            f"GPS frames {len(gps)} "
            f"!= IMU frames {N}"
        )


    if len(gt) != N:

        raise RuntimeError(
            f"GT frames {len(gt)} "
            f"!= IMU frames {N}"
        )


    print(
        "IMU acceleration:",
        acc.shape,
    )


    print(
        "IMU gyro:",
        gyro.shape,
    )


    print(
        "Corrupted GPS:",
        gps.shape,
    )


    print(
        "Ground Truth:",
        gt.shape,
    )


    return (
        acc,
        gyro,
        gps,
        gt,
    )


# ============================================================
# Oracle reliability
# ============================================================

def calculate_oracle_reliability(
    gps,
    gt,
):

    error_vector = (
        gps
        -
        gt
    )


    gps_error = np.linalg.norm(
        error_vector,
        axis=1,
    )


    reliability = np.exp(
        -0.5
        *
        (
            gps_error
            /
            RELIABILITY_SCALE
        )
        ** 2
    )


    reliability = np.clip(
        reliability,
        0.0,
        1.0,
    )


    return (
        gps_error,
        reliability,
    )


# ============================================================
# Reliability -> sigma / factor state
# ============================================================

def calculate_factor_parameters(
    reliability,
):

    N = len(
        reliability
    )


    sigma = np.zeros(
        N,
        dtype=np.float64,
    )


    # mask:
    #
    # 2 = high reliability
    # 1 = medium reliability
    # 0 = factor removed

    mask = np.zeros(
        N,
        dtype=np.int32,
    )


    for i in range(N):

        r = float(
            reliability[i]
        )


        # ----------------------------------------
        # High reliability
        # ----------------------------------------

        if (
            r
            >=
            HIGH_RELIABILITY_THRESHOLD
        ):

            sigma[i] = (
                GPS_SIGMA_HIGH
            )


            mask[i] = 2


        # ----------------------------------------
        # Medium reliability
        # ----------------------------------------

        elif (
            r
            >=
            LOW_RELIABILITY_THRESHOLD
        ):

            normalized = (
                HIGH_RELIABILITY_THRESHOLD
                -
                r
            ) / (
                HIGH_RELIABILITY_THRESHOLD
                -
                LOW_RELIABILITY_THRESHOLD
            )


            normalized = np.clip(
                normalized,
                0.0,
                1.0,
            )


            sigma[i] = (
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


            mask[i] = 1


        # ----------------------------------------
        # Low reliability
        # ----------------------------------------

        else:

            sigma[i] = (
                GPS_SIGMA_MEDIUM_MAX
            )


            mask[i] = 0


    return (
        sigma,
        mask,
    )


# ============================================================
# Improve initial trajectory
# ============================================================

def build_gated_initial_trajectory(
    gps,
    mask,
):

    """
    Create a better initial trajectory for nonlinear
    optimization.

    Directly using corrupted GPS as initialization is bad
    when a long degraded segment contains a large position
    bias.

    For low-reliability intervals we preserve local GPS
    trajectory shape, but smoothly reconnect the segment
    between reliable boundary points.

    No GT positions are used here.
    Only the oracle mask is used.
    """

    gps = np.asarray(
        gps,
        dtype=np.float64,
    )


    N = len(
        gps
    )


    initial = gps.copy()


    bad = (
        mask == 0
    )


    i = 0


    while i < N:

        if not bad[i]:

            i += 1

            continue


        start = i


        while (
            i + 1 < N
            and
            bad[i + 1]
        ):
            i += 1


        end = i


        left = (
            start - 1
        )


        right = (
            end + 1
        )


        # ----------------------------------------
        # Both boundaries available
        # ----------------------------------------

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


            # estimate first valid step

            if left > 0:

                start_delta = (
                    gps[left]
                    -
                    gps[left - 1]
                )

            else:

                start_delta = np.zeros(
                    3,
                    dtype=np.float64,
                )


            start_position = (
                initial[left]
                +
                start_delta
            )


            # Preserve relative trajectory shape.
            #
            # Constant GPS bias cancels in relative
            # displacement.

            relative = (
                gps[start:end + 1]
                -
                gps[start]
            )


            base = (
                start_position[None, :]
                +
                relative
            )


            # Estimate desired end position using
            # the first reliable sample after outage.

            if (
                right + 1
                <
                N
            ):

                end_delta = (
                    gps[right + 1]
                    -
                    gps[right]
                )

            else:

                end_delta = (
                    start_delta
                )


            desired_end = (
                gps[right]
                -
                end_delta
            )


            correction = (
                desired_end
                -
                base[-1]
            )


            alpha = np.linspace(
                0.0,
                1.0,
                segment_length,
            )[:, None]


            initial[
                start:end + 1
            ] = (
                base
                +
                alpha
                *
                correction[None, :]
            )


        # ----------------------------------------
        # Only left boundary
        # ----------------------------------------

        elif (
            left >= 0
        ):

            if left > 0:

                delta = (
                    initial[left]
                    -
                    initial[left - 1]
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

                initial[k] = (
                    initial[k - 1]
                    +
                    delta
                )


        # ----------------------------------------
        # Only right boundary
        # ----------------------------------------

        elif (
            right < N
        ):

            initial[
                start:end + 1
            ] = gps[
                start:end + 1
            ]


        i += 1


    return initial


# ============================================================
# Build sparse soft-anchor mask
# ============================================================

def build_soft_anchor_mask(
    mask,
    interval=SOFT_ANCHOR_INTERVAL,
):

    """
    Select sparse weak anchors inside low-reliability regions.

    mask:
        2 = high reliability
        1 = medium reliability
        0 = low reliability

    For each contiguous low-reliability segment:
        - keep the first frame as a soft anchor
        - then keep one anchor every `interval` frames
        - keep the final frame as a soft anchor

    This avoids adding a weak pseudo-measurement at every frame.
    """

    mask = np.asarray(
        mask,
        dtype=np.int32,
    )


    N = len(
        mask
    )


    soft_mask = np.zeros(
        N,
        dtype=np.int32,
    )


    i = 0


    while i < N:

        if mask[i] != 0:

            i += 1

            continue


        start = i


        while (
            i + 1 < N
            and
            mask[i + 1] == 0
        ):

            i += 1


        end = i


        # first anchor in this low-reliability segment

        soft_mask[start] = 1


        # periodic anchors

        k = (
            start
            +
            interval
        )


        while k <= end:

            soft_mask[k] = 1

            k += interval


        # last anchor in this segment

        soft_mask[end] = 1


        i += 1


    return soft_mask


# ============================================================
# Create robust noise
# ============================================================

def create_robust_gps_noise(
    sigma,
):

    base_noise = (
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


    robust_noise = (
        noiseModel
        .Robust
        .Create(
            huber,
            base_noise,
        )
    )


    return robust_noise


# ============================================================
# Create relative-motion noise
# ============================================================

def create_relative_motion_noise():

    """
    6D Pose3 between-factor noise.

    GTSAM Pose3 tangent order:
        [rotation_x, rotation_y, rotation_z,
         translation_x, translation_y, translation_z]

    Rotation is nearly unconstrained.
    Translation is constrained moderately.
    """

    sigmas = np.array(
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


    return noiseModel.Diagonal.Sigmas(
        sigmas
    )


# ============================================================
# Add sparse low-reliability relative-motion factors
# ============================================================

def add_relative_motion_factors(
    graph,
    mask,
    initial_positions,
):

    """
    Add sparse short-window relative-position constraints only
    when both endpoints are inside low-reliability GPS regions.

    The relative displacement comes from the corrected initial
    trajectory, not from the corrupted absolute GPS position.

    This provides local motion continuity while avoiding direct
    use of degraded absolute GPS measurements.
    """

    N = len(
        mask
    )


    noise = (
        create_relative_motion_noise()
    )


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


        # Require the entire local window to be low reliability.
        # This avoids mixing reliable and degraded regimes.

        if np.all(
            mask[i:j + 1]
            ==
            0
        ):

            delta = (
                initial_positions[j]
                -
                initial_positions[i]
            )


            measured_relative_pose = Pose3(
                Rot3(),
                Point3(
                    *delta
                ),
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
                    measured_relative_pose,
                    noise,
                )
            )


            count += 1


            i = j


        else:

            i += 1


    return count


# ============================================================
# IMU parameters
# ============================================================

def create_imu_parameters():

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


# ============================================================
# Build reliability-aware factor graph
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


    graph = (
        NonlinearFactorGraph()
    )


    initial = (
        Values()
    )


    params = (
        create_imu_parameters()
    )


    bias0 = (
        gtsam
        .imuBias
        .ConstantBias()
    )


    pose_noise = (
        noiseModel
        .Isotropic
        .Sigma(
            6,
            0.1,
        )
    )


    velocity_noise = (
        noiseModel
        .Isotropic
        .Sigma(
            3,
            1.0,
        )
    )


    bias_noise = (
        noiseModel
        .Isotropic
        .Sigma(
            6,
            1e-3,
        )
    )


    # ====================================================
    # Initial state
    # ====================================================

    pose0 = Pose3(
        Rot3(),
        Point3(
            *initial_positions[0]
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

    soft_anchor_count = 0

    removed_count = 0


    # ====================================================
    # Factors
    # ====================================================

    for i in range(
        N - 1
    ):

        # ----------------------------------------------
        # IMU preintegration
        # ----------------------------------------------

        pim = (
            PreintegratedImuMeasurements(
                params,
                bias0,
            )
        )


        pim.integrateMeasurement(
            acc[i],
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


        # ----------------------------------------------
        # Bias random walk
        # ----------------------------------------------

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


        # ----------------------------------------------
        # GPS factor
        # ----------------------------------------------

        factor_state = int(
            mask[i]
        )


        # High reliability

        if factor_state == 2:

            gps_noise = (
                noiseModel
                .Isotropic
                .Sigma(
                    3,
                    float(
                        sigma[i]
                    ),
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
                    gps_noise,
                )
            )


            high_count += 1


        # Medium reliability

        elif factor_state == 1:

            gps_noise = (
                create_robust_gps_noise(
                    sigma[i]
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
                    gps_noise,
                )
            )


            medium_count += 1


        # Low reliability
        #
        # Do NOT use the corrupted GPS measurement directly.
        # V9 adds only sparse weak soft anchors from the
        # corrected initialization trajectory.

        else:

            if soft_anchor_mask[i] == 1:

                soft_anchor_noise = (
                    create_robust_gps_noise(
                        SOFT_ANCHOR_SIGMA
                    )
                )


                graph.add(
                    GPSFactor(
                        symbol(
                            "x",
                            i,
                        ),
                        Point3(
                            *initial_positions[i]
                        ),
                        soft_anchor_noise,
                    )
                )


                soft_anchor_count += 1


            else:

                removed_count += 1


        # ----------------------------------------------
        # Initial state
        # ----------------------------------------------

        pose_next = Pose3(
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
            pose_next,
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


        if (
            i
            %
            500
            ==
            0
        ):

            print(
                "Add factors:",
                i
            )


    # ====================================================
    # Last GPS factor
    # ====================================================

    last = (
        N - 1
    )


    if mask[last] == 2:

        gps_noise = (
            noiseModel
            .Isotropic
            .Sigma(
                3,
                float(
                    sigma[last]
                ),
            )
        )


        graph.add(
            GPSFactor(
                symbol(
                    "x",
                    last,
                ),
                Point3(
                    *gps[last]
                ),
                gps_noise,
            )
        )


        high_count += 1


    elif mask[last] == 1:

        gps_noise = (
            create_robust_gps_noise(
                sigma[last]
            )
        )


        graph.add(
            GPSFactor(
                symbol(
                    "x",
                    last,
                ),
                Point3(
                    *gps[last]
                ),
                gps_noise,
            )
        )


        medium_count += 1


    else:

        if soft_anchor_mask[last] == 1:

            soft_anchor_noise = (
                create_robust_gps_noise(
                    SOFT_ANCHOR_SIGMA
                )
            )


            graph.add(
                GPSFactor(
                    symbol(
                        "x",
                        last,
                    ),
                    Point3(
                        *initial_positions[last]
                    ),
                    soft_anchor_noise,
                )
            )


            soft_anchor_count += 1


        else:

            removed_count += 1


    relative_factor_count = (
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
        soft_anchor_count,
        relative_factor_count,
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


    for i in range(N):

        pose = result.atPose3(
            symbol(
                "x",
                i,
            )
        )


        t = (
            pose
            .translation()
        )


        trajectory.append(
            [
                float(
                    t[0]
                ),
                float(
                    t[1]
                ),
                float(
                    t[2]
                ),
            ]
        )


    return np.asarray(
        trajectory,
        dtype=np.float64,
    )


# ============================================================
# Save diagnostic results
# ============================================================

def save_diagnostics(
    trajectory,
    reliability,
    sigma,
    gps_error,
    mask,
    soft_anchor_mask,
    initial_trajectory,
):

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
        SIGMA_PATH,
        sigma,
        fmt="%.8f",
    )


    np.savetxt(
        GPS_ERROR_PATH,
        gps_error,
        fmt="%.8f",
    )


    np.savetxt(
        GPS_MASK_PATH,
        mask,
        fmt="%d",
    )


    np.savetxt(
        INITIAL_TRAJECTORY_PATH,
        initial_trajectory,
        fmt="%.6f",
    )


    np.savetxt(
        SOFT_ANCHOR_MASK_PATH,
        soft_anchor_mask,
        fmt="%d",
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
        "Dynamic Covariance Factor Graph V11"
    )


    print(
        "Oracle Reliability + Soft Anchors + Relative Motion Constraints"
    )


    print(
        "="
        *
        60
    )


    # ====================================================
    # Load
    # ====================================================

    (
        acc,
        gyro,
        gps,
        gt,
    ) = load_data()


    N = len(
        gps
    )


    print(
        "Frames:",
        N,
    )


    # ====================================================
    # Oracle reliability
    # ====================================================

    print()

    print(
        "Calculate oracle GPS reliability..."
    )


    (
        gps_error,
        reliability,
    ) = calculate_oracle_reliability(
        gps,
        gt,
    )


    (
        sigma,
        mask,
    ) = calculate_factor_parameters(
        reliability
    )


    print()

    print(
        "GPS error statistics:"
    )


    print(
        "min:",
        float(
            gps_error.min()
        ),
    )


    print(
        "max:",
        float(
            gps_error.max()
        ),
    )


    print(
        "mean:",
        float(
            gps_error.mean()
        ),
    )


    print()

    print(
        "Reliability statistics:"
    )


    print(
        "min:",
        float(
            reliability.min()
        ),
    )


    print(
        "max:",
        float(
            reliability.max()
        ),
    )


    print(
        "mean:",
        float(
            reliability.mean()
        ),
    )


    print()

    print(
        "Sigma statistics:"
    )


    print(
        "min:",
        float(
            sigma.min()
        ),
    )


    print(
        "max:",
        float(
            sigma.max()
        ),
    )


    print(
        "mean:",
        float(
            sigma.mean()
        ),
    )


    print()

    print(
        "Initial factor classification:"
    )


    print(
        "High reliability:",
        int(
            np.sum(
                mask == 2
            )
        ),
    )


    print(
        "Medium reliability:",
        int(
            np.sum(
                mask == 1
            )
        ),
    )


    print(
        "Removed:",
        int(
            np.sum(
                mask == 0
            )
        ),
    )


    # ====================================================
    # Initial trajectory
    # ====================================================

    print()

    print(
        "Build gated initial trajectory..."
    )


    initial_trajectory = (
        build_gated_initial_trajectory(
            gps,
            mask,
        )
    )


    soft_anchor_mask = (
        build_soft_anchor_mask(
            mask,
            interval=SOFT_ANCHOR_INTERVAL,
        )
    )


    print(
        "Soft anchors selected:",
        int(
            np.sum(
                soft_anchor_mask
            )
        ),
    )


    # ====================================================
    # Build graph
    # ====================================================

    print()

    print(
        "Build reliability-aware factor graph..."
    )


    (
        graph,
        initial,
        high_count,
        medium_count,
        soft_anchor_count,
        relative_factor_count,
        removed_count,
    ) = build_factor_graph(
        acc,
        gyro,
        gps,
        sigma,
        mask,
        soft_anchor_mask,
        initial_trajectory,
    )


    print()

    print(
        "Graph size:",
        graph.size(),
    )


    print(
        "High GPS factors:",
        high_count,
    )


    print(
        "Medium GPS factors:",
        medium_count,
    )


    print(
        "Soft anchor factors:",
        soft_anchor_count,
    )


    print(
        "Relative motion factors:",
        relative_factor_count,
    )


    print(
        "Removed low-reliability frames:",
        removed_count,
    )


    # ====================================================
    # Optimize
    # ====================================================

    print()

    print(
        "Optimizing..."
    )


    optimizer = (
        LevenbergMarquardtOptimizer(
            graph,
            initial,
        )
    )


    result = (
        optimizer
        .optimize()
    )


    # ====================================================
    # Extract
    # ====================================================

    trajectory = (
        extract_trajectory(
            result,
            N,
        )
    )


    # ====================================================
    # Save
    # ====================================================

    save_diagnostics(
        trajectory,
        reliability,
        sigma,
        gps_error,
        mask,
        soft_anchor_mask,
        initial_trajectory,
    )


    print()

    print(
        "Trajectory:",
        trajectory.shape,
    )


    print(
        "Final position:",
        trajectory[-1],
    )


    print()

    print(
        "Saved:",
        TRAJECTORY_PATH,
    )


    print(
        "Reliability:",
        RELIABILITY_PATH,
    )


    print(
        "Sigma:",
        SIGMA_PATH,
    )


    print(
        "GPS mask:",
        GPS_MASK_PATH,
    )


    print(
        "Soft anchor mask:",
        SOFT_ANCHOR_MASK_PATH,
    )


    print()

    print(
        "="
        *
        60
    )


    print(
        "V11 finished"
    )


    print(
        "="
        *
        60
    )


if __name__ == "__main__":

    main()