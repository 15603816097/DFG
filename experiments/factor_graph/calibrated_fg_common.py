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
from src.reliability.reliability_calibrator import (
    ReliabilityCalibrator,
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

PREDICTION_PATH = os.path.join(
    ROOT,
    "results",
    "predictive_reliability_new",
    "predicted_reliability.txt",
)

GT_PATH = os.path.join(
    ROOT,
    "results",
    "ground_truth",
    "trajectory.txt",
)

DT = 0.1


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


def load_data(
    require_prediction=False,
    require_gt=False,
):
    gps = np.loadtxt(
        GPS_PATH,
        dtype=np.float64,
    )

    acceleration, gyro = (
        load_imu()
    )

    arrays = [
        gps,
        acceleration,
        gyro,
    ]

    prediction = None
    gt = None

    if require_prediction:
        prediction = np.loadtxt(
            PREDICTION_PATH,
            dtype=np.float64,
        ).reshape(-1)

        arrays.append(
            prediction
        )

    if require_gt:
        gt = np.loadtxt(
            GT_PATH,
            dtype=np.float64,
        )

        arrays.append(
            gt
        )

    n = min(
        len(x)
        for x in arrays
    )

    return {
        "gps": gps[:n],
        "acceleration":
            acceleration[:n],
        "gyro":
            gyro[:n],
        "prediction":
            None
            if prediction is None
            else prediction[:n],
        "ground_truth":
            None
            if gt is None
            else gt[:n],
        "frames": n,
    }


def make_preintegration_params():
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


def optimize_graph(
    gps,
    acceleration,
    gyro,
    sigma,
    output_dir,
    title,
    reliability=None,
    extra_outputs=None,
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

    params = (
        make_preintegration_params()
    )

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

    print("=" * 60)
    print(title)
    print("Frames:", n)
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
        "Graph:",
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

    if reliability is not None:
        np.savetxt(
            os.path.join(
                output_dir,
                "reliability.txt",
            ),
            np.asarray(
                reliability,
                dtype=np.float64,
            )[:n],
            fmt="%.8f",
        )

    for filename, values in (
        extra_outputs or {}
    ).items():
        np.savetxt(
            os.path.join(
                output_dir,
                filename,
            ),
            np.asarray(
                values,
                dtype=np.float64,
            )[:n],
            fmt="%.8f",
        )

    print(
        "Trajectory:",
        trajectory.shape,
    )

    print(
        "Saved:",
        output_dir,
    )

    return trajectory


def factor_graph_feedback(
    gps,
    stage1_trajectory,
    scale=8.0,
):
    """
    Feedback reliability from GPS-vs-stage1 factor-graph position
    innovation.  Stage1 is built using predictive covariance first.

    This is a two-pass approximation of an online innovation-feedback
    mechanism.  It avoids the previous GPS-vs-GPS constant-velocity
    residual that confused real vehicle dynamics with sensor failure.
    """
    gps = np.asarray(
        gps,
        dtype=np.float64,
    )

    stage1_trajectory = np.asarray(
        stage1_trajectory,
        dtype=np.float64,
    )

    n = min(
        len(gps),
        len(stage1_trajectory),
    )

    innovation = (
        np.linalg.norm(
            gps[:n]
            -
            stage1_trajectory[:n],
            axis=1,
        )
    )

    feedback = (
        residual_reliability(
            innovation,
            scale=scale,
        )
    )

    return (
        feedback,
        innovation,
    )


def calibrate_prediction(
    raw_prediction,
):
    calibrator = (
        ReliabilityCalibrator(
            center=0.55,
            slope=1.35,
            temperature=0.85,
            minimum=0.02,
            maximum=0.995,
        )
    )

    return (
        calibrator.transform(
            raw_prediction
        )
    )


def map_sigma(
    reliability,
):
    return (
        reliability_to_sigma(
            reliability,
            sigma_min=3.0,
            sigma_max=20.0,
            gamma=3.0,
        )
    )
