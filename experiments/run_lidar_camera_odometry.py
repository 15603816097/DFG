import os
import subprocess
import sys


ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


STEPS = [
    (
        "LiDAR relative odometry",
        "experiments/odometry/"
        "run_lidar_odometry.py",
    ),

    (
        "Stereo visual odometry",
        "experiments/odometry/"
        "run_stereo_visual_odometry.py",
    ),
]


def main():
    print("=" * 92)
    print(
        "DFG LIDAR + CAMERA "
        "ODOMETRY PIPELINE"
    )
    print("=" * 92)

    for i, (
        title,
        relative_path,
    ) in enumerate(
        STEPS,
        start=1,
    ):
        print()
        print(
            f"[{i}/{len(STEPS)}] "
            f"{title}"
        )

        subprocess.run(
            [
                sys.executable,
                os.path.join(
                    ROOT,
                    relative_path,
                ),
            ],
            cwd=ROOT,
            check=True,
        )

    print()
    print(
        "LiDAR + Camera odometry "
        "pipeline finished."
    )


if __name__ == "__main__":
    main()
