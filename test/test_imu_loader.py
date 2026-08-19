from src.loader.imu_loader import IMULoader


DATASET_ROOT = (
    "dataset/kitti/"
    "2011_10_03/"
    "2011_10_03_drive_0027_sync"
)



def test_imu_length():

    loader = IMULoader(
        DATASET_ROOT
    )

    print(
        "IMU frames:",
        len(loader)
    )

    assert len(loader) > 0



def test_load_first_imu():

    loader = IMULoader(
        DATASET_ROOT
    )


    imu = loader[0]


    print(imu)


    assert "acceleration" in imu
    assert "angular_velocity" in imu
