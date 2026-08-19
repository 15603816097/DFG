import numpy as np


from src.feature.fusion_feature import (
    FusionFeatureExtractor,
)



def create_features():

    imu=np.random.randn(
        21
    )

    gps=np.random.randn(
        12
    )

    lidar=np.random.randn(
        10
    )

    camera=np.random.randn(
        8
    )


    return (
        imu,
        gps,
        lidar,
        camera,
    )



def test_fusion_dimension():


    extractor=FusionFeatureExtractor()


    imu,gps,lidar,camera=create_features()


    feature=extractor.extract(
        imu,
        gps,
        lidar,
        camera,
    )


    assert feature.shape==(51,)



def test_missing_sensor():


    extractor=FusionFeatureExtractor()


    imu,gps,lidar,camera=create_features()


    feature=extractor.extract(
        imu,
        gps,
        lidar,
        None,
    )


    assert feature.shape==(43,)



def test_sequence_feature():


    extractor=FusionFeatureExtractor()


    imu=np.random.randn(
        20,
        21
    )

    gps=np.random.randn(
        20,
        12
    )

    lidar=np.random.randn(
        20,
        10
    )

    camera=np.random.randn(
        20,
        8
    )


    feature=extractor.extract_sequence(
        imu,
        gps,
        lidar,
        camera,
    )


    assert feature.shape==(20,51)



def test_no_nan():


    extractor=FusionFeatureExtractor()


    imu,gps,lidar,camera=create_features()


    feature=extractor.extract(
        imu,
        gps,
        lidar,
        camera,
    )


    assert not np.isnan(
        feature
    ).any()
