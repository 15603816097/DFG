import numpy as np


from src.feature.imu_feature import (
    IMUFeatureExtractor,
)



def test_imu_feature_dimension():

    extractor = IMUFeatureExtractor()


    imu=np.random.randn(
        20,
        6
    )


    feature = extractor.extract(
        imu
    )


    assert feature.shape == (21,)



def test_imu_feature_no_nan():

    extractor = IMUFeatureExtractor()


    imu=np.random.randn(
        20,
        6
    )


    feature=extractor.extract(
        imu
    )


    assert not np.isnan(
        feature
    ).any()



def test_imu_sequence_feature():

    extractor = IMUFeatureExtractor(
        window_size=5
    )


    imu=np.random.randn(
        20,
        6
    )


    features = extractor.extract_sequence(
        imu
    )


    assert features.shape[0] == 16



def test_constant_motion():

    extractor = IMUFeatureExtractor()


    imu=np.ones(
        (
            20,
            6
        )
    )


    feature=extractor.extract(
        imu
    )


    assert np.all(
        np.isfinite(feature)
    )



def test_invalid_dimension():

    extractor=IMUFeatureExtractor()


    imu=np.random.randn(
        10,
        5
    )


    try:

        extractor.extract(
            imu
        )

        assert False


    except ValueError:

        assert True
