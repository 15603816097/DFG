import numpy as np

from src.feature.lidar_feature import LidarFeatureExtractor



def test_lidar_feature():

    extractor=LidarFeatureExtractor()


    points=np.random.randn(
        100,
        4
    )


    feature=extractor.extract(
        points
    )


    assert feature.shape==(10,)



def test_lidar_nan():

    feature=LidarFeatureExtractor().extract(
        np.random.randn(20,4)
    )


    assert not np.isnan(feature).any()
