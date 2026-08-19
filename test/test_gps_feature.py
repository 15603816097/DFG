import numpy as np

from src.feature.gps_feature import GPSFeatureExtractor



def test_gps_feature():

    extractor=GPSFeatureExtractor()


    gps=np.random.randn(
        20,
        3
    )


    feature=extractor.extract(
        gps
    )


    assert feature.shape==(12,)



def test_gps_nan():

    extractor=GPSFeatureExtractor()


    gps=np.random.randn(
        20,
        3
    )


    feature=extractor.extract(
        gps
    )


    assert not np.isnan(feature).any()
