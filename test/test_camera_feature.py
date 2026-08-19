import numpy as np
import cv2

from src.feature.camera_feature import CameraFeatureExtractor



def test_camera_feature():

    extractor=CameraFeatureExtractor()


    image=np.random.randint(
        0,
        255,
        (
            100,
            100,
            3
        ),
        dtype=np.uint8
    )


    feature=extractor.extract(
        image
    )


    assert feature.shape==(8,)



def test_camera_nan():

    image=np.ones(
        (
            50,
            50,
            3
        ),
        dtype=np.uint8
    )


    feature=CameraFeatureExtractor().extract(
        image
    )


    assert not np.isnan(feature).any()
