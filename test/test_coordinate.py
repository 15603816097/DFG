import numpy as np

from src.preprocessing.coordinate import (
    GPSCoordinateConverter,
    CoordinateTransform,
)



def test_gps_origin():

    converter = GPSCoordinateConverter()


    enu = converter.gps_to_enu(
        39.0,
        116.0,
        50.0,
    )


    assert enu.shape == (3,)



def test_gps_same_point():

    converter = GPSCoordinateConverter()


    enu = converter.gps_to_enu(
        39.0,
        116.0,
        50.0,
    )


    assert np.allclose(
        enu,
        np.zeros(3),
        atol=1e-6,
    )



def test_gps_batch():

    converter = GPSCoordinateConverter()


    gps=np.array(
        [
            [
                39.0,
                116.0,
                50
            ],
            [
                39.00001,
                116.00001,
                51
            ],
        ]
    )


    enu=converter.gps_array_to_enu(
        gps
    )


    assert enu.shape==(2,3)



def test_transform():

    points=np.array(
        [
            [
                1,
                2,
                3
            ]
        ]
    )


    T=np.eye(4)


    result=CoordinateTransform.transform(
        points,
        T
    )


    assert np.allclose(
        result,
        points
    )



def test_distance():

    d=CoordinateTransform.distance(
        [0,0,0],
        [3,4,0]
    )


    assert d==5
