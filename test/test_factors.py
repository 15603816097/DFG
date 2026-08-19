import numpy as np


from src.optimization.factors import (
    GPSFactor,
    IMUFactor,
)



def test_gps_factor():


    factor=GPSFactor(
        [1,2,3]
    )


    state=np.array(
        [1.5,2,3]
    )


    r=factor.residual(
        state
    )


    assert r.shape==(3,)



def test_factor_weight():


    factor=IMUFactor(
        [0,0,0],
        covariance=2
    )


    assert factor.weight()<1
