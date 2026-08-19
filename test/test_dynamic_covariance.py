import torch


from src.model.dynamic_covariance import (
    DynamicCovariance,
)



def test_covariance_shape():


    model=DynamicCovariance()


    reliability=torch.rand(
        2,
        20,
        4
    )


    covariance=model(
        reliability
    )


    assert covariance.shape==(
        2,
        20,
        4
    )



def test_high_reliability_small_covariance():


    model=DynamicCovariance()


    high=torch.ones(
        1,
        1,
        4
    )


    low=torch.ones(
        1,
        1,
        4
    )*0.1



    high_cov=model(
        high
    )


    low_cov=model(
        low
    )



    assert torch.all(
        high_cov < low_cov
    )



def test_no_nan():


    model=DynamicCovariance()


    reliability=torch.rand(
        4,
        10,
        4
    )


    covariance=model(
        reliability
    )


    assert not torch.isnan(
        covariance
    ).any()



def test_custom_covariance():


    model=DynamicCovariance(
        base_covariance=[
            2,
            3,
            4,
            5
        ]
    )


    reliability=torch.ones(
        1,
        1,
        4
    )


    covariance=model(
        reliability
    )


    assert torch.allclose(
        covariance[0,0],
        torch.tensor(
            [
                2,
                3,
                4,
                5
            ],
            dtype=torch.float32
        )
    )
