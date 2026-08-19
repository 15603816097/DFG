import torch


from src.model.reliability_head import (
    ReliabilityHead,
)



def test_reliability_shape():


    model=ReliabilityHead()


    x=torch.randn(
        2,
        20,
        128
    )


    y=model(
        x
    )


    assert y.shape==(
        2,
        20,
        4
    )



def test_reliability_range():


    model=ReliabilityHead()


    x=torch.randn(
        1,
        10,
        128
    )


    y=model(
        x
    )


    assert torch.all(
        y>=0
    )


    assert torch.all(
        y<=1
    )



def test_reliability_no_nan():


    model=ReliabilityHead()


    x=torch.randn(
        4,
        50,
        128
    )


    y=model(
        x
    )


    assert not torch.isnan(
        y
    ).any()



def test_reliability_gradient():


    model=ReliabilityHead()


    x=torch.randn(
        2,
        20,
        128,
        requires_grad=True
    )


    y=model(
        x
    )


    loss=y.mean()


    loss.backward()


    assert x.grad is not None
