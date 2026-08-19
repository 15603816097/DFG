import torch


from src.model.mamba_encoder import (
    MambaEncoder,
)



def test_mamba_output_shape():


    model=MambaEncoder()


    x=torch.randn(
        2,
        20,
        51
    )


    y=model(
        x
    )


    assert y.shape==(
        2,
        20,
        128
    )



def test_mamba_forward():


    model=MambaEncoder()


    x=torch.randn(
        1,
        10,
        51
    )


    y=model(
        x
    )


    assert not torch.isnan(
        y
    ).any()



def test_mamba_batch():

    model=MambaEncoder(
        input_dim=51,
        hidden_dim=64
    )


    x=torch.randn(
        4,
        30,
        51
    )


    y=model(
        x
    )


    assert y.shape==(
        4,
        30,
        64
    )



def test_gradient():

    model=MambaEncoder()


    x=torch.randn(
        2,
        10,
        51,
        requires_grad=True
    )


    y=model(
        x
    )


    loss=y.mean()


    loss.backward()


    assert x.grad is not None
