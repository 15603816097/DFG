import torch


from src.loss.state_loss import (
    StateLoss,
)



def test_loss_value():


    criterion=StateLoss()



    prediction=torch.tensor(
        [
            [
                1,
                2,
                3,
                4,
                5,
                6
            ]
        ],
        dtype=torch.float32
    )



    target=torch.zeros_like(
        prediction
    )



    loss=criterion(

        prediction,

        target

    )


    assert loss>0




def test_loss_gradient():


    criterion=StateLoss()



    prediction=torch.randn(
        2,
        6,
        requires_grad=True
    )



    target=torch.randn(
        2,
        6
    )



    loss=criterion(

        prediction,

        target

    )


    loss.backward()



    assert prediction.grad is not None




def test_zero_loss():


    criterion=StateLoss()



    x=torch.randn(
        4,
        6
    )



    loss=criterion(
        x,
        x
    )


    assert torch.isclose(
        loss,
        torch.tensor(0.0)
    )
