import torch
import torch.nn as nn



class StateLoss(nn.Module):

    """
    Vehicle state regression loss.



    Prediction:

        B,6



    Target:

        B,6



    """



    def __init__(
        self,
        reduction="mean"
    ):

        super().__init__()



        self.loss=nn.MSELoss(
            reduction=reduction
        )



    def forward(
        self,
        prediction,
        target
    ):


        return self.loss(

            prediction,

            target

        )
