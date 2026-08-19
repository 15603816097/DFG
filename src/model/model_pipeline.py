import torch
import torch.nn as nn


class ModelPipeline(nn.Module):
    """
    Overall DFG model pipeline


    Input:
        feature:
            [B,T,51]


    Output:

        state:
            [B,6]

        reliability:
            [B,4]

    """


    def __init__(
        self,
        input_dim=51,
        hidden_dim=64,
        state_dim=6,
        sensor_num=4
    ):

        super().__init__()


        self.encoder = nn.Sequential(

            nn.Linear(
                input_dim,
                hidden_dim
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                hidden_dim
            ),

            nn.ReLU()

        )


        self.state_head = nn.Sequential(

            nn.Linear(
                hidden_dim,
                32
            ),

            nn.ReLU(),

            nn.Linear(
                32,
                state_dim
            )

        )


        self.reliability_head = nn.Sequential(

            nn.Linear(
                hidden_dim,
                32
            ),

            nn.ReLU(),

            nn.Linear(
                32,
                sensor_num
            ),

            nn.Sigmoid()

        )



    def forward(
        self,
        x
    ):


        """
        x:

        [B,T,51]

        """


        if x.dim()==2:

            x=x.unsqueeze(0)



        # 时间平均
        # 后续可以替换成Mamba

        feature=torch.mean(
            x,
            dim=1
        )


        hidden=self.encoder(
            feature
        )


        state=self.state_head(
            hidden
        )


        reliability=self.reliability_head(
            hidden
        )



        return {

            "state":state,

            "reliability":reliability

        }
