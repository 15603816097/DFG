import torch
import torch.nn as nn



class ReliabilityHead(nn.Module):
    """
    Dynamic sensor reliability prediction head.


    Input:

        B,T,128


    Output:

        B,T,4


    Four reliability values:

        0 : IMU
        1 : GPS
        2 : LiDAR
        3 : Camera


    Range:

        0 ~ 1


    """


    def __init__(
        self,
        input_dim=128,
        hidden_dim=64,
        sensor_num=4,
    ):

        super().__init__()


        self.sensor_num=sensor_num


        self.network=nn.Sequential(

            nn.Linear(
                input_dim,
                hidden_dim
            ),

            nn.ReLU(),


            nn.Linear(
                hidden_dim,
                sensor_num
            ),

            nn.Sigmoid()

        )



    def forward(
        self,
        x,
    ):

        """
        Args:

            x:

            B,T,128


        Returns:

            reliability:

            B,T,4

        """


        return self.network(
            x
        )
