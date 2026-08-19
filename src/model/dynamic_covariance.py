import torch
import torch.nn as nn



class DynamicCovariance(nn.Module):
    """
    Reliability-guided dynamic covariance estimation.


    Input:

        reliability:

            B,T,4


    Output:

        covariance:

            B,T,4



    Formula:

        R(t)=R0/(r(t)+eps)

    """



    def __init__(
        self,
        base_covariance=None,
        eps=1e-6,
    ):

        super().__init__()


        self.eps = eps



        if base_covariance is None:

            base_covariance = [
                1.0,
                1.0,
                1.0,
                1.0
            ]



        base_covariance = torch.as_tensor(
            base_covariance,
            dtype=torch.float32
        )


        self.register_buffer(
            "base_covariance",
            base_covariance
        )



    def forward(
        self,
        reliability,
    ):

        """
        Args:

            reliability:

                B,T,4


        Returns:

            covariance:

                B,T,4

        """


        covariance = (

            self.base_covariance

            /

            (
                reliability
                +
                self.eps
            )

        )


        return covariance



    def update_base_covariance(
        self,
        covariance,
    ):

        """
        Update initial covariance.


        Args:

            covariance:

                list or Tensor

        """


        if not isinstance(
            covariance,
            torch.Tensor
        ):

            covariance = torch.as_tensor(
                covariance,
                dtype=torch.float32
            )


        else:

            covariance = covariance.detach().clone().float()



        self.base_covariance.copy_(
            covariance
        )
