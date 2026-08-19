import torch
import torch.nn as nn



class SelectiveStateSpaceBlock(nn.Module):
    """
    Lightweight Mamba-style State Space Block.


    Input:

        B,T,D


    Output:

        B,T,D



    State update:


        h_t =
        A*h_(t-1)
        +
        B*x_t



        y_t =
        C*h_t

    """


    def __init__(
        self,
        dim,
        state_dim=16,
    ):

        super().__init__()


        self.dim = dim

        self.state_dim = state_dim



        # State transition

        self.A = nn.Parameter(
            torch.randn(
                dim,
                state_dim
            )
        )


        # Input projection

        self.B = nn.Linear(
            dim,
            state_dim
        )


        # Output projection

        self.C = nn.Linear(
            state_dim,
            dim
        )



        self.norm = nn.LayerNorm(
            dim
        )



    def forward(
        self,
        x,
    ):


        """
        x:

            B,T,D

        """


        B,T,D=x.shape


        h=torch.zeros(
            B,
            self.state_dim,
            device=x.device
        )


        outputs=[]


        for t in range(T):


            xt=x[:,t,:]


            state_input=self.B(
                xt
            )


            h=torch.tanh(
                state_input
                +
                h
            )


            yt=self.C(
                h
            )


            outputs.append(
                yt.unsqueeze(1)
            )



        y=torch.cat(
            outputs,
            dim=1
        )


        return self.norm(
            y+x
        )




class MambaEncoder(nn.Module):
    """
    Mamba-based temporal encoder
    for sensor degradation modeling.



    Input:

        B,T,51


    Output:

        B,T,128



    """


    def __init__(
        self,
        input_dim=51,
        hidden_dim=128,
        state_dim=16,
        num_layers=2,
    ):

        super().__init__()



        self.input_projection=nn.Linear(
            input_dim,
            hidden_dim
        )



        layers=[]


        for _ in range(
            num_layers
        ):

            layers.append(
                SelectiveStateSpaceBlock(
                    hidden_dim,
                    state_dim
                )
            )



        self.layers=nn.ModuleList(
            layers
        )



        self.output_norm=nn.LayerNorm(
            hidden_dim
        )



    def forward(
        self,
        x,
    ):

        """
        x:

            B,T,51


        return:

            B,T,128

        """


        x=self.input_projection(
            x
        )


        for layer in self.layers:

            x=layer(
                x
            )



        return self.output_norm(
            x
        )
