import numpy as np



class FactorGraph:
    """
    Lightweight factor graph.


    State:

        x=[px,py,pz,vx,vy,vz]


    """


    def __init__(self):

        self.factors=[]


        self.states=[]




    def add_state(
        self,
        state
    ):


        state=np.asarray(
            state,
            dtype=np.float64
        )


        self.states.append(
            state
        )


        return len(
            self.states
        )-1





    def add_factor(
        self,
        factor
    ):


        self.factors.append(
            factor
        )




    def compute_error(
        self,
        state
    ):


        total_error=0.0


        for factor in self.factors:


            residual=factor.residual(
                state
            )


            weight=factor.weight()


            total_error += (
                weight *
                np.sum(
                    residual**2
                )
            )


        return total_error
