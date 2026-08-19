import numpy as np



class GraphOptimizer:
    """
    Factor graph optimizer.


    Lightweight implementation.

    """


    def __init__(
        self,
        learning_rate=0.01,
        iterations=100,
    ):


        self.learning_rate=learning_rate

        self.iterations=iterations




    def optimize(
        self,
        graph,
        initial_state,
    ):


        state=np.asarray(
            initial_state,
            dtype=np.float64
        )


        for _ in range(
            self.iterations
        ):


            error=graph.compute_error(
                state
            )


            # numerical gradient

            grad=np.zeros_like(
                state
            )


            eps=1e-6


            for i in range(
                len(state)
            ):

                temp=state.copy()

                temp[i]+=eps


                grad[i]=(
                    graph.compute_error(temp)
                    -
                    error
                )/eps



            state -= (
                self.learning_rate *
                grad
            )


        return state
