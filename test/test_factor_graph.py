import numpy as np


from src.optimization.factor_graph import (
    FactorGraph,
)


from src.optimization.factors import (
    GPSFactor,
)



def test_graph_add_factor():


    graph=FactorGraph()


    graph.add_state(
        [
            0,
            0,
            0
        ]
    )


    graph.add_factor(
        GPSFactor(
            [
                1,
                1,
                1
            ]
        )
    )


    error=graph.compute_error(
        np.array(
            [
                0,
                0,
                0
            ]
        )
    )


    assert error>0
