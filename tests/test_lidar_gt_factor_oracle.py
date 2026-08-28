import numpy as np
from src.reliability.lidar_gt_factor_oracle import build_oracle

def test_oracle_drops_on_large_error():
    te=np.array([.02,.03,.04,.5,1.0]); re=np.array([.1,.1,.2,2.,5.]); clean=np.array([1,1,1,0,0],bool)
    r=build_oracle(te,re,clean)["lidar_reliability"]
    assert r[0]>=r[-1] and np.all((r>=0)&(r<=1))
