import torch
from experiments.uncertainty_v5.reliability_uncertainty_model_v5 import ReliabilityUncertaintyV5
def test_ranges():
    m=ReliabilityUncertaintyV5(21,32,1,0,3);r,u,a,b=m(torch.randn(4,64,21))
    assert r.shape==(4,3) and u.shape==(4,3)
    assert torch.all((r>=0)&(r<=1)) and torch.all((u>=0)&(u<=1))
    assert torch.all(a>=1) and torch.all(b>=1)
