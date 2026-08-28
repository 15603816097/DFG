import numpy as np
def score(x,q50,q90):return np.clip((x-q50)/max(q90-q50,1e-9),0,1)
def test_monotonic():
    assert np.all(np.diff(score(np.arange(5.),1,4))>=0)
def test_sigma_not_below_base():
    s=np.linspace(0,1,100); sig=.15*(1+2*s**2)
    assert sig.min()>=.15 and sig.max()<=.45+1e-12
def test_mapping_dynamic():
    s=score(np.linspace(0,1,101),.5,.9); assert np.ptp(.15*(1+2*s**2))>0
