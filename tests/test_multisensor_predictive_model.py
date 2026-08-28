import torch
from src.model.multisensor_predictive_reliability_model import MultiSensorPredictiveReliabilityModel
def build():
    return MultiSensorPredictiveReliabilityModel(21,20,19,16,16,64,1,.1)
def test_forward():
    m=build();o=m(torch.randn(2,8,21),torch.randn(2,8,20),torch.randn(2,8,19),torch.randn(2,8,16))
    assert o["embedding"].shape[0]==2
    for s in ("gps","imu","lidar","camera"):
        assert o[s]["current"].shape==(2,)
        assert o[s]["future"].shape==(2,)
def test_probabilities():
    m=build();o=m(torch.randn(2,8,21),torch.randn(2,8,20),torch.randn(2,8,19),torch.randn(2,8,16))
    for s in ("gps","imu","lidar","camera"):
        for h in ("current","future"):
            assert torch.all(o[s][h]>=0) and torch.all(o[s][h]<=1)
