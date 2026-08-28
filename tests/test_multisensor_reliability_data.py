import numpy as np
from src.multisensor_reliability.degradation.plans import create_multisensor_degradation_plan
from src.multisensor_reliability.degradation.apply import corrupt_gps,corrupt_imu,corrupt_lidar,corrupt_camera
from src.multisensor_reliability.labels import severity_to_reliability,future_window_min
from src.multisensor_reliability.features import lidar_features,camera_features

def test_plan_shapes():
    p=create_multisensor_degradation_plan(100)
    assert p["n_frames"]==100
    for s in ("gps","imu","lidar","camera"):
        assert p["severity"][s].shape==(100,)
        assert p["mode"][s].shape==(100,)

def test_reliability_limits():
    r=severity_to_reliability([0,.5,1])
    assert np.all((r>=0)&(r<=1));assert r[0]==1 and r[-1]==0

def test_future_window_min():
    r=np.array([1,.9,.7,.8,.6]);o=future_window_min(r,2);assert np.isclose(o[0],.7)

def test_deterministic():
    p=np.array([1.,2.,3.]);a=corrupt_gps(p,.8,1,10);b=corrupt_gps(p,.8,1,10);assert np.allclose(a,b)

def test_imu_shape():
    a,g=corrupt_imu(np.zeros(3),np.zeros(3),.7,1,5);assert a.shape==(3,) and g.shape==(3,)

def test_lidar_features():
    p=np.random.randn(100,4).astype(np.float32);q=corrupt_lidar(p,.6,1,3);assert len(lidar_features(q))==19

def test_camera_features():
    im=np.full((64,96,3),128,np.uint8);q=corrupt_camera(im,.8,1,7);assert q.shape==im.shape and len(camera_features(q))==16
