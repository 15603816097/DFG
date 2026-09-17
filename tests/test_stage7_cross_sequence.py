from pathlib import Path
import numpy as np
from experiments.cross_sequence.stage7_config import TEST_SEQUENCES,DEV_SEQUENCE,GPS_FIXED,LIDAR_T0,LIDAR_R0
from experiments.cross_sequence.build_sequence_data import _resample_plan_vector
def test_sequences_are_held_out():
    assert DEV_SEQUENCE not in TEST_SEQUENCES
    assert len(set(TEST_SEQUENCES))==3
def test_resample_plan():
    x=np.arange(10);y=_resample_plan_vector(x,25)
    assert len(y)==25 and y[0]==0 and y[-1]==9
def test_frozen_anchors():
    assert GPS_FIXED==5.0
    assert abs(LIDAR_T0-.15)<1e-12
    assert abs(LIDAR_R0-.012857142857142857)<1e-12
