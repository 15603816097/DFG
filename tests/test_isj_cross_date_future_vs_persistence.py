from __future__ import annotations

import numpy as np
import pytest

from experiments.isj_supplement.audit_future_prediction_vs_persistence import (
    WINDOW, LEAD, source_indices, evaluate_pair,
)
from experiments.dataflow_fix.build_same_source_factor_labels_v3 import (
    _future_target,
)
from mamba_v6.dataset import build_targets


def test_original_v3_future_head_alignment_three_steps_ahead():
    y = np.linspace(.05, .95, 100, dtype=np.float32)
    future = _future_target(y, 3)[:,None]
    heads = build_targets(y, future, horizon=3)
    assert np.allclose(heads[:-3,0], y[:-3])
    assert np.allclose(heads[:-3,1], y[3:])
    assert np.allclose(heads[:-3,2], y[2:-1])
    assert not np.allclose(heads[:-3,1], y[1:-2])


def test_time_indices_dev_and_cross_date_have_no_future_off_by_one():
    n=279
    cross=source_indices(n,"2011_09_30_drive_0016_sync")
    assert int(cross[0]) == WINDOW-1 == 63
    assert int(cross[-1]) == n-LEAD-1 == 275
    assert cross[-1]+LEAD == n-1
    dev=source_indices(4544,"dev")
    assert int(dev[0]) == int(4544*0.85)
    assert int(dev[-1]) == 4544-LEAD-1


def test_perfect_future_prediction_beats_imperfect_persistence():
    current = np.array([.5,.5,.5,.5])
    truth = np.array([.1,.3,.8,.9])
    s = evaluate_pair(truth,current,truth,np.array([.5]*4))
    assert s["rmse_forecast"] == pytest.approx(0)
    assert s["mae_forecast"] == pytest.approx(0)
    assert s["rmse_improvement_pct"] == pytest.approx(100)
    assert s["fraction_forecast_lower_abs_err"] == pytest.approx(1)


def test_identical_future_and_persistence_produce_zero_benefit():
    truth=np.array([.1,.2,.4,.5])
    pred=np.array([.3,.3,.3,.3])
    s=evaluate_pair(truth,pred,pred,truth)
    assert s["rmse_improvement_pct"] == pytest.approx(0)
    assert s["mae_improvement_pct"] == pytest.approx(0)


def test_invalid_input_is_rejected():
    with pytest.raises(ValueError):
        evaluate_pair(np.ones(3), np.ones(4),np.ones(3),np.ones(3))
    with pytest.raises(ValueError):
        evaluate_pair(np.array([1, np.nan, 1]),np.ones(3),np.ones(3),np.ones(3))
