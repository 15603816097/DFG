from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from experiments.isj_v8 import solver_health_v8 as h


def test_v8_solver_zero_progress_large_gradient_is_stalled():
    result = h.classify_step(
        objective_before=11090.544423221887,
        objective_after=11090.544423221887,
        translation_shift_rmse_m=0.,
        gradient_l2_after=274.44174169327647,
    )
    assert result["status"] == "STALLED_NO_PROGRESS"
    assert result["numerical_zero_step"] is True
    assert result["gradient_stationarity_certified"] is False


def test_v8_solver_original_blend_stuck_is_stalled():
    result = h.classify_step(
        objective_before=10967.319102847954,
        objective_after=10967.319102847954,
        translation_shift_rmse_m=0.,
        gradient_l2_after=263.9250452364541,
    )
    assert result["status"] == "STALLED_NO_PROGRESS"
    assert result["relative_objective_decrease"] == 0


def test_v8_solver_positive_small_improvement_is_candidate_only():
    result = h.classify_step(
        objective_before=2044.704264,
        objective_after=2044.704200,
        translation_shift_rmse_m=.0000176,
        gradient_l2_after=4.591578136,
    )
    assert result["status"] == "CONVERGED_CANDIDATE"
    assert result["numerical_zero_step"] is False
    assert result["gradient_stationarity_certified"] is False


def test_v8_solver_meaningful_progress_is_continue():
    result = h.classify_step(
        objective_before=200.,
        objective_after=150.,
        translation_shift_rmse_m=1.,
        gradient_l2_after=30.,
    )
    assert result["status"] == "CONTINUE"


def test_v8_solver_objective_increase_fails_closed():
    with pytest.raises(RuntimeError, match="increased"):
        h.classify_step(
            objective_before=1000.,
            objective_after=1000.1,
            translation_shift_rmse_m=0.2,
            gradient_l2_after=3.,
        )


@pytest.mark.parametrize("field", [
    "objective_before", "objective_after",
    "translation_shift_rmse_m", "gradient_l2_after",
])
def test_v8_solver_invalid_numeric_fails_closed(field):
    args = dict(objective_before=1000., objective_after=999.,
                translation_shift_rmse_m=.5, gradient_l2_after=3.)
    args[field] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        h.classify_step(**args)


def test_v8_solver_case_selection_never_picks_stalled_solution():
    selected = h.select_reliable_candidate({
        "fake_low": {"status": "STALLED_NO_PROGRESS", "objective": 1.0},
        "valid": {"status": "CONVERGED_CANDIDATE", "objective": 20.0},
        "valid_high": {"status": "CONVERGED_CANDIDATE", "objective": 30.0},
    })
    assert selected["selected_seed"] == "valid"
    assert selected["candidate_count"] == 2
    assert selected["status"] == "CANDIDATE_SELECTED_NOT_STATIONARITY_CERTIFIED"
    nothing = h.select_reliable_candidate({
        "fail": {"status": "STALLED_NO_PROGRESS", "objective": 1.}
    })
    assert nothing["selected_seed"] is None
    assert nothing["status"] == "NO_QUALIFIED_CANDIDATE"


class FakeParams:
    def setMaxIterations(self, n):
        self.n = n
    def setRelativeErrorTol(self, v):
        self.v = v


class FakeValues:
    def __init__(self, obj, grad, shift=0.):
        self.obj = obj
        self.grad = grad
        self.poses = np.tile(np.eye(4), (4, 1, 1))
        self.poses[:, 0, 3] = shift


class FakeGraph:
    def error(self, x):
        return x.obj
    def linearize(self, x):
        return types.SimpleNamespace(
            gradientAtZero=lambda: types.SimpleNamespace(
                norm=lambda: x.grad
            )
        )


class FakeOptimizer:
    calls = []
    next_state = None
    state_queue = []

    def __init__(self, graph, current, params):
        self.calls.append((graph, current, params.n, params.v))

    def optimize(self):
        return self.state_queue.pop(0) if self.state_queue else self.next_state


def _mock_gtsam(monkeypatch):
    fake = types.SimpleNamespace(
        LevenbergMarquardtParams=FakeParams,
        LevenbergMarquardtOptimizer=FakeOptimizer,
    )
    FakeOptimizer.calls = []
    FakeOptimizer.state_queue = []
    monkeypatch.setitem(sys.modules, "gtsam", fake)
    monkeypatch.setattr(
        h, "_positions", lambda _g, state, n: state.poses.copy()
    )


def test_v8_solver_driver_rejects_zero_progress_stable(monkeypatch):
    _mock_gtsam(monkeypatch)
    original = FakeValues(11090.5444, 274.4417)
    FakeOptimizer.next_state = FakeValues(11090.5444, 274.4417)
    result = h.solve_v8_health(FakeGraph(), original, 4)
    assert result["status"] == "STALLED_NO_PROGRESS"
    assert len(result["passes"]) == 1
    assert result["initial_gradient_l2"] == pytest.approx(274.4417)
    assert result["final_gradient_l2"] == pytest.approx(274.4417)
    assert result["candidate_not_stationarity_certificate"] is True
    assert FakeOptimizer.calls[0][2:] == (100, 1e-7)


def test_v8_solver_driver_genuine_small_progress_is_only_candidate(monkeypatch):
    _mock_gtsam(monkeypatch)
    original = FakeValues(100., 10.)
    FakeOptimizer.next_state = FakeValues(99.99995, 1., shift=.001)
    result = h.solve_v8_health(FakeGraph(), original, 4)
    assert result["status"] == "CONVERGED_CANDIDATE"
    assert result["passes"][0]["absolute_objective_decrease"] > 0.
    assert result["final_gradient_l2"] == 1.
    assert result["legacy_stable_flag_not_used"] is True


def test_v8_solver_driver_max_passes_not_converged(monkeypatch):
    _mock_gtsam(monkeypatch)
    monkeypatch.setattr(h, "MAX_OUTER_PASSES", 2)
    original = FakeValues(100., 100.)
    FakeOptimizer.state_queue = [
        FakeValues(80., 70., shift=1.),
        FakeValues(60., 50., shift=2.),
    ]
    result = h.solve_v8_health(FakeGraph(), original, 4)
    assert result["status"] == "MAX_PASSES_NOT_CONVERGED"
    assert len(result["passes"]) == 2


def test_v8_solver_gradient_api_failure_fails_closed():
    g = types.SimpleNamespace(linearize=lambda x: None)
    with pytest.raises(RuntimeError, match="Cannot calculate"):
        h.gradient_l2(g, object())


def test_v8_solver_selection_requires_a_case():
    with pytest.raises(ValueError, match="no solver candidates"):
        h.select_reliable_candidate({})

def test_v8_solver_rotation_shift_prevents_false_candidate():
    row = h.classify_step(
        objective_before=100.0,
        objective_after=99.99995,
        translation_shift_rmse_m=0.0001,
        rotation_shift_rmse_rad=0.15,
        gradient_l2_after=1.0,
    )
    assert row["status"] == "CONTINUE"
    assert row["rotation_shift_rmse_rad"] == pytest.approx(0.15)
