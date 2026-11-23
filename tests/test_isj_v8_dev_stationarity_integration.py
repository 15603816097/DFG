from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from experiments.isj_v8 import diagnose_dev_stationarity as mod


class FakePose3:
    def __init__(self, arr):
        self.array = np.asarray(arr).copy()


class FakeValues:
    def __init__(self):
        self.poses = {}
    def insert(self, key, pose):
        self.poses[key] = pose


class FakeGtsam:
    Pose3 = FakePose3
    Values = FakeValues


class FakeSeed:
    pass


class FakeGraph:
    def __init__(self, case):
        self.case = case
    def error(self, values):
        if isinstance(values, FakeSeed):
            return 20000.
        val = float(values.poses[0].array[0, 3])
        if self.case == mod.CASE_GOOD:
            return 2044. if abs(val-2.) < .001 else 11090.
        return 10967. if abs(val-7.) < .001 else 2236.
    def linearize(self, values):
        graph = self
        class Linear:
            def gradientAtZero(self):
                class Gradient:
                    def norm(self):
                        return 9000. if graph.error(values) > 10000 else 0.1
                return Gradient()
        return Linear()


def test_stationarity_driver_four_solutions_keep_archives_immutable(
        monkeypatch, tmp_path):
    output = tmp_path / "new"
    monkeypatch.setattr(mod, "OUT", output)
    monkeypatch.setattr(mod, "REPORT", output / "stationarity_results.json")
    monkeypatch.setitem(sys.modules, "gtsam", FakeGtsam)

    prior = {
        "solves": {
            mod.CASE_GOOD: {
                "base_seed": {"objective": 2044.},
                "cross_seed": {"objective": 11090.},
            },
            mod.CASE_BAD: {
                "base_seed": {"objective": 10967.},
                "cross_seed": {"objective": 2236.},
            },
        },
    }
    sigma = {mod.CASE_GOOD: np.ones((680, 3)),
             mod.CASE_BAD: np.ones((680, 3)) * 2}
    monkeypatch.setattr(
        mod, "validate_existing_probes", lambda: (prior, sigma))
    monkeypatch.setattr(
        mod, "load_degraded_four_sensor_measurements",
        lambda dirname: object())
    monkeypatch.setattr(
        mod, "subgraph_measurements", lambda data, **kw: data)
    observed = []
    def build_graph(m, destination, *, gps_sigmas_override, **kw):
        assert kw["build_graph_only"] is True
        case = (mod.CASE_GOOD if gps_sigmas_override[0, 0] == 1
                else mod.CASE_BAD)
        return FakeGraph(case), FakeSeed(), mod.EXPECTED_COUNTS.copy()
    monkeypatch.setattr(mod, "run_four_sensor_factor_graph", build_graph)
    monkeypatch.setattr(
        mod, "_positions",
        lambda gtsam, values, n: np.tile(np.eye(4), (680, 1, 1)),
    )
    def solve(graph, values, n):
        assert n == 680
        initial = isinstance(values, FakeSeed)
        observed.append((graph.case, "lidar" if initial else "cross"))
        mat = np.tile(np.eye(4), (680, 1, 1))
        value = (
            2. if graph.case == mod.CASE_GOOD else 7.
        ) if initial else (
            7. if graph.case == mod.CASE_GOOD else 2.
        )
        mat[:, 0, 3] = value
        obj = (2044. if graph.case == mod.CASE_GOOD else 10967.) if initial \
            else (11090. if graph.case == mod.CASE_GOOD else 2236.)
        return {
            "poses": mat, "objective": obj, "stabilized": True,
            "passes": [{"stabilized": True}],
        }
    monkeypatch.setattr(mod, "_solve_graph", solve)
    report = mod.run_diagnostic()
    assert observed == [
        (mod.CASE_GOOD, "lidar"), (mod.CASE_BAD, "lidar"),
        (mod.CASE_GOOD, "cross"), (mod.CASE_BAD, "cross"),
    ]
    assert report["status"] == "COMPLETE_0027_DEV_STATIONARITY_DIAGNOSTIC"
    assert report["no_reference_or_oracle_loaded_for_evaluation"] is True
    assert report["case_diagnostics"][mod.CASE_GOOD][
        "other_solution_initial"]["no_objective_progress"] is True
    assert mod.REPORT.exists()
    assert json.loads(mod.REPORT.read_text()) == report
