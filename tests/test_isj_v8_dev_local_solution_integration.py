from __future__ import annotations

import json
import sys

import numpy as np

from experiments.isj_v8 import diagnose_dev_local_solutions as mod


def test_local_probe_full_two_graph_four_solve_contract(
        monkeypatch, tmp_path):
    """Mock GTSAM + measurement loader; execute real V8.3.1 driver logic."""
    output = tmp_path / "new_results"
    monkeypatch.setattr(mod, "OUT", output)
    monkeypatch.setattr(mod, "REPORT", output / "cross_initialization_diagnostics.json")
    src = tmp_path / "degraded_sensor_data.npz"
    np.savez(src, clean_gps_local=np.zeros((4544, 3), dtype=float))
    monkeypatch.setattr(mod, "SOURCE", src)

    class FakeGtsam:
        class Values:
            def __init__(self):
                self.records = {}

            def insert(self, i, pose):
                self.records[i] = pose

        class Pose3:
            def __init__(self, matrix):
                self.matrix = np.asarray(matrix).copy()

    monkeypatch.setitem(sys.modules, "gtsam", FakeGtsam)

    sigmas = {
        mod.CASE_GOOD: np.full((680, 3), 2.0),
        mod.CASE_BAD: np.full((680, 3), 7.0),
    }
    frozen = {"results": {
        mod.CASE_GOOD: {"evaluation": {
            "origin_aligned_oxTS_relative_3d_rmse_m": 2.0}},
        mod.CASE_BAD: {"evaluation": {
            "origin_aligned_oxTS_relative_3d_rmse_m": 7.0}},
    }}
    monkeypatch.setattr(
        mod, "source_preflight",
        lambda: ({"status": "PREFLIGHT_ONLY"}, sigmas, frozen),
    )
    monkeypatch.setattr(
        mod, "load_degraded_four_sensor_measurements",
        lambda folder: object(),
    )
    monkeypatch.setattr(
        mod, "subgraph_measurements", lambda src, **kwargs: src,
    )

    class Initial:
        pass

    class Graph:
        def __init__(self, case):
            self.case = case
        def error(self, values):
            return 999.0

    observed = []
    def make_graph(measurements, out, *, gps_sigmas_override, **kwargs):
        assert kwargs["build_graph_only"] is True
        arr = gps_sigmas_override
        case = mod.CASE_GOOD if float(arr[0, 0]) == 2.0 else mod.CASE_BAD
        return Graph(case), Initial(), mod.EXPECTED_COUNTS.copy()

    monkeypatch.setattr(mod, "run_four_sensor_factor_graph", make_graph)
    monkeypatch.setattr(mod, "_positions", lambda gtsam, values, n:
                        np.tile(np.eye(4), (680, 1, 1)))

    def fake_solve(graph, values, n):
        assert n == 680
        is_default = isinstance(values, Initial)
        observed.append((graph.case, "lidar" if is_default else "cross"))
        poses = np.tile(np.eye(4), (680, 1, 1))
        if graph.case == mod.CASE_GOOD:
            err = 2.0 if is_default else 2.2
            objective = 100.0 if is_default else 95.0
        else:
            err = 7.0 if is_default else 6.0
            objective = 200.0 if is_default else 180.0
        poses[:, 0, 3] = err
        return {
            "poses": poses,
            "objective": objective,
            "stabilized": True,
            "passes": [{"stabilized": True}],
        }

    monkeypatch.setattr(mod, "_solve_graph", fake_solve)
    report = mod.run_diagnostic()
    assert observed == [
        (mod.CASE_GOOD, "lidar"), (mod.CASE_BAD, "lidar"),
        (mod.CASE_GOOD, "cross"), (mod.CASE_BAD, "cross"),
    ]
    assert report["status"] == "COMPLETE_0027_DEV_EXPLORATORY_LOCAL_PROBE"
    assert all(report["solves"][case]["same_objective_comparison"][
        "cross_seed_objective_is_lower"] for case in mod.CASES)
    assert mod.REPORT.is_file()
    assert json.loads(mod.REPORT.read_text()) == report
    assert src.is_file()
