import numpy as np
import gtsam

from experiments.isj_v8.backend_c8_numeric import (
    build_c8_graphs, directional_audit, verify_objective, yaw_aligned_seed,
)


def make_graph():
    n = 4
    poses = np.repeat(np.eye(4)[None], n, axis=0)
    poses[:, :3, 3] = np.array([[0, 0, 0], [1, 0, 0], [2, 1, .1], [3, 2, .2]])
    yaw = -1.7
    r = gtsam.Rot3.Rz(yaw).matrix()
    gps = poses[:, :3, 3] @ r.T
    sigma = np.tile([1., 2., 3.], (n, 1))
    graph = gtsam.NonlinearFactorGraph()
    noise = gtsam.noiseModel.Isotropic.Sigma(6, .2)
    graph.add(gtsam.PriorFactorPose3(0, gtsam.Pose3(), noise))
    for k in range(n):
        graph.add(gtsam.PriorFactorPose3(k, gtsam.Pose3(gtsam.Rot3(), gps[k]), noise))
        if k:
            graph.add(gtsam.BetweenFactorPose3(
                k-1, k, gtsam.Pose3(gtsam.Rot3.Rz(.15), np.array([1., .2, .1])), noise))
    return graph, poses, gps, sigma, yaw


def test_yaw_alignment():
    _, poses, gps, _, yaw = make_graph()
    values, info = yaw_aligned_seed(gtsam, poses, gps)
    assert np.isclose(info["yaw_deg"], np.degrees(yaw))
    assert np.allclose(np.stack([values.atPose3(k).translation() for k in range(4)]), gps)


def test_native_gps_is_rotation_invariant():
    noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([1., 2., 3.]))
    gps = gtsam.GPSFactor(0, np.array([10., 5., 1.]), noise)
    errs = []
    for yaw in (0., .8, 1.6, 2.9):
        vals = gtsam.Values()
        vals.insert(0, gtsam.Pose3(gtsam.Rot3.Rz(yaw), np.array([13., 7., 2.])))
        errs.append(gps.error(vals))
    assert np.allclose(errs, errs[0], atol=1e-10, rtol=0)


def test_objective_and_derivatives_at_nonzero_residuals():
    graph, poses, gps, sigma, _ = make_graph()
    native, corrected, keepalive, between_count = build_c8_graphs(gtsam, graph, gps, sigma)
    assert between_count == 3 and len(keepalive) == 4
    assert native.size() == corrected.size() == graph.size()
    values, _ = yaw_aligned_seed(gtsam, poses, gps)
    trial = gtsam.Values()
    for k in range(4):
        trial.insert(k, values.atPose3(k).retract(
            np.array([.05, -.03, .09, .02, -.01, .03]) * (k+1)))
    for state in (values, trial):
        assert verify_objective(native, corrected, state) > 0
        audit = directional_audit(gtsam, native, corrected, state, 4)
        assert audit["passed"], audit


def test_alignment_rejects_stationary_track():
    import pytest
    _, poses, gps, _, _ = make_graph()
    poses[:, :3, 3] = 0
    with pytest.raises(ValueError, match="stationary"):
        yaw_aligned_seed(gtsam, poses, gps)
