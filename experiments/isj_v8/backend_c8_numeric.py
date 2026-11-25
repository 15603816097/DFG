"""Development-only C8 backend: preserve residuals, correct linearization.

No legacy graph or archive is modified. Full-window yaw alignment is offline.
Numerical derivatives are a reference implementation, not a real-time claim.
"""
from __future__ import annotations

import numpy as np

JACOBIAN_STEP = 1e-6
FD_STEPS = (1e-4, 1e-5)
FD_ATOL = 1e-5
FD_RTOL = 1e-2


def poses_to_values(gtsam, poses):
    values = gtsam.Values()
    for k, pose in enumerate(np.asarray(poses, dtype=float)):
        values.insert(k, gtsam.Pose3(gtsam.Rot3(pose[:3, :3]), pose[:3, 3]))
    return values


def yaw_aligned_seed(gtsam, poses, gps):
    poses = np.asarray(poses, dtype=float)
    gps = np.asarray(gps, dtype=float)
    if (poses.ndim != 3 or poses.shape[1:] != (4, 4)
            or gps.shape != (len(poses), 3) or len(poses) < 3
            or not np.isfinite(poses).all() or not np.isfinite(gps).all()):
        raise ValueError("invalid alignment inputs")
    a = poses[:, :2, 3] - poses[:, :2, 3].mean(axis=0)
    b = gps[:, :2] - gps[:, :2].mean(axis=0)
    if min(np.linalg.norm(a), np.linalg.norm(b)) <= 1e-8:
        raise ValueError("heading not observable from stationary tracks")
    u, s, vt = np.linalg.svd(a.T @ b)
    if s[0] <= 1e-8:
        raise ValueError("degenerate heading alignment")
    d = np.diag([1., np.sign(np.linalg.det(vt.T @ u.T))])
    r = np.eye(3)
    r[:2, :2] = vt.T @ d @ u.T
    out = poses.copy()
    out[:, :3, :3] = r @ poses[:, :3, :3]
    out[:, :3, 3] = poses[:, :3, 3] @ r.T
    return poses_to_values(gtsam, out), {
        "yaw_deg": float(np.degrees(np.arctan2(r[1, 0], r[0, 0]))),
        "singular_values": s.tolist(),
        "full_window_offline": True,
    }


def numerical_factor(gtsam, original, step=JACOBIAN_STEP):
    """Wrap Gaussian Pose3 factor using its exact whitened residual.

    Isotropic unit noise avoids double whitening. Central differences in
    the actual Pose3 retract chart include the omitted local-map terms.
    """
    keys = [int(k) for k in original.keys()]
    if len(keys) not in (1, 2) or not np.isfinite(step) or step <= 0:
        raise ValueError("invalid factor or difference step")
    if not isinstance(original, (gtsam.PriorFactorPose3, gtsam.BetweenFactorPose3)):
        raise TypeError("only Gaussian Pose3 prior/between factors are supported")
    basis = np.eye(6)

    def residual(states):
        local = gtsam.Values()
        for key, pose in zip(keys, states):
            local.insert(key, pose)
        r = np.asarray(original.whitenedError(local), dtype=float).reshape(6)
        if not np.isfinite(r).all():
            raise ValueError("nonfinite factor residual")
        return r

    def callback(this, values, H):
        states = [values.atPose3(k) for k in keys]
        r = residual(states)
        if H is not None:
            for i, state in enumerate(states):
                j = np.empty((6, 6), dtype=float, order="F")
                for k in range(6):
                    plus, minus = list(states), list(states)
                    delta = step * basis[k]
                    plus[i] = state.retract(delta)
                    minus[i] = state.retract(-delta)
                    j[:, k] = (residual(plus) - residual(minus)) / (2 * step)
                H[i] = j
        return r

    return gtsam.CustomFactor(
        gtsam.noiseModel.Isotropic.Sigma(6, 1.), keys, callback), callback


def build_c8_graphs(gtsam, legacy, gps, sigma):
    """Construct native C8 and same-objective numerical C8 graphs."""
    gps, sigma = np.asarray(gps, float), np.asarray(sigma, float)
    n = len(gps)
    if (gps.shape != (n, 3) or sigma.shape != (n, 3)
            or not np.isfinite(gps).all() or not np.isfinite(sigma).all()
            or np.any(sigma <= 0)):
        raise ValueError("invalid GPS measurements or sigmas")
    first = legacy.at(0)
    if not isinstance(first, gtsam.PriorFactorPose3) or list(first.keys()) != [0]:
        raise ValueError("unexpected legacy anchor")
    native, corrected = gtsam.NonlinearFactorGraph(), gtsam.NonlinearFactorGraph()
    prior = gtsam.PriorFactorPose3(
        0, gtsam.Pose3(), gtsam.noiseModel.Diagonal.Sigmas(
            np.array([1e3, 1e3, 1e3, 1e-3, 1e-3, 1e-3])))
    native.add(prior)
    wrapped, cb = numerical_factor(gtsam, prior)
    corrected.add(wrapped)
    callbacks = [cb]
    seen, between_count = set(), 0
    for j in range(1, legacy.size()):
        f = legacy.at(j)
        keys = list(f.keys())
        if len(keys) == 1 and isinstance(f, gtsam.PriorFactorPose3):
            k = int(keys[0])
            if k in seen or not 0 <= k < n:
                raise ValueError("invalid GPS key")
            seen.add(k)
            new = gtsam.GPSFactor(k, gps[k], gtsam.noiseModel.Diagonal.Sigmas(sigma[k]))
            native.add(new)
            corrected.add(new)
        elif len(keys) == 2 and isinstance(f, gtsam.BetweenFactorPose3):
            native.add(f)
            wrapped, cb = numerical_factor(gtsam, f)
            corrected.add(wrapped)
            callbacks.append(cb)
            between_count += 1
        else:
            raise ValueError(f"unexpected factor {j}")
    if seen != set(range(n)) or native.size() != legacy.size() or corrected.size() != legacy.size():
        raise ValueError("factor count drift")
    return native, corrected, callbacks, between_count


def verify_objective(native, corrected, values):
    if native.size() != corrected.size():
        raise ValueError("graph size mismatch")
    for j in range(native.size()):
        if not np.isclose(native.at(j).error(values), corrected.at(j).error(values),
                          rtol=1e-9, atol=1e-8):
            raise ValueError(f"factor objective mismatch: {j}")
    a, b = float(native.error(values)), float(corrected.error(values))
    if not np.isclose(a, b, rtol=1e-10, atol=1e-7):
        raise ValueError("graph objective mismatch")
    return a


def directional_audit(gtsam, native, corrected, values, n):
    """Check negative gradient and two fixed random unit directions.

    Tolerances are derivative consistency only, not a convergence claim.
    """
    vector = corrected.linearize(values).gradientAtZero()
    grad = np.stack([np.asarray(vector.at(k), float).reshape(6) for k in range(n)])
    norm = float(np.linalg.norm(grad))
    if not np.isfinite(norm):
        raise ValueError("nonfinite gradient")
    rng = np.random.default_rng(42)
    directions = []
    if norm > 1e-12:
        directions.append(("negative_gradient", -grad / norm))
    for i in range(2):
        d = rng.normal(size=grad.shape)
        directions.append((f"random_{i}", d / np.linalg.norm(d)))

    def objective(d, h):
        perturbed = gtsam.Values()
        for k in range(n):
            perturbed.insert(k, values.atPose3(k).retract(h * d[k]))
        return float(native.error(perturbed))

    checks = []
    for name, d in directions:
        predicted = float(np.sum(grad * d))
        for h in FD_STEPS:
            actual = (objective(d, h) - objective(d, -h)) / (2 * h)
            error = abs(actual - predicted)
            tol = FD_ATOL + FD_RTOL * max(abs(actual), abs(predicted))
            checks.append(dict(direction=name, step=h, predicted=predicted,
                               actual=actual, absolute_error=error,
                               tolerance=tol, passed=bool(error <= tol)))
    return dict(gradient_l2=norm, checks=checks,
                passed=all(c["passed"] for c in checks))
