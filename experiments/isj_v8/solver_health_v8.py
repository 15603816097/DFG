from __future__ import annotations

"""V8-only LM solver with honest numerical status reporting.

This module does NOT change the original V7 optimizer or V8.3 archive.

IMPORTANT: "CONVERGED_CANDIDATE" is an *engineering stop condition*,
not a mathematical stationarity certificate or a global optimum. In
particular, a zero-progress LM call is NEVER reported as converged.
The full unscaled GTSAM gradient is always reported for interpretation.
"""

import math
from typing import Any

import numpy as np

from experiments.isj_v7.cross_sequence_v7 import _positions

MAX_OUTER_PASSES = 12
MAX_LM_ITERATIONS_PER_PASS = 100
INNER_REL_TOL = 1e-7
REL_OBJ_TOL = 1e-6
POSITION_SHIFT_TOL_M = 0.01
ROTATION_SHIFT_TOL_RAD = 0.01

# Numerical zero step detection: this threshold is NOT a gradient
# stationarity threshold. The policy is fixed before solver replay.
ZERO_OBJ_ABS_TOL = 1e-10
ZERO_OBJ_REL_TOL = 1e-12
ZERO_POSITION_SHIFT_M = 1e-10
ZERO_ROTATION_SHIFT_RAD = 1e-7
EPS_OBJECTIVE_INCREASE_REL = 1e-8


def _valid_nonnegative(value: Any, name: str) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}: nonnumeric") from exc
    if not math.isfinite(v) or v < 0:
        raise ValueError(f"{name}: nonfinite or negative")
    return v


def gradient_l2(graph, values) -> float:
    """Full GTSAM linearized graph-gradient norm in mixed Pose3 tangents."""
    try:
        norm = graph.linearize(values).gradientAtZero().norm()
    except Exception as exc:
        raise RuntimeError(
            "Cannot calculate GTSAM gradient; refuse convergence claim"
        ) from exc
    return _valid_nonnegative(norm, "GTSAM gradient norm")


def classify_step(
    *,
    objective_before: float,
    objective_after: float,
    translation_shift_rmse_m: float,
    rotation_shift_rmse_rad: float = 0.0,
    gradient_l2_after: float,
) -> dict:
    """Return non-overlapping STALLED, CANDIDATE, or CONTINUE status.

    "STALLED" overrides the old stable=True behavior whenever LM
    returns the same objective and pose with no measurable progress,
    regardless of the nonzero GTSAM gradient. No OXTS reference used.
    """
    before = _valid_nonnegative(objective_before, "objective before")
    after = _valid_nonnegative(objective_after, "objective after")
    shift = _valid_nonnegative(translation_shift_rmse_m, "pose shift")
    rotation_shift = _valid_nonnegative(
        rotation_shift_rmse_rad, "rotation shift")
    grad = _valid_nonnegative(gradient_l2_after, "gradient")

    gain = before-after
    tolerance = ZERO_OBJ_ABS_TOL + ZERO_OBJ_REL_TOL * max(before, 1.0)
    if gain < -EPS_OBJECTIVE_INCREASE_REL * max(before, 1.0):
        raise RuntimeError(
            "LM graph objective increased; cannot accept optimized state"
        )
    no_progress = bool(abs(gain) <= tolerance
                       and shift <= ZERO_POSITION_SHIFT_M
                       and rotation_shift <= ZERO_ROTATION_SHIFT_RAD)
    relative_gain = gain / max(before, 1.0)
    if no_progress:
        status = "STALLED_NO_PROGRESS"
    elif (gain > tolerance and relative_gain <= REL_OBJ_TOL
          and shift <= POSITION_SHIFT_TOL_M
          and rotation_shift <= ROTATION_SHIFT_TOL_RAD):
        status = "CONVERGED_CANDIDATE"
    else:
        status = "CONTINUE"

    return {
        "status": status,
        "objective_before": before,
        "objective_after": after,
        "absolute_objective_decrease": gain,
        "relative_objective_decrease": relative_gain,
        "translation_shift_rmse_m": shift,
        "rotation_shift_rmse_rad": rotation_shift,
        "gradient_l2": grad,
        "gradient_stationarity_certified": False,
        "zero_step_objective_tolerance": tolerance,
        "numerical_zero_step": no_progress,
    }


def solve_v8_health(graph, initial, frame_count: int) -> dict:
    """Solve with original LM settings but a non-misleading outer status.

    Attempts at most 12 LM calls and returns a numerical status even
    if the original LM API declares success with no state update.
    It does NOT change damping, optimize against truth, or rerun cases
    selected using OXTS-relative RMSE.
    """
    import gtsam
    if not isinstance(frame_count, int) or frame_count < 2:
        raise ValueError("invalid Pose3 frame_count")
    before_pose = _positions(gtsam, initial, frame_count)
    if (before_pose.shape != (frame_count, 4, 4)
            or not np.isfinite(before_pose).all()):
        raise ValueError("invalid initial pose sequence")
    state = initial
    initial_gradient = gradient_l2(graph, initial)
    initial_objective = _valid_nonnegative(
        graph.error(initial), "initial objective"
    )
    steps = []
    status = "MAX_PASSES_NOT_CONVERGED"
    for i in range(MAX_OUTER_PASSES):
        objective_before = _valid_nonnegative(
            graph.error(state), "objective before optimization"
        )
        params = gtsam.LevenbergMarquardtParams()
        params.setMaxIterations(MAX_LM_ITERATIONS_PER_PASS)
        params.setRelativeErrorTol(INNER_REL_TOL)
        revised = gtsam.LevenbergMarquardtOptimizer(
            graph, state, params
        ).optimize()
        new_pose = _positions(gtsam, revised, frame_count)
        if (new_pose.shape != before_pose.shape
                or not np.isfinite(new_pose).all()):
            raise RuntimeError("LM returned invalid poses")
        shift = float(np.sqrt(np.mean(np.sum(
            (new_pose[:, :3, 3]-before_pose[:, :3, 3])**2, axis=1
        ))))
        rel_rotation = np.matmul(
            before_pose[:, :3, :3].transpose(0, 2, 1),
            new_pose[:, :3, :3],
        )
        cos_theta = np.clip(
            (np.trace(rel_rotation, axis1=1, axis2=2) - 1.) / 2.,
            -1., 1.,
        )
        rotation_shift = float(np.sqrt(np.mean(np.arccos(cos_theta)**2)))
        obj_after = _valid_nonnegative(
            graph.error(revised), "objective after optimization"
        )
        grad_after = gradient_l2(graph, revised)
        rec = classify_step(
            objective_before=objective_before,
            objective_after=obj_after,
            translation_shift_rmse_m=shift,
            rotation_shift_rmse_rad=rotation_shift,
            gradient_l2_after=grad_after,
        )
        rec["pass"] = i+1
        steps.append(rec)
        state, before_pose = revised, new_pose
        if rec["status"] in ("STALLED_NO_PROGRESS", "CONVERGED_CANDIDATE"):
            status = rec["status"]
            break

    return {
        "status": status,
        "poses": before_pose,
        "values": state,
        "objective": _valid_nonnegative(
            graph.error(state), "final objective"
        ),
        "initial_objective": initial_objective,
        "initial_gradient_l2": initial_gradient,
        "final_gradient_l2": gradient_l2(graph, state),
        "passes": steps,
        "candidate_not_stationarity_certificate": True,
        "legacy_stable_flag_not_used": True,
        "max_outer_passes": MAX_OUTER_PASSES,
    }


def select_reliable_candidate(solutions: dict[str, dict]) -> dict:
    """Use objective selection only *within one fixed target graph*.

    When all initializations stall, DO NOT pick lowest positional RMSE
    or disguise the stalled result as a converged solution.
    """
    if not solutions:
        raise ValueError("no solver candidates")
    eligible = {
        seed: val for seed, val in solutions.items()
        if val.get("status") == "CONVERGED_CANDIDATE"
    }
    if not eligible:
        return {
            "status": "NO_QUALIFIED_CANDIDATE",
            "selected_seed": None,
            "selection_criterion": "none",
            "candidate_count": 0,
        }
    selected = min(
        eligible, key=lambda s: _valid_nonnegative(
            eligible[s]["objective"], "selection objective")
    )
    return {
        "status": "CANDIDATE_SELECTED_NOT_STATIONARITY_CERTIFIED",
        "selected_seed": selected,
        "selection_criterion": "minimum_objective_within_one_fixed_graph",
        "candidate_count": len(eligible),
        "selected_objective": float(eligible[selected]["objective"]),
    }
