from __future__ import annotations

"""Falsifiable GPS-only negative-evidence gating candidate.

Unlike the rejected positive+negative soft-count update, sensor AGREEMENT
is not treated as proof of high absolute-position reliability. Only observed
GPS-vs-LiDAR disagreement can reduce the reliability estimate, never raise it.

Let r be frozen model GPS reliability (Beta posterior mean), U its Beta
vacuity, and q in [0,1] a separate sensor-consistency score. Define
the fraction of contradictory measurement evidence s = 1 - q >= 0.

If the model's effective Beta evidence count is 2/U, adding k*s
unreliability-only pseudo-observations yields:

    r_adjusted = (2*r) / (2 + k*U*s).

For ablation without learned U, replace U with fixed, declared constant
0.5. Control r-only is exactly unchanged. This beta-inspired mechanism
is NOT a calibrated Bayesian sensor-failure posterior, because q is a
heuristic inter-modal consistency score and k has not been estimated
from probabilistic observations.

Use the quality mask returned by cross_modal_consistency; invalid LiDAR
evidence, model warmup, and unpredicted tail frames must not alter r.
Never use GT/clean GPS/FGO posterior residuals to calculate q.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NegativeEvidenceSettings:
    contradiction_count: float = 2.0
    no_u_constant: float = 0.5
    warmup_frames: int = 63
    forecast_horizon: int = 3

    def __post_init__(self):
        for key in ("contradiction_count", "no_u_constant"):
            value = float(getattr(self, key))
            if not np.isfinite(value):
                raise ValueError(f"{key} must be finite")
        if self.contradiction_count < 0:
            raise ValueError("contradiction_count must be nonnegative")
        if not (0 <= self.no_u_constant <= 1):
            raise ValueError("no_u_constant must be in [0,1]")
        if self.warmup_frames < 0 or self.forecast_horizon < 1:
            raise ValueError("Invalid warmup/horizon")


def predicted_mask(n: int, config: NegativeEvidenceSettings = NegativeEvidenceSettings()):
    """Training predictions saved on 0027 are valid at 63..n-3 inclusive."""
    if n <= config.warmup_frames + config.forecast_horizon - 1:
        raise ValueError("Sequence too short for valid model prediction")
    valid = np.zeros(n, dtype=bool)
    valid[
        config.warmup_frames:n-config.forecast_horizon+1
    ] = True
    return valid


def negative_evidence_gate(
    r, u, q, q_valid,
    *,
    variant: str,
    config: NegativeEvidenceSettings = NegativeEvidenceSettings(),
):
    """Return (r_star, effective_multiplier) as new arrays.

    variant: "reliability_only", "no_u", "with_u".
    The model output mask removes warmup/tail; q_valid removes missing
    intermodal geometry. Original arrays are not modified. q=1 => r*=r.
    """
    variants = ("reliability_only", "no_u", "with_u")
    if variant not in variants:
        raise ValueError(f"Unknown negative-evidence variant: {variant}")
    r = np.asarray(r, dtype=np.float64).reshape(-1)
    u = np.asarray(u, dtype=np.float64).reshape(-1)
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    m = np.asarray(q_valid)
    if m.dtype.kind != "b":
        raise ValueError("q_valid must be boolean")
    if not (r.shape == u.shape == q.shape == m.shape and r.size > 0):
        raise ValueError("r/u/q/q_valid must have equal nonempty 1D shape")
    if not all(np.isfinite(x).all() for x in (r,u,q)):
        raise ValueError("Nonfinite GPS gate signal")
    if any(np.any((x<0)|(x>1)) for x in (r,u,q)):
        raise ValueError("GPS gate values must lie in [0,1]")
    pred_valid = predicted_mask(len(r), config)
    effective_mask = pred_valid & m

    if variant == "with_u" and np.any(effective_mask):
        # V5 and V6 implement alpha >= 1, beta >= 1.
        excess = u[effective_mask] - 2*np.minimum(
            r[effective_mask], 1-r[effective_mask]
        )
        if np.any(u[effective_mask] <= 0) or np.any(excess > 1e-5):
            raise ValueError("Saved model r/U violate original Beta constraints")
    adjusted = r.copy()
    multiplier = np.ones_like(r)
    if variant == "reliability_only":
        return adjusted, multiplier

    applied_u = u if variant == "with_u" else np.full_like(
        u, config.no_u_constant
    )
    suspicion = np.maximum(0., 1.-q[effective_mask])
    multiplier[effective_mask] = (
        2.0 / (2.0 + config.contradiction_count *
               applied_u[effective_mask]*suspicion)
    )
    adjusted[effective_mask] = r[effective_mask] * multiplier[effective_mask]
    return adjusted, multiplier
