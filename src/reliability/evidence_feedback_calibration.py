from __future__ import annotations

"""Small, deterministic Beta-evidence/measurement-feedback research prototype.

Not a validated new ISJ contribution yet. The feedback quality MUST be
computed from an independently available, correctly aligned pre-update
measurement-consistency statistic (e.g. a separate modality). Never pass
evaluation ground truth, or post-fit residuals of the same factor and call
the method "causal". A full-trajectory Stage-1 residual is only an OFFLINE
two-stage diagnostic, NOT a streaming measurement innovation.

The original network outputs beta mean r and vacuity U = 2/(alpha+beta).
Interpret feedback quality q in [0,1] as a soft pseudo-observation with
strength k>=0 and form an evidence-count update:

    alpha = (2/U)*r, beta=(2/U)*(1-r)
    alpha' = alpha + k*q, beta' = beta + k*(1-q)
    r' = (2*r + k*U*q)/(2+k*U)

As a simple, optional deterministic quality proxy, apply a cutoff to an
independently calculated normalized innovation squared (NIS):
    q = min(1, sqrt(chi_square_cutoff / NIS)).

This is a candidate mechanism, NOT a proof that U is calibrated or that
an NIS-based quality score is a probability. Validate on development
data and prospectively held-out sequences; compare vs reliability-only,
r+feedback without U, and standard robust FGO.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CalibrationSettings:
    pseudo_count: float = 2.0
    expected_warmup: int = 63

    def __post_init__(self):
        if not np.isfinite(self.pseudo_count) or self.pseudo_count < 0:
            raise ValueError("pseudo_count must be finite and nonnegative")
        if self.expected_warmup < 0:
            raise ValueError("expected_warmup must be nonnegative")


def normalized_innovation_quality(nis, chi2_cutoff):
    """q=1 in the accepted region, decreases outside; output [0,1].

    The input must already be squared Mahalanobis distance under a
    defensible covariance model. For 3D and 6D 95% chi-square cutoff,
    previous repository code used 2.795**2 and 3.548**2 respectively.
    """
    a = np.asarray(nis, dtype=np.float64)
    cutoff = float(chi2_cutoff)
    if a.ndim != 1 or len(a) == 0:
        raise ValueError("nis must be a nonempty 1D array")
    if not np.all(np.isfinite(a)) or np.any(a < 0):
        raise ValueError("nis must contain finite nonnegative values")
    if not np.isfinite(cutoff) or cutoff <= 0:
        raise ValueError("chi2_cutoff must be positive finite")
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.sqrt(cutoff / np.maximum(a, cutoff))
    return np.clip(q, 0.0, 1.0)


def beta_soft_evidence_update(
    reliability,
    uncertainty,
    feedback_quality,
    *,
    valid_feedback=None,
    settings: CalibrationSettings = CalibrationSettings(),
):
    """Return (r_updated, feedback_weight) for *valid model windows only*.

    Shapes must match; all values must be finite, in [0,1].
    For well-defined Beta evidence, alpha>=1 and beta>=1 impose
    U<=2*min(r,1-r). This routine checks that constraint for every
    valid predicted sample. Warmup frames and unavailable independent
    feedback are left exactly unchanged.

    Output r_updated is the beta mean after a soft-count update, NOT
    an independently calibrated measurement variance.
    """
    r = np.asarray(reliability, dtype=np.float64)
    u = np.asarray(uncertainty, dtype=np.float64)
    q = np.asarray(feedback_quality, dtype=np.float64)
    if not (r.ndim == u.ndim == q.ndim == 1
            and r.shape == u.shape == q.shape and len(r) > 0):
        raise ValueError("r, U, feedback_quality must be equal nonempty 1D arrays")
    if not all(np.isfinite(v).all() for v in (r, u, q)):
        raise ValueError("Nonfinite input to evidence calibration")
    if any(np.any((v < 0) | (v > 1)) for v in (r, u, q)):
        raise ValueError("r, U and feedback_quality must be in [0,1]")

    n = len(r)
    if valid_feedback is None:
        mask = np.ones(n, dtype=bool)
    else:
        mask = np.asarray(valid_feedback)
        if mask.shape != (n,) or mask.dtype.kind != "b":
            raise ValueError("valid_feedback must be a boolean 1D mask")
        mask = mask.copy()
    mask[:settings.expected_warmup] = False

    if np.any(mask):
        # Alpha = 2*r/U, beta = 2*(1-r)/U require >=1.
        # U=0 is an ideal infinite-evidence limit, which has no correction.
        if np.any(u[mask] > 2.0 * np.minimum(r[mask], 1.0-r[mask]) + 1e-5):
            raise ValueError("Beta evidence invalid (alpha/beta < 1)")
    weight = np.zeros(n, dtype=np.float64)
    weight[mask] = (
        settings.pseudo_count*u[mask] /
        (2.0 + settings.pseudo_count*u[mask])
    )
    updated = r.copy()
    updated[mask] = (1.0 - weight[mask])*r[mask] + weight[mask]*q[mask]
    return np.clip(updated, 0.0, 1.0), weight
