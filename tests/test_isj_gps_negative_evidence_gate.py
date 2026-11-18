from __future__ import annotations
import numpy as np
import pytest
from src.reliability.gps_negative_evidence_gate import (
    NegativeEvidenceSettings, predicted_mask, negative_evidence_gate,
)

def input_data(n=90):
    r=np.full(n,0.8)
    u=np.full(n,0.2)
    q=np.ones(n)
    q[70]=0.25
    mask=np.ones(n,dtype=bool)
    return r,u,q,mask

def test_exact_no_boost_when_no_disagreement():
    r,u,q,mask=input_data()
    q[:]=1
    for case in ("reliability_only","no_u","with_u"):
        out,mult=negative_evidence_gate(r,u,q,mask,variant=case)
        assert np.array_equal(out,r)
        assert np.allclose(mult,1)

def test_only_conflicting_valid_frames_are_downweighted():
    r,u,q,mask=input_data()
    a,_=negative_evidence_gate(r,u,q,mask,variant="reliability_only")
    b,_=negative_evidence_gate(r,u,q,mask,variant="no_u")
    c,_=negative_evidence_gate(r,u,q,mask,variant="with_u")
    assert a[70]==pytest.approx(0.8)
    assert b[70] ==pytest.approx(1.6/(2+2*.5*.75))
    assert c[70] ==pytest.approx(1.6/(2+2*.2*.75))
    assert b[70]<c[70]<a[70]
    for t in (0,30,62,63,64,87,88,89):
        assert b[t]==a[t]
        assert c[t]==a[t]
    assert np.array_equal(r,np.full_like(r,.8))

def test_uncertainty_changes_penalty_only_if_disagreement():
    r,u,q,mask=input_data()
    u[70]=0.5  # r=.8 implies Beta-valid U<=.4; must reject
    with pytest.raises(ValueError,match="Beta"):
        negative_evidence_gate(r,u,q,mask,variant="with_u")
    u[70]=0.4
    out,_=negative_evidence_gate(r,u,q,mask,variant="with_u")
    assert out[70]<r[70]
    mask[70]=False
    neutral,_=negative_evidence_gate(r,u,q,mask,variant="with_u")
    assert np.array_equal(neutral,r)

def test_predicted_frame_range_and_invalid_inputs():
    m=predicted_mask(90)
    assert not m[:63].any()
    assert m[63:88].all()
    assert not m[88:].any()
    with pytest.raises(ValueError):
        predicted_mask(65)
    with pytest.raises(ValueError):
        NegativeEvidenceSettings(contradiction_count=-1)
    r,u,q,mask=input_data()
    with pytest.raises(ValueError,match="boolean"):
        negative_evidence_gate(r,u,q,np.ones(len(r),int),variant="with_u")
    with pytest.raises(ValueError,match="Unknown"):
        negative_evidence_gate(r,u,q,mask,variant="oops")
