from __future__ import annotations

import numpy as np
import pytest

from experiments.isj_supplement import audit_dev_v5_v6_provenance as audit


def test_static_future_horizon_mismatch_and_train_val_crossing():
    x=audit.source_timing_audit(4544)
    assert x["GRU_V5"]["TRAIN"]["range"]==(63,3180)
    assert x["GRU_V5"]["VAL"]["range"]==(3180,3862)
    assert x["MAMBA_V6"]["TRAIN"]["range"]==(63,3179)
    assert x["MAMBA_V6"]["VAL"]["range"]==(3179,3860)
    for model,splits in x.items():
        for name in ("TRAIN","VAL","TEST"):
            assert splits[name]["head_target_offsets"]==(0,3,2)
        for name in ("TRAIN","VAL"):
            start,end=splits[name]["range"]
            flagged=splits[name]["indices_with_label_beyond_split"]
            assert flagged["h1_stored_future_t_plus_3"]==[
                end-3,end-2,end-1
            ]
            assert flagged["h2_shifted_current_t_plus_2"]==[
                end-2,end-1
            ]
            assert start<end-3
    with pytest.raises(ValueError,match="Insufficient"):
        audit.source_timing_audit(40)


def test_v5_expected_mean_std_vs_v6_median_iqr_differ():
    n=4544
    x=np.zeros((n,2),dtype=np.float32)
    x[:,0]=np.linspace(0,1,n)
    x[:,1]=(np.arange(n)%11).astype(np.float32)
    x[0,1]=1000.
    v5mu,v5sd=audit.v5_expected_norm(x)
    v6mu,v6sd=audit.v6_expected_norm(x)
    assert v5mu.shape==(1,2)
    assert v6mu.shape==(2,)
    assert np.all(v5sd>0)
    assert np.all(v6sd>0)
    assert not np.allclose(v5mu.reshape(-1),v6mu)
    assert not np.allclose(v5sd.reshape(-1),v6sd)
    v5ck={"mean":v5mu,"std":v5sd}
    v6ck={"mean":v6mu,"scale":v6sd}
    assert audit.scaler_match(v5ck,x,"GRU_V5")[0]
    assert audit.scaler_match(v6ck,x,"MAMBA_V6")[0]
    assert not audit.scaler_match(v5ck,x,"MAMBA_V6")[0]
    assert not audit.scaler_match(v6ck,x,"GRU_V5")[0]


def test_placeholder_validation_matches_train_source_code():
    n=audit.N_EXPECTED
    r=np.linspace(.2,.9,n)
    u=np.linspace(.5,.8,n)
    r[:63]=.5
    r[-2:]=.5
    u[:63]=1.
    u[-2:]=1.
    assert audit.prediction_placeholder_report(r,u,"GRU_V5")==(
        True,True
    )
    r[:63]=1.
    r[-2:]=1.
    assert audit.prediction_placeholder_report(r,u,"MAMBA_V6")==(
        True,True
    )
    r[0]=.6
    assert not audit.prediction_placeholder_report(
        r,u,"MAMBA_V6"
    )[0]
    with pytest.raises(ValueError,match="length"):
        audit.prediction_placeholder_report(r[:-1],u,"MAMBA_V6")


def test_read_candidate_reports_absence_without_creating_path(tmp_path):
    missing=tmp_path/"no_dataset.npz"
    assert audit.read_candidate(missing) is None
    assert not missing.exists()
    # Create a deliberately invalid sensor-wise data archive. There
    # must be no implicit fallback, data substitution or write.
    p=tmp_path/"wrong.npz"
    np.savez_compressed(p,gps_features=np.ones((audit.N_EXPECTED,21)))
    with pytest.raises(KeyError,match="missing"):
        audit.read_candidate(p)


def test_hash_is_read_only_and_changes_with_contents(tmp_path):
    a=tmp_path/"a"
    a.write_bytes(b"test")
    first=audit.file_hash(a)
    assert len(first)==64
    assert first==audit.file_hash(a)
    a.write_bytes(b"test2")
    assert audit.file_hash(a)!=first
