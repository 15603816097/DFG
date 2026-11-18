from __future__ import annotations
import json
import numpy as np
import pytest
from experiments.isj_supplement.run_dev_gps_negative_evidence_fgo import (
    VARIANTS,MODELS,cache_check,case_name,
)

def test_three_predeclared_cases_and_sources():
    assert VARIANTS==("reliability_only","no_u","with_u")
    assert MODELS=={"gru":"v5","mamba":"v6"}
    assert len(set(map(case_name,VARIANTS)))==3

def test_new_dev_cache_refuses_any_partial_or_mismatched_result(tmp_path):
    dest=tmp_path/"GRU"
    spec={"source_model":"gru","variant":"no_u"}
    assert cache_check(dest,spec,80) is None
    dest.mkdir(parents=True)
    with pytest.raises(RuntimeError,match="Partial"):
        cache_check(dest,spec,80)
    poses=np.repeat(np.eye(4)[None],80,axis=0)
    np.save(dest/"poses.npy",poses)
    (dest/"metrics.json").write_text(json.dumps({
        "spec":spec,"metrics":{"ATE3D":1.2}
    }))
    assert cache_check(dest,spec,80)["metrics"]["ATE3D"]==1.2
    with pytest.raises(ValueError,match="mismatch"):
        cache_check(dest,{**spec,"variant":"with_u"},80)
    np.save(dest/"poses.npy",np.eye(4)[None])
    with pytest.raises(ValueError,match="Invalid"):
        cache_check(dest,spec,80)
