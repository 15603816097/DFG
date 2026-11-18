from __future__ import annotations

import numpy as np
import pytest

from experiments.isj_supplement.run_future_h3_fgo import (
    FIRST, WINDOW, LEAD, build_h3_arrays, specs_for,
)


def test_horizon_and_fixed_warmup_equal_for_both():
    assert WINDOW == 64
    assert LEAD == 3
    assert FIRST == 66
    n=279
    current=np.linspace(0.2,0.9,n)
    future=np.full(n,np.nan)
    future[FIRST:]=np.linspace(.1,.8,n-FIRST)
    rp,up=build_h3_arrays(current,future,"persist")
    rf,uf=build_h3_arrays(current,future,"predict")
    assert np.array_equal(rp[:FIRST],rf[:FIRST])
    assert np.allclose(rp[:FIRST],.5)
    assert np.array_equal(up,uf)
    assert np.all(up[:FIRST] == 1)
    assert np.all(up[FIRST:] == 0)
    assert rp[FIRST] == pytest.approx(current[63])
    assert rp[-1] == pytest.approx(current[-4])
    assert rf[FIRST] == pytest.approx(future[66])
    assert rf[-1] == pytest.approx(future[-1])
    assert not np.allclose(rp[FIRST:],rf[FIRST:])


def test_identical_forecast_and_persistence_have_identical_covariances():
    n=140
    current=np.linspace(.1,.9,n)
    future=np.full(n,np.nan)
    future[FIRST:]=current[WINDOW-1:n-LEAD]
    rp,up=build_h3_arrays(current,future,"persist")
    rf,uf=build_h3_arrays(current,future,"predict")
    assert np.array_equal(rp,rf)
    assert np.array_equal(up,uf)


def test_invalid_future_cannot_silently_leak():
    n=100
    current=np.ones(n)*.5
    future=np.full(n,np.nan)
    future[FIRST:]=.4
    with pytest.raises(ValueError):
        build_h3_arrays(current,future,"wrong")
    bad=future.copy()
    bad[65]=.4
    with pytest.raises(ValueError):
        build_h3_arrays(current,bad,"predict")
    bad=future.copy()
    bad[66]=np.nan
    with pytest.raises(ValueError):
        build_h3_arrays(current,bad,"predict")
    bad=future.copy()
    bad[66]=1.2
    with pytest.raises(ValueError):
        build_h3_arrays(current,bad,"predict")


def test_spec_names_and_pairing():
    specs=specs_for(("gru","mamba"))
    assert len(specs)==4
    assert len(set(s["name"] for s in specs))==4
    assert {s["source"] for s in specs} == {"v5","v6"}
    assert {s["mode"] for s in specs} == {"persist","predict"}
    for s in specs:
        assert s["type"]=="dynamic"
        assert s["horizon"]==3
