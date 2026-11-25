import numpy as np
import pytest
import gtsam

from experiments.isj_v8.backend_v9 import (
    COLD, HUBER_K, map_h0, build_graphs, solve_uniform, directional_audit_v9,
)
from experiments.isj_v8.backend_c8_numeric import (
    yaw_aligned_seed, verify_objective, directional_audit,
)


def fit():
    return {"centers": [.1,.3,.5,.7,.9],
            "sigma_bins_xyz": [[8,7,3],[5,4,2],[3,2,1],[1,.8,.4],[.1,.1,.1]]}


@pytest.mark.parametrize("n",[64,279,680,1104])
def test_variable_length_mapping(n):
    fixed=np.array([4.,3.,2.])
    a=np.arange(COLD,n)
    p=np.linspace(.01,.99,len(a))
    s=map_h0(a,p,n,fit(),fixed)
    assert s.shape==(n,3)
    assert np.array_equal(s[:COLD],np.tile(fixed,(COLD,1)))
    assert np.all((fixed/s)**2<=4.+1e-12)
    assert np.all(np.diff(s[COLD:],axis=0)<=1e-12)
    assert np.allclose(s[COLD:],np.maximum(
        np.column_stack([np.interp(p,fit()["centers"],
                                   np.array(fit()["sigma_bins_xyz"])[:,i])
                         for i in range(3)]),fixed/2.))


def test_mapping_rejects_wrong_anchor():
    with pytest.raises(ValueError):
        map_h0(np.arange(64,70),np.ones(6),70,fit(),[4,3,2])


def graph_fixture(n):
    poses=np.repeat(np.eye(4)[None],n,axis=0)
    t=np.arange(n,dtype=float)
    poses[:,:3,3]=np.column_stack([t,.02*t*t,.01*t])
    r=gtsam.Rot3.Rz(.8).matrix()
    gps=poses[:,:3,3]@r.T
    gps[n//2]+=np.array([9.,-5.,1.])
    sigma=np.tile([1.,2.,3.],(n,1))
    graph=gtsam.NonlinearFactorGraph()
    noise=gtsam.noiseModel.Isotropic.Sigma(6,.2)
    graph.add(gtsam.PriorFactorPose3(0,gtsam.Pose3(),noise))
    for k in range(n):
        graph.add(gtsam.PriorFactorPose3(k,gtsam.Pose3(gtsam.Rot3(),gps[k]),noise))
        if k:
            graph.add(gtsam.BetweenFactorPose3(k-1,k,gtsam.Pose3(
                gtsam.Rot3(),poses[k,:3,3]-poses[k-1,:3,3]),noise))
    return graph,poses,gps,sigma


@pytest.mark.parametrize("n",[8,17])
def test_huber_graph_exact_cost_and_derivative(n):
    legacy,poses,gps,sigma=graph_fixture(n)
    a,b,keep,count=build_graphs(gtsam,legacy,gps,sigma,huber=True)
    x,_=yaw_aligned_seed(gtsam,poses,gps)
    assert a.size()==b.size()==legacy.size()
    assert count==n-1 and len(keep)==n
    expected_count=0
    for j in range(a.size()):
        f=a.at(j)
        if isinstance(f,gtsam.GPSFactor):
            k=int(f.keys()[0])
            norm=np.linalg.norm((x.atPose3(k).translation()-gps[k])/sigma[k])
            expected=.5*norm**2 if norm<=HUBER_K else HUBER_K*(norm-.5*HUBER_K)
            assert np.isclose(f.error(x),expected,atol=1e-8,rtol=1e-8)
            expected_count+=1
    assert expected_count==n
    assert verify_objective(a,b,x)>0.
    audit=directional_audit(gtsam,a,b,x,n)
    assert audit["passed"],audit


def test_uniform_solver_huber():
    legacy,poses,gps,sigma=graph_fixture(8)
    a,b,keep,_=build_graphs(gtsam,legacy,gps,sigma,huber=True)
    x,_=yaw_aligned_seed(gtsam,poses,gps)
    res=solve_uniform(gtsam,a,b,x,8)
    assert res["poses"].shape==(8,4,4)
    assert res["derivative_audit"]["passed"]
    assert res["last_step_stable"]
    assert res["stationarity_certified"] is False
    for name in ("warm_derivative_audit", "continuation_derivative_audit", "derivative_audit"):
        audit = res[name]
        assert audit["passed"]
        assert {c["step"] for c in audit["checks"] if c["required"]} == {1e-5, 1e-6}
        assert {c["step"] for c in audit["checks"] if not c["required"]} == {1e-4}
        assert all(c["passed"] for c in audit["checks"] if c["required"])


def test_v9_audit_rejects_wrong_jacobian():
    # Identical costs but deliberately incorrect Jacobian: the audit
    # must reject this, not merely accept all sufficiently small steps.
    def callback(this, values, H):
        p = values.atPose3(0)
        if H is not None:
            H[0] = np.zeros((3, 6), dtype=float, order="F")
        return p.translation() - np.array([1., 2., 3.])
    a = gtsam.NonlinearFactorGraph()
    b = gtsam.NonlinearFactorGraph()
    noise = gtsam.noiseModel.Isotropic.Sigma(3, 1.)
    a.add(gtsam.GPSFactor(0, np.array([1., 2., 3.]), noise))
    b.add(gtsam.CustomFactor(noise, [0], callback))
    x = gtsam.Values()
    x.insert(0, gtsam.Pose3())
    audit = directional_audit_v9(gtsam, a, b, x, 1)
    assert not audit["passed"]
    assert any(not c["passed"] for c in audit["checks"] if c["required"])
