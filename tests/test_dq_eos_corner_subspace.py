from types import SimpleNamespace
import numpy as np
from wd_spectra._dq.dq_eos_corner_subspace import corner_columns, refine_corner_subspace


def test_corner_subspace_makes_progress_without_deleting_equations():
    s=SimpleNamespace(n=2,eos_trial_owner=SimpleNamespace(helium_reos3=
        SimpleNamespace(temperature_grid=np.exp([-1.,0.,1.]))))
    state=np.array([-.0002,.3,.0])
    def evaluated(d):
        return SimpleNamespace(residual=np.array([1.,d[1]-.01,d[2]-.02]),
            jacobian=np.diag([100.,1.,1.]))
    solved=SimpleNamespace(x=np.zeros(3),cost=.50025)
    best,info=refine_corner_subspace(s,state,solved,.04,evaluated)
    assert info['eos_subspace_selected']
    np.testing.assert_allclose(best.x,[0.,.01,.02],atol=1e-9)
    assert best.cost>=.5  # Unresolved physical row is not removed.
    np.testing.assert_array_equal(corner_columns(s,state[:2]),[0])
    assert not len(corner_columns(s,state[:2]+.01))  # Active set released.


def test_no_corner_has_no_extra_optimization():
    s=SimpleNamespace(n=1,eos_trial_owner=None)
    solved=SimpleNamespace(x=np.zeros(2))
    best,info=refine_corner_subspace(s,np.zeros(2),solved,.04,
        lambda d: (_ for _ in ()).throw(AssertionError('unused')))
    assert best is solved and info=={}


def test_corner_refinement_preserves_coordinate_specific_bounds():
    s=SimpleNamespace(n=2,eos_trial_owner=SimpleNamespace(helium_reos3=
        SimpleNamespace(temperature_grid=np.exp([-1.,0.,1.]))))
    state=np.array([-.0002,.3,.0])
    def evaluated(d):
        return SimpleNamespace(residual=np.array([1.,d[1]-.01,d[2]+.3]),
            jacobian=np.diag([100.,1.,1.]))
    solved=SimpleNamespace(x=np.array([0.,0.,-.2]),cost=.6)
    lower=np.array([-.04,-.04,-np.inf]);upper=np.full(3,.04)
    best,info=refine_corner_subspace(s,state,solved,.04,evaluated,
        bounds=(lower,upper))
    assert info['eos_subspace_selected']
    np.testing.assert_allclose(best.x,[0.,.01,-.3],atol=1e-8)
