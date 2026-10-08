"""Research optimization must preserve the reference ray and tangent algebra."""
from pathlib import Path
import importlib
import numpy as np
import pytest
from wd_spectra._dq import dq_refractive_finite_volume as fv
from wd_spectra._dq import dq_refractive_fv_response as response


@pytest.fixture
def reference(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parent))
    return importlib.import_module('dq_reference_kernels')


@pytest.mark.parametrize('size',[5,17,40,89])
@pytest.mark.parametrize('scale',[1e-6,1.,1e6])
def test_ray_bitwise_equivalence(reference,size,scale):
    rng=np.random.default_rng(11)
    depth=np.geomspace(1e-5,1e3,size)*scale
    g=np.r_[1/(1+.5*depth[0]),2/(depth[:-1]+depth[1:]),1/(1+.5*depth[-1])]
    rhs=rng.normal(size=(size+1,size))
    for a,b in zip(reference.ray_solve(depth,g,rhs),fv.ray_solve(depth,g,rhs)):
        np.testing.assert_array_equal(a,b)


@pytest.mark.parametrize('shape',['monotonic','two_peaks'])
def test_known_tangent_bitwise_equivalence(reference,shape):
    mass=np.geomspace(1e-4,12,19);t=np.linspace(.8,2.,len(mass))
    profile=1-np.exp(-mass)
    if shape=='two_peaks':profile=np.exp(-np.log(mass/.02)**2)+1.4*np.exp(-np.log(mass/2.)**2)
    a=.3*t**.7;s=.5*t**-.3;b=t**4;n=1+.12*profile*t**-.2
    nodes,weights=np.polynomial.legendre.leggauss(4);angles=(nodes+1)/2;weights/=2
    k=a+s;dk=.7*a-.3*s;dn=-.2*(n-1)
    lam,_,vol=fv.operators(mass,k,n,angles,weights)
    eps=a[:-1]/k[:-1];deps=(.7*a[:-1]-eps*dk[:-1])/k[:-1]
    j=np.linalg.solve(np.eye(len(mass)-1)-lam[:,:-1]*(1-eps),lam[:,:-1]@(eps*b[:-1])+lam[:,-1]*b[-1])
    source=np.r_[eps*b[:-1]+(1-eps)*j,b[-1]]
    direct=eps*4*b[:-1]+deps*(b[:-1]-j)
    args=(mass,k,n,dk,dn,source,j,direct,4*b[-1],angles,weights)
    for x,y in zip(reference.known_response(*args),response.known_response(*args)):
        np.testing.assert_array_equal(x,y)
