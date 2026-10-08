import numpy as np
from scipy.special import expn

from wd_spectra._dq.dq_refractive_finite_volume import geometry, solve


def test_transparent_slab_has_one_blackbody_flux_not_two():
    mass=np.linspace(1e-10,1e-6,10)
    a=np.ones((1,len(mass)))
    for n in (np.ones_like(mass),np.linspace(1.,1.3,len(mass))):
        result=solve(mass,a,0.,a,n,n_angle=16)
        # I'_out=B and vacuum surface imply F=pi*B as optical depth ->0,
        # even for refracted rays. P'_bottom=B wrongly gives 2*pi*B.
        np.testing.assert_allclose(result['flux'][0,0],np.pi,rtol=2e-6)
        assert result['maximum_conservation_error']<2e-13


def test_exact_finite_lte_turning_buffer():
    from wd_spectra._dq.dq_refractive_ray_bundles import bundles
    mass=np.array([.001,.1,1.,2.])
    n=np.array([1.,1.5,1.5,1.01]);k=np.ones(4)
    zero=np.zeros(4);t=np.array([.5]);w=np.array([1.]);checked=0
    _,n2=geometry(mass,n)
    levels=np.unique(np.r_[0.,n2,n[-1]**2])
    all_bundles=bundles(mass,k,n,zero,zero,t,w,False)
    for lo,hi in zip(levels[:-1],levels[1:]):
        q=hi-(hi-lo)*.25
        if n[-1]**2<q<n2[-1]:
            from wd_spectra._dq.dq_refractive_ray_bundles import segment_depth_value
            buffer=segment_depth_value(.5,k[-1],n2[-1],n[-1]**2,q)
            # Locate the matching bundle by its invariant-ray weight.
            for first,last,depth,g,dd,dg,weight,dweight in all_bundles:
                if last==2 and np.isclose(weight,(hi-lo)*.5,rtol=1e-13,atol=0.):
                    transmitted=np.exp(-2*buffer)
                    expected_resistance=.5*depth[-1]+(1+transmitted)/(1-transmitted)
                    np.testing.assert_allclose(g[-1],1/expected_resistance,rtol=1e-14)
                    checked+=1
    assert checked>0


def test_nonrefractive_volume_and_exact_cell_energy_conservation():
    mass=np.geomspace(1e-6,30,55)
    a=np.array([.1+mass*.01,.4+mass*.005])
    b=1+.2*mass
    result=solve(mass,a,.7,b,1.,n_angle=12)
    faces=np.r_[0.,.5*(mass[:-1]+mass[1:])]
    np.testing.assert_allclose(result['path_volume'],(a+.7)[:,:-1]*np.diff(faces),rtol=2e-14)
    assert result['maximum_conservation_error']<2e-13
    assert result['maximum_source_error']<2e-13


def test_reflected_volume_and_scattering_energy_conservation():
    mass=np.geomspace(1e-6,30,55)
    index=1+.25*(1-np.exp(-mass))
    a=np.ones((1,len(mass)));b=1+.1*mass
    result=solve(mass,a,.8,b,index,n_angle=48)
    faces,n2=geometry(mass,index)
    # Integral of kappa0*n over each piecewise-linear-n^2 cell.
    expected=(a+.8)[0,:-1]*np.diff(faces)*2/3*(n2[1:]+np.sqrt(n2[1:]*n2[:-1])+n2[:-1])/(np.sqrt(n2[1:])+np.sqrt(n2[:-1]))
    np.testing.assert_allclose(result['path_volume'][0],expected,rtol=2e-4)
    assert result['maximum_conservation_error']<2e-13
    assert result['maximum_source_error']<2e-13


def test_isothermal_nonrefractive_slab_refinement():
    errors=[]
    for count in (100,200,400):
        mass=np.geomspace(1e-7,30,count)
        result=solve(mass,np.ones((1,count)),0.,1.,1.,n_angle=32)
        exact=1-.5*expn(2,mass)
        use=(mass<2)&(mass>.002)
        errors.append(np.max(abs(result['reduced_mean_intensity'][0,use]-exact[use])))
        np.testing.assert_allclose(result['flux'][0,0],np.pi,rtol=8e-3)
    assert errors[1]<.3*errors[0]
    assert errors[2]<.3*errors[1]
    assert errors[-1]<2e-4




def test_trapped_ray_families_and_bottom_response():
    from wd_spectra._dq.dq_refractive_ray_bundles import bundles
    from wd_spectra._dq.dq_refractive_finite_volume import ray_solve
    mass=np.geomspace(1e-5,25.,48)
    n=1+.25*np.exp(-np.log(mass/.02)**2)+.3*np.exp(-np.log(mass/2.)**2)
    t,w=np.polynomial.legendre.leggauss(8);zero=np.zeros(len(mass));closed=0
    for first,last,depth,g,dd,dg,weight,dweight in bundles(
            mass,np.ones_like(mass),n,zero,zero,(t+1)/2,w/2,False):
        if g[0]==0 and g[-1]==0:
            rhs=np.r_[depth,1.][:,None]
            intensity,flux=ray_solve(depth,g,rhs)
            np.testing.assert_allclose(intensity,1.,rtol=1e-11)
            np.testing.assert_allclose(flux,0.,atol=1e-10)
            closed+=1
    assert closed>0
    a=np.ones((1,len(mass)));b=a.copy()
    result=solve(mass,a,.7,b,n,n_angle=8)
    assert result['maximum_conservation_error']<2e-13
    # Only the bottom LTE reservoir/buffer emits; linear response must
    # reproduce a separate full scattering solve, without flux rescaling.
    b[:,:-1]=0.
    bottom=solve(mass,a,.7,b,n,n_angle=8)
    np.testing.assert_allclose(result['boundary_surface_flux'],bottom['flux'][:,0],rtol=2e-8,atol=1e-13)


def test_independent_prescribed_source_sweep_matches_coupled_solution():
    from wd_spectra._dq.dq_refractive_finite_volume import formal_field
    mass=np.geomspace(1e-6,20,40)
    n=1+.2*np.exp(-np.log(mass/2.)**2)
    a=np.array([.1+mass*.01,.2+mass*.03]);s=.9*a;b=1+.3*mass
    coupled=solve(mass,a,s,b,n,n_angle=8)
    source=(a*b+s*coupled['reduced_mean_intensity'])/(a+s)
    formal=formal_field(mass,a+s,source,n,n_angle=8)
    np.testing.assert_allclose(formal['reduced_mean_intensity'],coupled['reduced_mean_intensity'],rtol=2e-12)
    np.testing.assert_allclose(formal['flux'],coupled['flux'],rtol=2e-11,atol=1e-11)
    assert formal['maximum_conservation_error']<1e-12


def test_deep_diffusion_limit_matches_published_refractive_rosseland_factor():
    mass=np.linspace(1e-5,40.,160);slope=.01
    b=(1+slope*mass)[None,:];a=np.ones_like(b)
    for index in (1.,1.15,1.3):
        # Physical Rayleigh sigma=.5; reduced-transfer input sigma0=n*.5.
        result=solve(mass,a,index*.5,b,index,n_angle=16)
        expected=4*np.pi*index**3*slope/(3*(1+index*.5))
        faces=np.r_[0.,.5*(mass[:-1]+mass[1:])]
        use=(faces>12)&(faces<25)
        np.testing.assert_allclose(result['flux'][0,use],expected,rtol=2e-4)
