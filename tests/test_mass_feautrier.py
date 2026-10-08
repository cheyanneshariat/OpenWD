"""Independent conservation and tangent tests for the opt-in mass operator."""
import numpy as np
import pytest
from wd_spectra._compat import trapezoid
from wd_spectra.opacity import optical_depth_from_mass_opacity
from wd_spectra._stable_feautrier import cancellation_safe_field
from wd_spectra._mass_feautrier import mass_field, mass_energy, mass_energy_response
from wd_spectra._mass_feautrier import (mass_emissivity_field, mass_emissivity_energy,
    mass_response, MassResponseOperator, InvalidRadiationFieldError)


@pytest.mark.parametrize("n_angle", [2, 3, 4])
def test_pg1159_gray_lte_flux_normalization_at_supported_angle_orders(n_angle):
    """Rule out a shared angular or 4-pi normalization error in PG1159 transfer."""
    from types import SimpleNamespace
    from wd_spectra._pg1159_transfer import transfer_field
    from wd_spectra.constants import STEFAN_BOLTZMANN
    from wd_spectra.nlte_core import NLTETransferCoefficients
    from wd_spectra.spectrum import planck_lambda_angstrom

    effective_temperature = 1.0e5
    tau = np.geomspace(1.0e-7, 1.0e3, 80)
    temperature = (
        0.75 * effective_temperature**4 * (tau + 2.0 / 3.0)
    ) ** 0.25
    atmosphere = SimpleNamespace(column_mass=tau, n_depth=tau.size)
    wavelength = np.geomspace(10.0, 1.0e6, 400)
    planck = planck_lambda_angstrom(
        wavelength[:, None], temperature[None, :]
    )
    absorption = np.ones_like(planck)
    coefficients = NLTETransferCoefficients(
        wavelength,
        absorption,
        absorption * planck,
        np.zeros_like(planck),
        {},
    )

    _, field, _ = transfer_field(
        atmosphere, coefficients, n_angle=n_angle, check_source=False
    )
    normalized = trapezoid(field.interface_flux, wavelength, axis=0) / (
        STEFAN_BOLTZMANN * effective_temperature**4
    )

    # The Eddington gray temperature law is not the exact discrete-ordinate
    # solution near the surface, so its interior profile differs by a few
    # percent.  Both boundaries retain the absolute sigma Teff^4
    # normalization, and the entire profile rules out the proposed ~42%
    # angular-quadrature error.
    np.testing.assert_allclose(normalized[[0, -1]], 1.0, rtol=3.0e-4)
    assert np.max(abs(normalized - 1.0)) < 0.04


@pytest.mark.parametrize('angles',[1,3])
@pytest.mark.parametrize('gain',[False,True])
def test_frozen_response_operator_reuses_anchor_for_distinct_material_directions(angles,gain):
    mass,wave,a,s,eta,tau,source,field=_gain_slab(np.zeros(5))
    if not gain:
        a=np.abs(a)+.1
        tau=optical_depth_from_mass_opacity(mass,a+s)
    ext=a+s;fraction=s/ext
    frozen=MassResponseOperator(tau,wave,source,fraction,mass,extinction=ext,
        n_angle=angles,wavelength_chunk_size=2,allow_stimulated_gain=gain)
    rng=np.random.default_rng(159)
    for _ in range(2):
        direct,db,dk=rng.normal(size=(3,)+source.shape)
        expected=mass_response(tau,wave,source,direct,db,fraction,mass,dk,
            extinction=ext,n_angle=angles,wavelength_chunk_size=2,allow_stimulated_gain=gain)
        actual=frozen.apply(direct,db,dk)
        for before,after in zip(expected,actual):
            np.testing.assert_array_equal(before,after)
    # A caller preparing its next atmosphere must not mutate the saved anchor.
    source[:]=0.;tau[:]*=2;ext[:]*=3;fraction[:]=0.
    again=frozen.apply(direct,db,dk)
    for before,after in zip(actual,again):
        np.testing.assert_array_equal(before,after)


def _gain_slab(x):
    mass=np.array([.02,.1,.4,1.,3.]);wave=np.array([1000.,4000.,9000.])
    a=np.broadcast_to(np.array([-.1,-.02,0.,.2,.3]),(3,5))+.03*x
    s=np.full((3,5),.5)*np.exp(-.1*x)
    eta=np.broadcast_to(np.array([.01,.03,.1,.4,1.]),(3,5))*np.exp(2*x)
    tau=optical_depth_from_mass_opacity(mass,a+s)
    source,field=mass_emissivity_field(tau,eta,a,s,column_mass=mass,
        bottom_source=eta[:,-1]/a[:,-1],n_angle=1)
    return mass,wave,a,s,eta,tau,source,field


def test_signed_absorption_matches_independent_matrix_and_energy_conservation():
    mass,wave,a,s,eta,tau,source,field=_gain_slab(np.zeros(5))
    # Assemble the original second-order difference equations independently.
    # This includes an exactly zero absorption cell and two gain cells.
    mu=.5;h=np.diff(tau[0],prepend=0.);mh=np.diff(mass,prepend=0.)
    matrix=np.zeros((6,6));rhs=np.zeros(6)
    matrix[0,0]=1+mu/h[0];matrix[0,1]=-mu/h[0]
    for i in range(4):
        volume=(a+s)[0,i]*.5*(mh[i]+mh[i+1])
        left=mu**2/(h[i]*volume);right=mu**2/(h[i+1]*volume)
        matrix[i+1,i]=-left;matrix[i+1,i+1]=left+right+a[0,i]/(a+s)[0,i]
        matrix[i+1,i+2]=-right;rhs[i+1]=eta[0,i]/(a+s)[0,i]
    matrix[-1,-1]=1;rhs[-1]=eta[0,-1]/a[0,-1]
    u=np.linalg.solve(matrix,rhs)
    np.testing.assert_allclose(field.mean_intensity[0],u[1:],rtol=2e-13)
    np.testing.assert_allclose(field.interface_flux[0],4*np.pi*mu**2*np.diff(u)/h,rtol=2e-12)
    energy,_=mass_emissivity_energy(wave,mass,eta,field.mean_intensity,a)
    np.testing.assert_allclose(energy,np.diff(trapezoid(field.interface_flux,wave,axis=0)),rtol=2e-12,atol=1e-9)
    _,boundary=mass_emissivity_field(tau,np.zeros_like(eta),a,s,column_mass=mass,
        bottom_source=eta[:,-1]/a[:,-1],n_angle=1)
    _,volume=mass_emissivity_field(tau,eta,a,s,column_mass=mass,bottom_source=np.zeros(3),n_angle=1)
    np.testing.assert_allclose(field.interface_flux,boundary.interface_flux+volume.interface_flux,rtol=2e-12,atol=1e-13)


def test_signed_absorption_material_response_crosses_zero_without_singularity():
    mass,wave,a,s,eta,tau,source,field=_gain_slab(np.zeros(5))
    da=np.full_like(a,.03);ds=-.1*s;deta=2*eta;ext=a+s
    direct=(deta+ds*field.mean_intensity-(da+ds)*source)/ext
    db=np.zeros_like(eta);db[:,-1]=(deta[:,-1]*a[:,-1]-eta[:,-1]*da[:,-1])/a[:,-1]**2
    fj,mj,_=mass_response(tau,wave,source,direct,db,s/ext,mass,da+ds,
        extinction=ext,n_angle=1,allow_stimulated_gain=True)
    for i in range(5):
        dx=np.eye(5)[i]*1e-5
        plus=_gain_slab(dx)[-1];minus=_gain_slab(-dx)[-1]
        np.testing.assert_allclose(mj[:,:,i],(plus.mean_intensity-minus.mean_intensity)/2e-5,rtol=3e-7,atol=2e-7)
        fd=trapezoid((plus.interface_flux-minus.interface_flux)/2e-5,wave,axis=0)
        np.testing.assert_allclose(fj[:,i],fd,rtol=3e-7,atol=2e-5)


def test_excessive_stimulated_gain_is_not_clipped_into_a_valid_field():
    mass=np.geomspace(.01,100.,12);a=np.full((1,12),-.1);a[:,-1]=.1
    s=np.full_like(a,.2);eta=np.ones_like(a)
    tau=optical_depth_from_mass_opacity(mass,a+s)
    with pytest.raises(InvalidRadiationFieldError):
        mass_emissivity_field(tau,eta,a,s,column_mass=mass,bottom_source=np.array([10.]),n_angle=1)


@pytest.mark.parametrize('angles', [1, 2, 3])
def test_small_positive_intensity_below_bright_boundary_matches_high_precision(angles):
    """Do not reconstruct a faint cell as the difference of two ~1e16 values."""
    import mpmath as mp
    from wd_spectra.radiative_transfer import angular_quadrature
    mass = np.array([1e-7, 1., 1e4])
    ext = np.array([[1., 1e4, 1e4]])
    absorption = .8 * ext
    scattering = .2 * ext
    emissivity = np.full_like(ext, 1e-30)
    tau = optical_depth_from_mass_opacity(mass, ext)
    _, field = mass_emissivity_field(
        tau, emissivity, absorption, scattering, column_mass=mass,
        bottom_source=np.array([1e16]), n_angle=angles,
    )
    mu, weights = angular_quadrature(angles)
    # Assemble the original equations independently at 80-digit precision.
    # This checks the intensity AND the retained depth increments/fluxes.
    with mp.workdps(80):
        m = list(map(mp.mpf, mass))
        t = list(map(mp.mpf, tau[0]))
        h = [t[0]] + [t[i] - t[i-1] for i in range(1, len(t))]
        mh = [m[0]] + [m[i] - m[i-1] for i in range(1, len(m))]
        matrix = mp.matrix(4 * angles)
        rhs = mp.matrix(4 * angles, 1)
        for r in range(angles):
            cosine = mp.mpf(float(mu[r]))
            matrix[r, r] = 1 + cosine/h[0]
            matrix[r, angles+r] = -cosine/h[0]
            for i in range(2):
                row = (i+1)*angles+r
                volume = mp.mpf(float(ext[0, i])) * (mh[i]+mh[i+1])/2
                left = cosine**2/(h[i]*volume)
                right = cosine**2/(h[i+1]*volume)
                matrix[row, i*angles+r] = -left
                matrix[row, row] = 1+left+right
                matrix[row, (i+2)*angles+r] = -right
                for q in range(angles):
                    matrix[row, (i+1)*angles+q] -= (
                        mp.mpf(float(scattering[0, i]/ext[0, i]))
                        * mp.mpf(float(weights[q]))
                    )
                rhs[row] = mp.mpf(float(emissivity[0, i]/ext[0, i]))
            matrix[3*angles+r, 3*angles+r] = 1
            rhs[3*angles+r] = mp.mpf('1e16')
        u = mp.lu_solve(matrix, rhs)
        mean = [float(sum(mp.mpf(float(weights[r])) * u[(i+1)*angles+r]
                          for r in range(angles))) for i in range(3)]
        flux = [float(4*mp.pi*sum(
            mp.mpf(float(weights[r]))*mp.mpf(float(mu[r]))**2
            * (u[(i+1)*angles+r]-u[i*angles+r])/h[i]
            for r in range(angles))) for i in range(3)]
    assert np.all(field.mean_intensity > 0)
    np.testing.assert_allclose(field.mean_intensity[0], mean, rtol=3e-13, atol=0)
    np.testing.assert_allclose(field.interface_flux[0], flux, rtol=3e-12, atol=0)


@pytest.mark.parametrize('chunk',[0,-1,True,1.5])
def test_invalid_chunks_are_rejected(chunk):
    with pytest.raises(ValueError,match='chunk_size'):
        mass_field(np.array([[.1,1.]]),np.ones((1,2)),1.,0.,column_mass=np.array([.1,1.]),
            wavelength_chunk_size=chunk)


@pytest.mark.parametrize('mass',[[0.,1.],[1.,1.],[2.,1.],[1.,np.nan]])
def test_invalid_mass_grid_is_rejected(mass):
    with pytest.raises(ValueError,match='column mass'):
        mass_field(np.array([[.1,1.]]),np.ones((1,2)),1.,0.,column_mass=np.array(mass))


@pytest.mark.parametrize('epsilon', [1., .1, 1e-6])
def test_constant_extinction_reproduces_optical_depth_discretization(epsilon):
    mass=np.geomspace(1e-12,100.,40)
    b=np.broadcast_to(1.+mass*.2,(3,len(mass))).copy()
    ext=np.broadcast_to(np.array([.2,1.,5.])[:,None],b.shape)
    tau=optical_depth_from_mass_opacity(mass,ext)
    old_source,old=cancellation_safe_field(tau,b,epsilon*ext,(1-epsilon)*ext)
    source,new=mass_field(tau,b,epsilon*ext,(1-epsilon)*ext,column_mass=mass)
    np.testing.assert_allclose(source,old_source,rtol=3e-13)
    np.testing.assert_allclose(new.interface_flux,old.interface_flux,rtol=3e-12,atol=1e-12)


@pytest.mark.parametrize('epsilon', [1., .1, 1e-6])
def test_variable_extinction_physical_energy_is_flux_divergence(epsilon):
    mass=np.geomspace(1e-8,100.,30)
    wave=np.array([1000.,2500.,9000.])
    ext=(.2+np.exp(2*np.sin(np.arange(30))))[None,:]*np.array([.2,1.,5.])[:,None]
    b=np.broadcast_to(1+mass*.1,ext.shape)
    tau=optical_depth_from_mass_opacity(mass,ext)
    _,field=mass_field(tau,b,epsilon*ext,(1-epsilon)*ext,column_mass=mass)
    energy,_=mass_energy(wave,mass,b,field.mean_intensity,epsilon*ext)
    divergence=np.diff(trapezoid(field.interface_flux,wave,axis=0))
    np.testing.assert_allclose(energy,divergence,rtol=2e-8,atol=2e-8)


@pytest.mark.parametrize('epsilon', [1., .1, 1e-6])
def test_variable_extinction_complete_material_response(epsilon):
    mass=np.geomspace(1e-10,100.,24);wave=np.array([1000.,4000.,20000.])
    ext=(.5+np.exp(np.sin(np.arange(24))))[None,:]*np.array([.3,1.,3.])[:,None]
    base_b=np.broadcast_to(1+mass*.2,ext.shape)
    def evaluate(x):
        b=base_b*np.exp(4*x);a=epsilon*ext*np.exp(1.3*x);s=(1-epsilon)*ext*np.exp(-.2*x)
        tau=optical_depth_from_mass_opacity(mass,a+s)
        source,field=mass_field(tau,b,a,s,column_mass=mass)
        energy,cooling=mass_energy(wave,mass,b,field.mean_intensity,a)
        return (trapezoid(field.interface_flux,wave,axis=0),energy,cooling),(tau,b,a,s,source,field)
    _,(tau,b,a,s,source,field)=evaluate(np.zeros(24))
    fj,ej,cj=mass_energy_response(wave,tau,mass,b,field.mean_intensity,source,a,s,4*b,1.3*a,-.2*s)
    # Physical local cooling must not acquire neighbouring-opacity derivatives.
    offdiag=cj.copy();offdiag[np.arange(23),np.arange(23)]=0
    assert np.count_nonzero(offdiag)==0
    np.testing.assert_allclose(ej,np.diff(fj,axis=0),rtol=3e-7,atol=2e-7)
    for i in (0,7,16,23):
        # Near conservative scattering a 1e-5 perturbation of a thin node
        # subtracts almost identical global fields. A measured step-size
        # study resolves the tangent at 1e-3 (O(h^2) truncation < 5e-6).
        h=1e-3
        dx=np.eye(24)[i]*h
        plus,_=evaluate(dx);minus,_=evaluate(-dx)
        for analytic,p,m in zip((fj,ej,cj),plus,minus):
            fd=(p-m)/(2*h)
            np.testing.assert_allclose(analytic[:,i],fd,rtol=3e-5,atol=2e-5)


def test_isothermal_semi_infinite_solution_converges_with_mass_resolution():
    # Analytic emergent flux is pi B, independent of smooth opacity changes.
    errors=[]
    for n in (40,80,160):
        mass=np.geomspace(1e-8,100.,n)
        absorption=(1+9*mass/(1+mass))[None,:]
        tau=optical_depth_from_mass_opacity(mass,absorption)
        _,field=mass_field(tau,np.ones((1,n)),absorption,0.,column_mass=mass,n_angle=8)
        errors.append(abs(field.interface_flux[0,0]/np.pi-1))
    assert errors[1]<errors[0]/3
    assert errors[2]<errors[1]/3


def test_variable_mass_volume_against_high_precision_matrix():
    mp=pytest.importorskip('mpmath')
    mass=np.geomspace(1e-10,30.,20)
    ext=(1+9*mass/(1+mass))[None,:];b=(1+.2*mass)[None,:]
    a=.01*ext;s=.99*ext;tau=optical_depth_from_mass_opacity(mass,ext)
    _,field=mass_field(tau,b,a,s,column_mass=mass,n_angle=1)
    with mp.workdps(70):
        m=[mp.mpf(float(v)) for v in mass];t=[mp.mpf(float(v)) for v in tau[0]]
        h=[t[0]]+[t[i]-t[i-1] for i in range(1,len(t))]
        mh=[m[0]]+[m[i]-m[i-1] for i in range(1,len(m))]
        n=len(m);matrix=mp.matrix(n+1);rhs=mp.matrix(n+1,1);mu=mp.mpf('.5')
        matrix[0,0]=1+mu/h[0];matrix[0,1]=-mu/h[0]
        for i in range(n-1):
            volume=mp.mpf(float(ext[0,i]))*.5*(mh[i]+mh[i+1])
            left=mu**2/(h[i]*volume);right=mu**2/(h[i+1]*volume)
            matrix[i+1,i]=-left;matrix[i+1,i+1]=left+right+mp.mpf('.01');matrix[i+1,i+2]=-right
            rhs[i+1]=mp.mpf('.01')*mp.mpf(float(b[0,i]))
        matrix[n,n]=1;rhs[n]=float(b[0,-1])
        u=mp.lu_solve(matrix,rhs)
        mean=np.array([float(u[i+1]) for i in range(n)])
        flux=np.array([float(4*mp.pi*mu**2*(u[i+1]-u[i])/h[i]) for i in range(n)])
    np.testing.assert_allclose(field.mean_intensity[0],mean,rtol=3e-13)
    np.testing.assert_allclose(field.interface_flux[0],flux,rtol=3e-12,atol=1e-12)


def test_deep_wien_transfer_preserves_nonnegative_intensity_without_clipping():
    """40-depth PG1159 row that formerly failed at a -5e-323 intensity."""
    from pathlib import Path
    from wd_spectra._mass_feautrier import _mass_emission_field
    with np.load(Path(__file__).parent/'data/pg1159_deep_wien_transfer.npz') as saved:
        data={key:saved[key] for key in saved.files}
    source=data['emission_source'][None,:]
    bottom=np.array([float(data['bottom_source'])])
    options=dict(column_mass=data['column_mass'],n_angle=int(data['n_angle']),
        wavelength_chunk_size=1,require_nonnegative=True,reconstruct_intensity=True)
    _,physical=_mass_emission_field(data['tau'][None,:],source,
        data['fraction'][None,:],data['extinction'][None,:],bottom,**options)
    assert np.all(physical.mean_intensity>=0)
    # Independently use ordinary-magnitude input in the same linear system.
    scale=max(float(source.max()),float(bottom[0]))
    _,normalized=_mass_emission_field(data['tau'][None,:],source/scale,
        data['fraction'][None,:],data['extinction'][None,:],bottom/scale,**options)
    np.testing.assert_allclose(physical.mean_intensity,normalized.mean_intensity*scale,
        rtol=2e-13,atol=20*np.nextafter(0.,1.))
    np.testing.assert_allclose(physical.interface_flux,normalized.interface_flux*scale,
        rtol=2e-13,atol=20*np.nextafter(0.,1.))
