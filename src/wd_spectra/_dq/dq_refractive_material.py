from pathlib import Path
"""DQ-only research adapter: refractive radiation, EOS and grey ML2 losses.

This process-local adapter temporarily replaces the EXISTING mass-transfer
callbacks, then restores them on success/error. No production source file is
modified. Use one atmosphere per process, never concurrent threads.

Radiation uses frequency-dependent published refractivity. The published
Rayleigh cross section is already a physical transport cross section:
convert it to the reduced-transfer input sigma0=n*sigma_Ray (Iglesias et al.
2002, Eq. 4.6). Free-free screening does NOT include its separate 1/n
propagation factor. Rosseland transport uses (kappa0+sigma0)/n^3.
ML2 remains a grey closure: its element optical depth
uses a static-index approximation, while preserving the frequency-integrated
diffusion limit and the n*chi0 optically thin limit in the grey case. This
approximation is explicit, not a claim to match private Blouin convection.
"""
from collections import OrderedDict
from dataclasses import replace
from functools import lru_cache
import time
import resource
import sys

import numpy as np

from wd_spectra._compat import trapezoid
from . import base as dq
from wd_spectra.convection import _ml2_local_coefficients_from_thermodynamics
from wd_spectra._rosseland import rosseland_mean_from_opacity_grid
from wd_spectra.constants import PLANCK, LIGHT_SPEED
from wd_spectra.dense_helium_continuum import dielectric_from_refractivity,HARTREE_ERG
from wd_spectra.radiative_transfer import RadiationField
from wd_spectra.spectrum import Spectrum,planck_lambda_angstrom
from .dq_dense_continuum_inputs import atomic_polarizability
from .dq_pair_virial_quadrature import pair_virial_integrals
from .dq_hornkohl_bounded_energy import BoundedDirectEnergyDQMaterial
from .dq_hornkohl_cell import CellIntegratedDQMaterial
from .dq_hornkohl_gated_energy import gated_solve
from .dq_refractive_finite_volume import solve as fv_solve, formal_field
from .dq_refractive_fv_response import integrated_response


@lru_cache(maxsize=4096)
def pair_coefficients(temperature):
    return pair_virial_integrals(float(temperature),short_range='constant')


def refractive_index(atmosphere,wavelength):
    w=np.asarray(wavelength,float)
    energy=PLANCK*LIGHT_SPEED/(w*1e-8*HARTREE_ERG)
    alpha=atomic_polarizability(energy)
    pairs=np.array([pair_coefficients(float(t)) for t in atmosphere.temperature])
    return np.sqrt(dielectric_from_refractivity(
        atmosphere.helium_lte_state.neutral_he_density[None,:],alpha[:,None],
        pairs[None,:,0]+energy[:,None]**2*pairs[None,:,1]))


def static_index(atmosphere):
    pair=np.array([pair_coefficients(float(t))[0] for t in atmosphere.temperature])
    return np.sqrt(dielectric_from_refractivity(atmosphere.helium_lte_state.neutral_he_density,
        atomic_polarizability(0.),pair))


def grey_ml2_loss_factor(tau_diffusion,index):
    """Correct ML2's grey thin-cell bridge with exact grey refractive limits.

    Let tau_d=chi0*l*rho/n^3. Physical element tau=chi0*l*rho/n=n^2*tau_d.
    The reduced Planck intensity corresponds to physical n^2*B. Thus the
    multiplier is n^2*G(n^2*tau_d)/G(tau_d), G(tau)=8*tau/(1+tau^2/2).
    Evaluate its algebraic ratio without overflow at large optical depth.
    """
    tau=np.asarray(tau_diffusion,float);n=np.asarray(index,float)
    if np.any(tau<0) or np.any(n<1) or np.any(~np.isfinite(tau+n)):
        raise ValueError('Finite positive optical depth and n>=1 required')
    inverse=1/(1+.5*np.minimum(tau,1e150)**2)
    return n**4/(inverse+(1-inverse)*n**4)


class RefractiveBackend:
    # A transfer response is consumed synchronously before the next physical
    # evaluation begins.  Retaining older full radiation fields cannot serve a
    # later Jacobian: their opacity, source and optical-depth identities differ.
    # Keep only the current field.  Four opacity registrations cover its base,
    # hot and cold constitutive states plus one transient caller registration.
    # The previous default of six *full* fields retained hundreds of MiB and
    # made a same-process lower-domain extension cross the explicit memory cap.
    transfer_context_limit=1
    opacity_registration_limit=4

    def __init__(self,material,wave):
        self.material=material;self.wave=np.asarray(wave);self.records=OrderedDict()
        self.contexts=OrderedDict();self.means=OrderedDict();self.probes=None
        self.rosseland_memo=None
        self.absorption_memo=None
        self.last_progress=time.monotonic();self.calls=0;self.maximum_conservation_error=0.

    def check_memory(self):
        check=getattr(self.material,'check_budget',None)
        if check is not None:check()
        peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform!='darwin':peak*=1024
        limit=getattr(self.material,'maximum_memory_bytes',4*1024**3)
        if peak>limit:
            raise MemoryError(f'DQ research memory ceiling exceeded: {peak/1024**3:.2f} GiB > {limit/1024**3:.2f} GiB; no physics fallback')
        return peak

    def release(self):
        """Drop transfer-owned arrays after a domain solve has consumed them.

        Boundary screening is evaluated inside the adaptive solve, so none of
        these identity-bound buffers can be reused by the next, deeper domain.
        Clear them explicitly instead of relying on cyclic garbage collection
        to run before that domain allocates its new opacity and index arrays.
        """
        self.records.clear()
        self.contexts.clear()
        self.means.clear()
        self.probes=None
        self.rosseland_memo=None
        self.absorption_memo=None

    @staticmethod
    def keep(cache,key,value,limit=6):
        cache[key]=value;cache.move_to_end(key)
        while len(cache)>limit:cache.popitem(last=False)

    def index(self,current):
        state=self.material.chemistry(current)[0]
        index=(refractive_index(state,self.wave) if self.material.refraction_enabled
            else np.ones((len(self.wave),state.n_depth)))
        return state,index

    def absorption(self,current,original):
        value=original(current)
        if not isinstance(value,np.ndarray) or value.dtype!=np.float64 or value.ndim!=2:
            raise TypeError('Refractive absorption must be a float64 matrix for identity registration')
        state,index=self.index(current)
        self.keep(self.records,id(value),(value,state,index),
            limit=self.opacity_registration_limit)
        self.absorption_memo=(current,original,value,index)
        return value

    def context(self,tau):
        ctx=self.contexts[id(tau)]
        if ctx['tau'] is not tau:raise RuntimeError('Stale refractive context')
        return ctx

    def field(self,tau,planck,absorption,scattering,*,column_mass,n_angle,**unused):
        self.check_memory()
        entry=self.records.get(id(absorption))
        primary=entry is not None and entry[0] is absorption
        if primary:
            ctx=dict(tau=tau,mass=column_mass,a=absorption,s=scattering,b=planck,n=entry[2],state=entry[1])
            self.keep(self.contexts,id(tau),ctx,
                limit=self.transfer_context_limit)
        else:
            ctx=self.context(tau)
            self.require_same('formal column mass',column_mass,ctx['mass'])
            if np.any(scattering!=0) or not np.array_equal(absorption,ctx['a']+ctx['s']):
                raise RuntimeError('Unregistered refractive opacity; no straight-ray fallback')
        result=(fv_solve(column_mass,absorption,scattering,planck,ctx['n'],n_angle=self.material.ray_angles)
            if primary else formal_field(column_mass,absorption,planck,ctx['n'],n_angle=self.material.ray_angles))
        mean=result['reduced_mean_intensity']
        source=(absorption*planck+scattering*mean)/(absorption+scattering)
        source[:,-1]=planck[:,-1]
        # Here mean_intensity is explicitly REDUCED J, matching the reduced
        # source equation; it is never exposed as a physical public J field.
        field=RadiationField(mean,result['flux'],result['flux'])
        if primary:
            ctx['result']=result
            ctx['source']=source
            self.keep(self.means,id(mean),(mean,ctx),
                limit=self.transfer_context_limit)
            self.maximum_conservation_error=max(self.maximum_conservation_error,result['maximum_conservation_error'])
        self.calls+=1
        print(f'DQ refractive field {self.calls}: {len(planck)} wavelengths, '
            f'{len(column_mass)} layers, conservation={result["maximum_conservation_error"]:.3g}, '
            f'peak_RSS={self.check_memory()/1024**3:.2f} GiB',flush=True)
        return source,field

    @staticmethod
    def require_same(name,actual,expected):
        if not np.array_equal(actual,expected):
            raise RuntimeError(f'Refractive callback {name} does not match its field; no fallback')

    def require_grid(self,ctx,wave,mass):
        self.require_same('wavelength',wave,self.wave)
        self.require_same('column mass',mass,ctx['mass'])

    def record_temperature_response_probes(self,probes):
        self.probes=probes

    def derivatives(self,ctx):
        if self.probes is None:raise RuntimeError('Missing constitutive probes')
        hot,cold,h=self.probes
        ha,hs=hot[1:];hn=self.records[id(ha)][2]
        np.testing.assert_allclose(hot[0].temperature,ctx['state'].temperature*np.exp(h),rtol=2e-13)
        if cold is None:
            return (ha-ctx['a'])/h,(hs-ctx['s'])/h,(hn-ctx['n'])/h
        ca,cs=cold[1:];cn=self.records[id(ca)][2]
        np.testing.assert_allclose(cold[0].temperature,ctx['state'].temperature*np.exp(-h),rtol=2e-13)
        return (ha-ca)/(2*h),(hs-cs)/(2*h),(hn-cn)/(2*h)

    def response_progress(self,done,total):
        self.check_memory()
        if time.monotonic()-self.last_progress>25:
            print(f'DQ refractive analytic response: {done}/{total} wavelengths',flush=True)
            self.last_progress=time.monotonic()

    def response(self,tau,wave,source,direct,db,fraction,mass,dk,*,extinction,**unused):
        ctx=self.context(tau);da,ds,dn=self.derivatives(ctx)
        self.require_grid(ctx,wave,mass)
        self.require_same('source',source,ctx['source'])
        self.require_same('extinction',extinction,ctx['a']+ctx['s'])
        np.testing.assert_allclose(dk,da+ds,rtol=2e-12,atol=1e-15)
        np.testing.assert_allclose(fraction,ctx['s']/extinction,rtol=2e-12,atol=1e-15)
        expected=(da*ctx['b']+ctx['a']*db+ds*ctx['result']['reduced_mean_intensity']
            -(da+ds)*source)/extinction
        np.testing.assert_allclose(direct,expected,rtol=2e-12,atol=1e-15)
        if unused.get('return_auxiliary_response',False) or unused.get('mean_response_consumer') is not None:
            raise NotImplementedError('Refractive adapter supplies integrated responses only')
        jac=integrated_response(mass,wave,ctx['a'],ctx['s'],ctx['b'],ctx['n'],
            da,ds,db,dn,n_angle=self.material.ray_angles,progress=self.response_progress)
        return jac[0],None,None

    def energy_response(self,wave,tau,mass,planck,mean,source,a,s,db,da,ds,**unused):
        ctx=self.context(tau);probe_da,probe_ds,dn=self.derivatives(ctx)
        self.require_grid(ctx,wave,mass)
        self.require_same('source',source,ctx['source'])
        for name,value in (('a',a),('s',s),('b',planck)):
            self.require_same(name,value,ctx[name])
        self.require_same('mean intensity',mean,ctx['result']['reduced_mean_intensity'])
        np.testing.assert_allclose(da,probe_da,rtol=2e-12,atol=1e-15)
        np.testing.assert_allclose(ds,probe_ds,rtol=2e-12,atol=1e-15)
        return integrated_response(mass,wave,a,s,planck,ctx['n'],da,ds,db,dn,
            n_angle=self.material.ray_angles,progress=self.response_progress)

    def energy(self,wave,mass,planck,mean,absorption):
        stored,ctx=self.means[id(mean)]
        if stored is not mean:raise RuntimeError('Stale refractive energy context')
        self.require_grid(ctx,wave,mass)
        self.require_same('absorption',absorption,ctx['a'])
        self.require_same('Planck intensity',planck,ctx['b'])
        return tuple(trapezoid(ctx['result'][key],wave,axis=0) for key in
            ('cell_heating','cell_thermal_emission'))

    def boundary(self,wave,mass,absorption,bottom_planck,target):
        # Measured response of the DISCRETE coupled field to its bottom B,
        # not an exponential bound that can miss coarse-cell leakage.
        matches=[ctx for ctx in self.contexts.values() if ctx['a'] is absorption]
        if not matches:raise RuntimeError('Missing refractive bottom response')
        ctx=matches[-1]
        self.require_grid(ctx,wave,mass)
        self.require_same('bottom Planck intensity',bottom_planck,ctx['b'][:,-1])
        if not np.isfinite(target) or target<=0:raise ValueError('Positive finite target flux required')
        value=trapezoid(ctx['result']['boundary_surface_flux'],wave)/target
        return float(value)

    def rosseland(self,current,original_absorption):
        # Match the public DQ identity memo: atmospheres are replaced, not
        # mutated, during a solve. Keep a strong reference, bounded to ONE
        # state. A distinct probe/atmosphere must always be recomputed.
        memo=self.rosseland_memo
        if memo is not None and memo[0] is current and memo[1] is original_absorption:
            return memo[2]
        entry=self.absorption_memo
        if entry is None or entry[0] is not current or entry[1] is not original_absorption:
            value=original_absorption(current)
            state,index=self.index(current)
        else:
            _,_,value,index=entry
        result=rosseland_mean_from_opacity_grid(self.wave,
            (value+self.material.scattering(current,self.wave))/index**3,current.temperature)
        result.flags.writeable=False
        self.rosseland_memo=(current,original_absorption,result)
        return result


class RefractiveDQMaterial(BoundedDirectEnergyDQMaterial):
    refraction_enabled=True
    ray_angles=4
    declared_eos_seed=False
    maximum_memory_bytes=4*1024**3

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.interface_index_cache=OrderedDict()
        from .provenance import digest
        self.experiment_metadata=dict(refraction_enabled=self.refraction_enabled,
            declared_eos_seed=self.declared_eos_seed,
            ray_angles_per_invariant_interval=self.ray_angles,
            angular_coordinate='quadratic high-end turning-point map',
            maximum_process_memory_bytes=self.maximum_memory_bytes,
            retained_transfer_contexts=RefractiveBackend.transfer_context_limit,
            retained_opacity_registrations=RefractiveBackend.opacity_registration_limit,
            atmosphere_and_synthesis_same_transfer=True,
            bottom_boundary='incident reduced LTE intensity; exact finite turning buffer',
            ml2_approximation='grey static-index thin bridge; frequency-dependent Rosseland n^3',
            scattering_convention='sigma0=Thomson0+n*Rayleigh_transport; Iglesias2002 Eq4.6',
            source_sha256={name:digest(Path(__file__).with_name(name)) for name in
                ('dq_refractive_material.py','dq_refractive_finite_volume.py',
                 'dq_refractive_fv_response.py','dq_refractive_ray_bundles.py',
                 'dqsolution_sampling.py','dq_hornkohl_cell.py','dq_exact_opacity_cache.py',
                 'dq_dense_continuum_inputs.py','dq_pair_virial_quadrature.py')})

    def scattering(self,current,wavelength):
        state=self.chemistry(current)[0]
        electron=dq.electron_scattering_mass_coefficient(state)[None,:]
        rayleigh=dq.helium_rayleigh_scattering_mass_coefficient(state,wavelength)
        if not self.refraction_enabled:return electron+rayleigh
        index=refractive_index(state,wavelength)
        # The generic ray kernel divides its input extinction by n. The
        # already-transport-form Rayleigh cross section must NOT be divided
        # again. Thomson is retained in the non-dispersive input convention.
        return electron+index*rayleigh

    def hydrostatic_seed(self,teff,logg,n_depth):
        if not self.declared_eos_seed:
            return super().hydrostatic_seed(teff,logg,n_depth)
        if self.helium_reos3 is None:
            raise ValueError('Declared REOS seed requires the configured table')
        print('DQ cold seed: using the declared REOS host in hydrostatic integration',flush=True)
        return dq.helium_continuum_atmosphere(teff,logg,n_depth=n_depth,
            correlated_microfields=True,metal_database=self.atomic,
            metal_abundances={'C':self.config.log_carbon_to_helium},
            include_dense_helium_metal_ionization=self.config.nonideal_carbon_ionization,
            helium_reos3_table=self.helium_reos3)

    def solve(self,*args,**kwargs):
        # The refractive path performs its own explicit flux-to-energy handoff
        # below. Skip the older research subclass whose implementation
        # replaces the module-level solver function.
        return CellIntegratedDQMaterial.solve(self,*args,**kwargs)

    def solve_adaptive_structure(self,seed,wave,**options):
        backend=RefractiveBackend(self,wave)
        absorption=options['true_absorption']
        options=dict(options,transfer_discretization='column-mass',
            column_mass_radiation_backend=backend,use_precision_polish=False,
            true_absorption=lambda a:backend.absorption(a,absorption),
            rosseland_opacity=lambda a:backend.rosseland(a,absorption))
        options['metadata']={**options.get('metadata',{}),
            'dq_refractive_transfer':self.refraction_enabled,
            'dq_refractive_ray_angles':self.ray_angles,
            'dq_transfer_discretization':'conservative-invariant-ray-mass-cells',
            'dq_ml2_refraction':'frequency-integrated diffusion; static-index grey thin-cell bridge',
            'dq_boundary_screening':'direct discrete bottom response'}

        def coefficients(atmosphere,opacity,cp,expansion,adiabatic,alpha):
            result=_ml2_local_coefficients_from_thermodynamics(
                atmosphere,opacity,cp,expansion,adiabatic,alpha)
            if not self.refraction_enabled:return result
            # Density/thermal derivatives remain the caller's explicit
            # interface averages. Refractivity uses the same coupled EOS at
            # interface P,T, and is re-evaluated for tangent probes.
            saved=self.cached
            try:
                interface=atmosphere
                key=(interface.temperature.tobytes(),interface.gas_pressure.tobytes())
                cached_index=self.interface_index_cache.get(key)
                if cached_index is not None:
                    index=cached_index
                else:
                    from wd_spectra.carbon_molecular import _reos_host_state
                    if self.helium_reos3 is None:
                        raise ValueError('This refractive experiment requires the declared REOS host')
                    if interface.helium_lte_state is None:
                        interface=replace(interface,helium_lte_state=_reos_host_state(
                            interface.temperature,interface.gas_pressure,self.helium_reos3))
                    state=self.chemistry(interface)[0]
                    index=static_index(state);index.flags.writeable=False
                    RefractiveBackend.keep(self.interface_index_cache,key,index,limit=64)
            finally:
                self.cached=saved
            tau=np.asarray(opacity)*alpha*atmosphere.gas_pressure/atmosphere.gravity
            return result[0],result[1]*grey_ml2_loss_factor(tau,index),result[2]

        options['ml2_coefficient_function']=coefficients
        try:
            result=gated_solve(super().solve_adaptive_structure,seed,wave,
                finish_in_energy_equations=getattr(self,'finish_conditioning_in_energy_equations',False),
                **options)
        finally:
            maximum_conservation_error=backend.maximum_conservation_error
            backend.release()
        return replace(result,metadata={**result.metadata,
            'dq_maximum_discrete_radiative_conservation_error':maximum_conservation_error})

    def spectrum(self,atmosphere,wavelength,n_angle,*,include_c2=True):
        state=self.chemistry(atmosphere)[0];wave=np.asarray(wavelength)
        flux=np.empty_like(wave);closure=0.;conservation=0.
        for first in range(0,len(wave),1000):
            w=wave[first:first+1000]
            a=self.absorption(state,w,include_c2=include_c2);s=self.scattering(state,w)
            b=planck_lambda_angstrom(w[:,None],state.temperature[None,:])
            index=refractive_index(state,w) if self.refraction_enabled else np.ones_like(a)
            result=fv_solve(state.column_mass,a,s,b,index,n_angle=self.ray_angles)
            flux[first:first+len(w)]=result['flux'][:,0]
            closure=max(closure,result['maximum_source_error'])
            conservation=max(conservation,result['maximum_conservation_error'])
            print(f'DQ refractive synthesis: {first+len(w)}/{len(wave)} wavelengths',flush=True)
        if np.any(~np.isfinite(flux)) or np.any(flux<=0):raise RuntimeError('Invalid refractive emergent spectrum')
        return Spectrum(wave,flux,dict(refraction=self.refraction_enabled,
            transfer_discretization='conservative-invariant-ray-mass-cells',opacity_scale=1.,
            independent_radiation_scaled_source_error=closure,maximum_cell_conservation_error=conservation))
