"""Material-exact, fixed-radiation thermal initialization; never convergence.

Research wrapper only. Each selected radiative cell follows its actual heating
or cooling direction to the first bracketed equilibrium, within physical EOS
limits. The updated atmosphere is subsequently solved by the normal coupled
formal-transfer / actual-ML2 equations, with every acceptance gate retained.
"""
import argparse
from contextlib import contextmanager
import json
import sys
from dataclasses import fields, is_dataclass, replace
from unittest.mock import patch
import numpy as np
from wd_spectra import adaptive_structure as adaptive
from wd_spectra._compat import trapezoid
from wd_spectra._stable_feautrier import cancellation_safe_field
from wd_spectra.opacity import optical_depth_from_mass_opacity
from wd_spectra.spectrum import planck_lambda_angstrom
from wd_spectra.constants import STEFAN_BOLTZMANN


def depth_subset(value,indices,n_depth):
    """Slice material fields, including nested LTE states, not their physics."""
    if isinstance(value,np.ndarray) and value.ndim and value.shape[0]==n_depth:
        return value[indices]
    if is_dataclass(value):
        return replace(value,**{f.name:depth_subset(getattr(value,f.name),indices,n_depth) for f in fields(value)})
    return value


class LocalOpacityCache:
    """Fixed-pressure material cache: evaluate only nodes whose T changed."""
    def __init__(self,atmosphere,options,absorption,scattering):
        self.options=options;self.n=atmosphere.n_depth
        self.logt=np.log(atmosphere.temperature)
        self.absorption=absorption.copy();self.scattering=scattering.copy()
        probe=np.unique([0,self.n//2,self.n-1])
        local=depth_subset(atmosphere,probe,self.n)
        # Refuse this optimization if a callback is not depth-local.
        np.testing.assert_allclose(options['true_absorption'](local),absorption[:,probe],rtol=2e-13,atol=0)
        np.testing.assert_allclose(options['scattering_opacity'](local),scattering[:,probe],rtol=2e-13,atol=0)
    def __call__(self,lt):
        changed=np.flatnonzero(lt!=self.logt)
        if len(changed):
            atmosphere=self.options['with_temperature'](np.exp(lt))
            local=depth_subset(atmosphere,changed,self.n)
            self.absorption[:,changed]=self.options['true_absorption'](local)
            self.scattering[:,changed]=self.options['scattering_opacity'](local)
            self.logt=lt.copy()
        return self.absorption,self.scattering


def local_exchange(wave,mass,base_extinction,mean,absorption,scattering,planck):
    """Exact nodal heating/cooling with neighbouring face opacities frozen."""
    ext=absorption+scattering
    width=.25*np.diff(mass)[None,:]*(ext[:,:-1]+base_extinction[:,1:])
    width[:,0]+=.5*mass[0]*ext[:,0]
    width[:,1:]+=.25*np.diff(mass)[:-1][None,:]*(base_extinction[:,:-2]+ext[:,1:-1])
    weight=width*absorption[:,:-1]/ext[:,:-1]
    cooling=trapezoid(weight*planck[:,:-1],wave,axis=0)
    heating=trapezoid(weight*mean[:,:-1],wave,axis=0)
    return (heating-cooling)/cooling


def bracketed_thermal_update(logt,exchange,eligible,minimum,maximum,maximum_change=.12):
    """Follow the thermal sign; no arbitrary hot/cold root selection.

    Return bounded changes and explicit unbracketed cells. Brackets are
    searched in increments smaller than the permitted final log-T change.
    The search does not treat reaching a physical-domain limit as a root.
    """
    initial=exchange(logt)
    direction=np.sign(initial)
    active=eligible & (abs(initial)>1e-8)
    near=logt.copy();far=logt.copy();found=np.zeros(len(logt),bool)
    previous=logt.copy()
    for distance in np.arange(.03,1.801,.03):
        candidate=np.clip(logt+direction*distance,minimum,maximum)
        candidate[~active]=logt[~active]
        candidate[found]=far[found]
        value=exchange(candidate)
        crossed=active & ~found & (value*initial<=0)
        near[crossed]=previous[crossed];far[crossed]=candidate[crossed]
        found|=crossed
        previous[~found]=candidate[~found]
        if np.all(found | ~active): break
        if int(round(distance/.03))%8==0:
            print(f'Thermal bracket search: |dlnT|={distance:.3g}, '
                  f'{np.sum(found)}/{np.sum(active)} roots bracketed',flush=True)
    # Once a bracket lies beyond the permitted change, the returned bounded
    # step is already known exactly. Do not keep recomputing those nodes'
    # opacities while refining other roots inside the step bound.
    saturated=found & (np.minimum(abs(near-logt),abs(far-logt))>=maximum_change)
    near[saturated]=far[saturated]=logt[saturated]+direction[saturated]*maximum_change
    for iteration in range(32):
        if np.max(abs(far-near))<1e-8:break
        midpoint=.5*(near+far)
        value=exchange(midpoint)
        same=value*initial>0
        near[found & same]=midpoint[found & same]
        far[found & ~same]=midpoint[found & ~same]
        if iteration%8==7:
            print(f'Thermal root refinement {iteration+1}: '
                  f'{np.sum(abs(far-near)>=1e-8)} active cells',flush=True)
    change=np.zeros_like(logt)
    change[found]=np.clip(.5*(near+far)[found]-logt[found],-maximum_change,maximum_change)
    return logt+change,found,active & ~found


@contextmanager
def thermal_initializer(sweeps,minimum,maximum,*,project_convection=False,allow_energy_barrier=False,
                        preserve_convective_transport=False):
    if allow_energy_barrier and not project_convection:
        raise ValueError('thermal branch escape requires actual convective trial correction')
    if preserve_convective_transport and not (project_convection and allow_energy_barrier):
        raise ValueError('transport-preserving branch initialization requires explicit branch escape')
    original_adaptive=adaptive.solve_adaptive_lte_structure
    original_newton=adaptive.solve_trust_region_newton
    context={}
    def capture(seed,wavelength,**options):
        options['metadata']={**options.get('metadata',{}),
            'experimental_exact_thermal_initializer_maximum_sweeps':sweeps,
            'experimental_exact_thermal_convective_projection':project_convection,
            'experimental_thermal_branch_escape_allows_energy_barrier':allow_energy_barrier,
            'experimental_thermal_branch_preserves_convective_transport':preserve_convective_transport,
            'experimental_exact_thermal_initializer_is_convergence':False}
        context.update(wave=wavelength,options=options,used=False)
        return original_adaptive(seed,wavelength,**options)
    def solve(initial,evaluate,**settings):
        first=evaluate(initial,False)
        if context['used'] or not first.payload['energy_balance_is_physical_flux']:
            return original_newton(initial,evaluate,**settings)
        context['used']=True
        options=context['options'];wave=context['wave']
        # State mapping is exposed with a differentiated evaluation.
        ev=evaluate(initial,True)
        mapping=ev.payload['log_temperature_from_state']
        state=initial.copy()
        for iteration in range(sweeps):
            atmosphere=ev.payload['atmosphere'];logt=np.log(atmosphere.temperature)
            a=options['true_absorption'](atmosphere)
            s=options['scattering_opacity'](atmosphere)
            ext=a+s;mass=atmosphere.column_mass
            planck=planck_lambda_angstrom(wave[:,None],atmosphere.temperature[None,:])
            tau=optical_depth_from_mass_opacity(mass,ext)
            _,field=cancellation_safe_field(tau,planck,a,s,n_angle=options['n_angle'])
            conv=ev.payload['convective_flux_interface']
            eligible=np.r_[(conv[:-1]==0)&(conv[1:]==0),False]
            cache=LocalOpacityCache(atmosphere,options,a,s)
            def exchange(lt):
                aa,ss=cache(lt)
                bb=planck_lambda_angstrom(wave[:,None],np.exp(lt)[None,:])
                return np.r_[local_exchange(wave,mass,ext,field.mean_intensity,aa,ss,bb),0.]
            updated,found,unbracketed=bracketed_thermal_update(
                logt,exchange,eligible,np.log(minimum),np.log(maximum))
            projection={};factor=1.;accepted=True
            if project_convection:
                from . import check_cool_db_transport_seed as runner
                from .exact_convective_trial_projection import project_trial
                from .dense_helium_limits import DenseHeliumDomainError
                merit=float(np.mean(ev.residual**2))
                accepted=False
                # Interpolate the raw thermal change FIRST, then solve the
                # finite-material convective constraint anew at each trial.
                # Scaling a projected temperature change afterward destroys
                # the constraint in efficient convection.
                while factor>=2.**-8:
                    try:
                        candidate=options['with_temperature'](np.exp(logt+factor*(updated-logt)))
                        if preserve_convective_transport:
                            from .convective_mesh_prolongation import transport_profile
                            target=STEFAN_BOLTZMANN*atmosphere.effective_temperature**4
                            candidate,error=transport_profile(candidate,conv/target,runner,options,project_stable=True)
                            projection=dict(preserved_previous_convective_fraction=True,
                                maximum_actual_convective_fraction_error=error)
                        else:
                            candidate,projection=project_trial(candidate,runner,options)
                        candidate_state=np.linalg.solve(mapping,np.log(candidate.temperature))
                        candidate_ev=evaluate(candidate_state,False)
                    except DenseHeliumDomainError:
                        factor*=.5
                        continue
                    bounded=(np.max(abs(np.log(candidate.temperature)-logt))<=.12*(1+1e-12))
                    if ((allow_energy_barrier and bounded)
                            or (not allow_energy_barrier and float(np.mean(candidate_ev.residual**2))<merit)):
                        state,ev=candidate_state,candidate_ev
                        updated=np.log(candidate.temperature)
                        accepted=True
                        break
                    factor*=.5
                if not accepted:updated=logt.copy()
            else:
                state=np.linalg.solve(mapping,updated)
                ev=evaluate(state,False)
            target=STEFAN_BOLTZMANN*atmosphere.effective_temperature**4
            print('THERMAL INITIALIZER (not converged): '+json.dumps(dict(
                sweep=iteration+1,bracketed_cells=int(sum(found)),unbracketed_cells=int(sum(unbracketed)),
                accepted=accepted,line_search_factor=factor,convective_projection=projection,
                requires_static_energy_merit_reduction=project_convection and not allow_energy_barrier,
                maximum_log_temperature_change=float(np.max(abs(updated-logt))),
                surface_temperature=float(np.exp(updated[0])),
                maximum_local_energy=float(np.max(abs(ev.payload['cell_energy_balance_relative_residual']))),
                maximum_stellar_flux_defect=float(np.max(abs(ev.payload['total_flux_interface']/target-1))))),flush=True)
            if not accepted or np.max(abs(updated-logt))<1e-5: break
        settings['allow_initial_convergence']=False
        return original_newton(state,evaluate,**settings)
    with patch.object(adaptive,'solve_adaptive_lte_structure',capture), \
            patch.object(adaptive,'solve_trust_region_newton',solve):
        yield


def main():
    parser=argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--thermal-sweeps',type=int,default=8)
    args,remaining=parser.parse_known_args()
    if args.thermal_sweeps<1: parser.error('thermal sweeps must be positive')
    from .dense_helium_fluid_experiment import KB_EV
    from . import dense_helium_molecular_experiment as experiment
    sys.argv=[sys.argv[0]]+remaining
    with thermal_initializer(args.thermal_sweeps,np.nextafter(.1/KB_EV,np.inf),17000):
        experiment.main()


if __name__=='__main__':main()
