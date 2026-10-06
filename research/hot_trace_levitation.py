"""Experimental radiative-force diagnostics on the fixed hot-DA atmosphere.

These are photon momentum deposition rates, not diffusion velocities or a
levitation equilibrium. Bound-free momentum is assigned entirely to the parent
element (no photoelectron redistribution); BB and BF are reported separately.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.integrate import trapezoid

from wd_spectra.constants import BOLTZMANN, LIGHT_SPEED
from wd_spectra.hot_nlte import transfer_field
from wd_spectra.hot_trace_metals import _atom_selection, fixed_electron_metal_reference
from wd_spectra.light_metal_nlte import (hot_metal_line_nlte_coefficients,
    light_metal_bound_free_nlte_coefficients, reduced_light_metal_wavelength)
from wd_spectra.metals import (ATOMIC_MASS_U, read_pg1159_atomic_database,
    read_verner_photoionization_database)
from wd_spectra.models.common import ModelData, numerical_resolution
from wd_spectra.models.hot import _model_from_config, structure_wavelength
from wd_spectra.nlte_core import NLTETransferCoefficients
from resynthesize_hot_trace import load_frozen_host, identity


def radiative_acceleration(wave, opacity, flux, mass_fraction):
    """Integral kappa_Z F_lambda d lambda / (c X_Z), in cm/s^2.

    kappa_Z is per total atmospheric mass (cm^2/g), F_lambda is the local
    outward 4*pi*H_lambda in erg/s/cm^2/Angstrom. Use the centered flux with
    centered opacities, not staggered interface flux. Signed flux/opacity are
    retained, including inward flux and stimulated emission. No extra 4*pi or
    Angstrom-to-cm factor belongs in this expression.
    """
    w, k, f, x = (np.asarray(a, dtype=float) for a in (wave, opacity, flux, mass_fraction))
    if (w.ndim != 1 or len(w) < 2 or np.any(w <= 0) or np.any(np.diff(w) <= 0)
            or k.ndim != 2 or k.shape != f.shape or k.shape[0] != len(w)
            or x.shape != (k.shape[1],) or np.any(x <= 0)
            or any(np.any(~np.isfinite(a)) for a in (w, k, f, x))):
        raise ValueError('invalid force grid, coefficients, flux, or mass fractions')
    return trapezoid(k*f, w, axis=0)/(LIGHT_SPEED*x)


def zero_drift_abundance_gradient(atmosphere, atomic_mass, mean_charge, acceleration):
    """Required d ln[n(Z)/n(H)] / d ln m in a restricted zero-drift model.

    With z outward and dm/dz=-rho, ideal electron force balance gives
    e E = rho/ne * d(ne*kT)/dm. Subtracting the H-nuclei pressure gradient
    from trace-element force balance gives the expression below. Electron
    inertia/radiative force, thermal diffusion, ion-specific mobilities and
    charge-exchange momentum transfer are omitted. This is a diagnostic of
    the current state, NOT an integrated abundance profile: acceleration,
    ionization and radiation must be recomputed as composition changes.
    In a fully ionized isothermal pure-H atmosphere it reduces to
    2*A*(1-g_rad/g) - (Z+1), apart from the m_u/m_H convention.
    """
    m = np.asarray(atmosphere.column_mass)
    temperature = np.asarray(atmosphere.temperature)
    rho = np.asarray(atmosphere.mass_density)
    ne = np.asarray(atmosphere.electron_density)
    nh = np.asarray(atmosphere.hydrogen_lte_state.hydrogen_nuclei_density)
    q, grad = np.asarray(mean_charge), np.asarray(acceleration)
    if (m.ndim != 1 or len(m)<3 or np.any(np.diff(m)<=0)
            or any(x.shape != m.shape for x in (temperature,rho,ne,nh,q,grad))
            or any(np.any(~np.isfinite(x)) for x in (m,temperature,rho,ne,nh,q,grad))
            or any(np.any(x<=0) for x in (m,temperature,rho,ne,nh))
            or np.any(q<0) or not np.isfinite(atomic_mass) or atomic_mass<=0):
        raise ValueError('invalid atmosphere, charge, or acceleration for diffusion diagnostic')
    electron_gradient = np.gradient(np.log(ne*temperature),np.log(m),edge_order=2)
    hydrogen_gradient = np.gradient(np.log(nh*temperature),np.log(m),edge_order=2)
    return (atomic_mass*1.66053906660e-24*m*(atmosphere.gravity-grad)
            /(rho*BOLTZMANN*temperature) - q*electron_gradient - hydrogen_gradient)


def mean_element_charge(reference, element, level_key, population, lte_population):
    """Combine explicit NLTE levels with the solver's inert LTE reservoir."""
    total = reference.element_number_density[element]
    charge_sum = np.arange(reference.ion_number_density[element].shape[0])@reference.ion_number_density[element]
    population,lte_population = np.asarray(population),np.asarray(lte_population)
    if (population.shape != lte_population.shape or population.shape != (len(level_key),len(total))
            or any(np.any(~np.isfinite(x)) or np.any(x<0) for x in (population,lte_population))
            or any(key[0]!=element for key in level_key)):
        raise ValueError('inconsistent explicit element populations')
    for key,n,n_lte in zip(level_key,population,lte_population):
        charge_sum += int(key[1])*(n-n_lte)
    return charge_sum/total


def homogeneous_support_acceleration(atmosphere, atomic_mass, mean_charge):
    """Acceleration giving zero abundance slope in the restricted diagnostic."""
    slope=zero_drift_abundance_gradient(atmosphere,atomic_mass,mean_charge,
                                       np.zeros_like(atmosphere.column_mass))
    coefficient=atomic_mass*1.66053906660e-24*atmosphere.column_mass/(
        atmosphere.mass_density*BOLTZMANN*atmosphere.temperature)
    return slope/coefficient


def radiation_grid(host, database, counts, refinement=1):
    """Include host continua plus all selected metal lines and edges."""
    from wd_spectra._hot_structure import HotEquations
    model = _model_from_config(host.config, ModelData.default())
    grid = HotEquations(host.atmosphere, model, structure_wavelength(
        host.config, numerical_resolution(host.config.quality).n_continuum)).wave
    grid = np.unique(np.concatenate([grid]+[reduced_light_metal_wavelength(
        database,e,count,n_continuum_wavelength=640) for e,count in counts.items()]))
    for _ in range(refinement):
        grid = np.unique(np.r_[grid,np.sqrt(grid[:-1]*grid[1:])])
    return grid


def element_coefficients(atmosphere, wave, database, photo, reference, element, counts, departures, *, charge=None):
    unity = {(element, ion.charge): np.ones(atmosphere.n_depth)
             for ion in database.ion_stages(element)}
    lines, bound_counts = _atom_selection(database, element, counts)
    if charge is not None:
        lines=frozenset(key for key in lines if key[1]==charge)
        bound_counts={q:(n if q==charge else 0) for q,n in bound_counts.items()}
    a, j = hot_metal_line_nlte_coefficients(
        atmosphere,wave,database,reference,unity,elements=(element,),
        level_departure_coefficient=departures,transition_keys=lines,
        minimum_oscillator_strength=0.,maximum_lines=None,
        include_static_linear_stark=False,retain_inverted_emissivity=True)
    b, q = light_metal_bound_free_nlte_coefficients(
        atmosphere,wave,database,reference,photo,unity,elements=(element,),
        level_departure_coefficient=departures,levels_per_charge=bound_counts,
        include_explicit_kramers=True)
    return a, b, j+q


def evaluate_forces(host, abundances, departures, counts, wave, *, database=None, photo=None, ion_breakdown=False):
    data = ModelData.default()
    database = database or read_pg1159_atomic_database(data.stout,elements=tuple(abundances))
    photo = photo or read_verner_photoionization_database(
        data.verner_photoionization,elements=tuple(abundances),require_all_elements=True)
    atmosphere = host.atmosphere
    reference = fixed_electron_metal_reference(atmosphere,database,abundances)
    background = _model_from_config(host.config,data).transfer_coefficients(
        atmosphere,wave,host.population_state)
    absorption = background.true_absorption.copy()
    emission = background.thermal_emissivity.copy()
    components = {}
    for e in abundances:
        a,b,j = element_coefficients(atmosphere,wave,database,photo,reference,e,counts[e],departures[e])
        components[e] = (a,b)
        absorption += a+b
        emission += j
    combined = NLTETransferCoefficients(wave,absorption,emission,background.scattering,{})
    _,field,closure = transfer_field(atmosphere,combined,n_angle=3)
    output = {'column_mass':atmosphere.column_mass,'temperature':atmosphere.temperature,
              'gravity':np.array(atmosphere.gravity),'wave':wave,
              'surface_flux':field.interface_flux[:,0],
              'total_acceleration':radiative_acceleration(wave,combined.total_extinction,
                                                         field.flux,np.ones(atmosphere.n_depth))}
    for e,(a,b) in components.items():
        x = reference.element_number_density[e]*ATOMIC_MASS_U[e]*1.66053906660e-24/atmosphere.mass_density
        output[e+'_mass_fraction'] = x
        output[e+'_bb'] = radiative_acceleration(wave,a,field.flux,x)
        output[e+'_bf'] = radiative_acceleration(wave,b,field.flux,x)
        output[e+'_total'] = output[e+'_bb']+output[e+'_bf']
        # Full bolometric force is essential; COS/STIS coverage misses EUV lines.
        uv = (wave >= 1150.) & (wave <= 1700.)
        output[e+'_hst_uv'] = radiative_acceleration(wave[uv],(a+b)[uv],field.flux[uv],x)
        if ion_breakdown:
            ion_sum=np.zeros(atmosphere.n_depth)
            for charge in counts[e]:
                bb,bf,_ = element_coefficients(atmosphere,wave,database,photo,reference,
                    e,counts[e],departures[e],charge=charge)
                force=radiative_acceleration(wave,bb+bf,field.flux,x)
                output[f'{e}_charge_{charge}']=force
                ion_sum+=force
            np.testing.assert_allclose(ion_sum,output[e+'_total'],rtol=2e-11,
                atol=2e-12*np.max(abs(output[e+'_total'])),err_msg='ion force partition does not close')
    return output, float(closure)


def saved_departures(directory):
    """Load the exact explicitly saved level populations, without object pickle."""
    metadata = json.loads((directory/'metadata.json').read_text())
    if not metadata['converged'] or metadata.get('smoke_only', True):
        raise ValueError('force diagnostic requires converged, non-smoke metal populations')
    counts = {e:{int(k):v for k,v in c.items()} for e,c in metadata['levels_per_charge'].items()}
    departures = {}
    with np.load(directory/'populations.npz',allow_pickle=False) as p:
        for e in counts:
            keys = [(str(v[0]),int(v[1]),int(v[2])) for v in p[e+'_level_key']]
            departures[e] = {key:row for key,row in zip(keys,
                p[e+'_population_density']/np.maximum(p[e+'_lte_population_density'],1e-300))}
    return metadata,counts,departures


def line_formation_depths(host, metadata, counts, departures):
    """Return tau_lambda=1 locations, not line contribution functions."""
    from wd_spectra.opacity import optical_depth_from_mass_opacity
    wave = np.array([1175.987,1176.370,1393.755,1402.770])
    data = ModelData.default(); a = host.atmosphere
    database = read_pg1159_atomic_database(data.stout,elements=tuple(counts))
    photo = read_verner_photoionization_database(data.verner_photoionization,
                                                elements=tuple(counts),require_all_elements=True)
    reference = fixed_electron_metal_reference(a,database,metadata['abundances'])
    base = _model_from_config(host.config,data).transfer_coefficients(a,wave,host.population_state)
    opacity = base.total_extinction.copy()
    for e in counts:
        bb,bf,_ = element_coefficients(a,wave,database,photo,reference,e,counts[e],departures[e])
        opacity += bb+bf
    tau = optical_depth_from_mass_opacity(a.column_mass,opacity)
    base_tau = optical_depth_from_mass_opacity(a.column_mass,base.total_extinction)
    result = {}
    for i,w in enumerate(wave):
        if not (tau[i,0]<1<tau[i,-1] and base_tau[i,0]<1<base_tau[i,-1]):
            raise ValueError('tau=1 is outside atmosphere')
        result[str(w)] = dict(element='C' if w<1200 else 'Si',
            core_mass=float(np.exp(np.interp(0.,np.log(tau[i]),np.log(a.column_mass)))),
            background_mass=float(np.exp(np.interp(0.,np.log(base_tau[i]),np.log(a.column_mass)))))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--refinement',type=int,choices=(0,1,2),default=1)
    parser.add_argument('--ion-breakdown',action='store_true',help='also integrate each ion stage and verify momentum closure')
    args = parser.parse_args()
    input_paths=[args.model/'metadata.json',args.model/'populations.npz',Path(__file__)]
    input_hashes={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in input_paths}
    metadata,counts,departures = saved_departures(args.model)
    data = ModelData.default(); before = identity(data)
    host = load_frozen_host(Path(metadata.get('host_directory',args.model)),data)
    database = read_pg1159_atomic_database(data.stout,elements=tuple(counts))
    wave = radiation_grid(host,database,counts,args.refinement)
    print(f'Evaluating {len(wave)} wavelengths, {host.atmosphere.n_depth} depths',flush=True)
    output,closure = evaluate_forces(host,metadata['abundances'],departures,counts,wave,
                                    database=database,ion_breakdown=args.ion_breakdown)
    if identity(data) != before: raise RuntimeError('code/data changed during force calculation')
    if any(hashlib.sha256(p.read_bytes()).hexdigest()!=input_hashes[str(p.resolve())] for p in input_paths):
        raise RuntimeError('metal populations or force code changed during calculation')
    args.output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output/'forces.npz',**output)
    g = float(output['gravity']); m = output['column_mass']
    report = dict(experimental=True,diffusion_equilibrium_solved=False,
        homogeneous_spectrum_changed=False,model=str(args.model.resolve()),
        numerical_identity=before,wavelength_points=len(wave),source_closure=closure,
        model_inputs=input_hashes,wavelength_range_angstrom=[float(wave[0]),float(wave[-1])],
        levels_per_charge=counts,
        force_partition_by_charge=args.ion_breakdown,
        force_convention='centered net extinction times local F_lambda / c / element mass fraction',
        bound_free_momentum='all assigned to element; photoelectron redistribution absent',
        missing_physics=['Fe/Ni and other metal blanketing','thermal diffusion','winds/accretion',
                         'full ion ladder opacity','diffusion boundary conditions'],
        sample_depths={})
    for mass in (1e-6,1e-5,1e-4,1e-3,1e-2,.03,.1,1.):
        report['sample_depths'][str(mass)] = {
            key:float(np.interp(np.log(mass),np.log(m),output[key]/g))
            for key in ('C_bb','C_bf','C_total','Si_bb','Si_bf','Si_total','total_acceleration')}
    (args.output/'forces.json').write_text(json.dumps(report,indent=2)+'\n')
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(1,2,figsize=(11,4.5),sharey=True)
    for ax,e in zip(axes,counts):
        for key,label,style in [('total','BB + BF','-'),('bb','Bound-bound','--'),
                                 ('hst_uv','1150–1700 Å only',':')]:
            ax.semilogx(m,output[e+'_'+key]/g,style,label=label)
        ax.axhline(1.,color='black',lw=.8,label='Gravity')
        ax.set(xlabel=r'Column mass (g cm$^{-2}$)',title=e,xlim=(1e-7,1.),ylim=(0,None))
        ax.grid(alpha=.2); ax.legend(fontsize=9)
    axes[0].set_ylabel(r'Radiative acceleration / gravity')
    fig.suptitle('G191-B2B: photon force at published uniform C/Si abundances\n'
                 'Fixed H/He atmosphere; experimental atoms; no diffusion equilibrium',fontsize=11)
    fig.tight_layout();fig.savefig(args.output/'forces.png',dpi=180);fig.savefig(args.output/'forces.pdf')
    print(json.dumps(report['sample_depths'],indent=2),flush=True)


if __name__ == '__main__': main()
