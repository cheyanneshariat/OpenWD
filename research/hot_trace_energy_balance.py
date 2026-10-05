"""Measure metal feedback omitted by the fixed-host approximation; never certify it."""
import hashlib
import json
from pathlib import Path

import numpy as np

from wd_spectra._compat import trapezoid
from wd_spectra.constants import STEFAN_BOLTZMANN
from wd_spectra._hot_structure import HotEquations
from wd_spectra._mass_feautrier import mass_emissivity_energy
from wd_spectra.hot_nlte import transfer_field
from wd_spectra.hot_trace_metals import fixed_electron_metal_reference, _atom_selection
from wd_spectra.light_metal_nlte import (hot_metal_line_nlte_coefficients,
                                       light_metal_bound_free_nlte_coefficients)
from wd_spectra.models.common import numerical_resolution
from wd_spectra.models.hot import structure_wavelength
from wd_spectra.nlte_core import NLTETransferCoefficients


def audit_energy_balance(host,model,database,photo,abundances,counts,thresholds,
                         states,rate_wave,background,host_background,output):
    """Compare H/He and H/He+metals on the same combined thermal/rate grid.

    This is a frozen-state diagnostic. The added grid points do not constitute
    a newly converged population solution or a thermal-structure update.
    """
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    atmosphere=host.atmosphere
    host_wave=HotEquations(atmosphere,model,structure_wavelength(host.config,
        numerical_resolution(host.config.quality).n_continuum)).wave
    wave=np.unique(np.r_[host_wave,rate_wave])
    target=STEFAN_BOLTZMANN*atmosphere.effective_temperature**4
    report={};arrays={'wavelength':wave,'column_mass':atmosphere.column_mass}
    def evaluate(label,coefficients):
        _,field,closure=transfer_field(atmosphere,coefficients,n_angle=model.n_angle)
        relative_flux=trapezoid(field.interface_flux,wave,axis=0)/target-1
        energy,emission=mass_emissivity_energy(wave,atmosphere.column_mass,
            coefficients.thermal_emissivity,field.mean_intensity,coefficients.true_absorption)
        energy_residual=energy/np.maximum(abs(emission),1e-30*target)
        report[label]=dict(surface_flux_ratio=float(relative_flux[0]+1),
            maximum_all_depth_flux_residual=float(np.max(abs(relative_flux))),
            maximum_relative_cell_energy_residual=float(np.max(abs(energy_residual))),
            source_closure_residual=float(closure))
        arrays[label+'_relative_flux']=relative_flux
        arrays[label+'_relative_cell_energy']=energy_residual
    evaluate('HHe',host_background(wave))
    base=background(wave)
    absorption=base.true_absorption.copy();emission=base.thermal_emissivity.copy()
    reference=fixed_electron_metal_reference(atmosphere,database,abundances)
    unity={(e,stage.charge):np.ones(atmosphere.n_depth)
           for e in abundances for stage in database.ion_stages(e)}
    for element in abundances:
        lines,bound_counts=_atom_selection(database,element,counts[element])
        departures=states[element].level_departure_coefficient
        a,j=hot_metal_line_nlte_coefficients(atmosphere,wave,database,reference,unity,
            elements=(element,),level_departure_coefficient=departures,transition_keys=lines,
            minimum_oscillator_strength=0.,maximum_lines=None,include_static_linear_stark=False,
            retain_inverted_emissivity=True)
        b,k=light_metal_bound_free_nlte_coefficients(atmosphere,wave,database,reference,photo,unity,
            elements=(element,),level_departure_coefficient=departures,levels_per_charge=bound_counts,
            include_explicit_kramers=True,photoionization_threshold_data=thresholds.get(element))
        absorption+=a+b;emission+=j+k
    evaluate('WithMetals',NLTETransferCoefficients(wave,absorption,emission,base.scattering,{}))
    report.update(wavelength_points=len(wave),wavelength_limits_angstrom=[float(wave[0]),float(wave[-1])],
        metal_induced_surface_flux_change=float(report['WithMetals']['surface_flux_ratio']-report['HHe']['surface_flux_ratio']),
        maximum_metal_induced_flux_change=float(np.max(abs(arrays['WithMetals_relative_flux']-arrays['HHe_relative_flux']))),
        atmosphere_recomputed=False,thermal_equilibrium_with_metals_certified=False,
        qualification='Frozen populations on a combined host/rate quadrature; H/He control exposes quadrature effects.',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (output/'energy-balance.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez(output/'energy-balance.npz',**arrays)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for label,color in [('HHe','#7c90a0'),('WithMetals','#c3473d')]:
        # There are nd interfaces and nd-1 material energy cells; the last
        # atmospheric point imposes the thermal lower boundary.
        mass=np.r_[.5*atmosphere.column_mass[0],
                   .5*(atmosphere.column_mass[:-1]+atmosphere.column_mass[1:])]
        axes[0].semilogx(mass,arrays[label+'_relative_flux'],color=color,label=label)
        axes[1].semilogx(atmosphere.column_mass[:-1],arrays[label+'_relative_cell_energy'],color=color,label=label)
    for ax in axes:ax.axhline(0.,c='.3',ls='--');ax.set_xlabel('Column mass (g cm⁻²)')
    axes[0].set(ylabel='F / (σ Teff⁴) − 1',title='Flux imbalance at fixed structure');axes[0].legend()
    axes[1].set(ylabel='Net cell heating / emission',title='Local energy imbalance')
    fig.suptitle('Metal feedback diagnostic: this does not solve the atmospheric structure')
    fig.savefig(output/'energy-balance.png',dpi=160);fig.savefig(output/'energy-balance.pdf');plt.close(fig)
    return report
