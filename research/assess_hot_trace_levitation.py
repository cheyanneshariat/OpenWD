#!/usr/bin/env python3
"""Assess saved force quadrature and atom-size tests near the UV line depths."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

from hot_trace_levitation import (saved_departures,line_formation_depths,
    mean_element_charge,zero_drift_abundance_gradient,homogeneous_support_acceleration)
from resynthesize_hot_trace import load_frozen_host, identity
from wd_spectra.hot_trace_metals import fixed_electron_metal_reference
from wd_spectra.metals import ATOMIC_MASS_U, read_pg1159_atomic_database
from wd_spectra.models.common import ModelData


def assess(baseline, finer, compact, output):
    paths=list(map(Path,(baseline,finer,compact)));output=Path(output)
    records=[json.loads((p/'forces.json').read_text()) for p in paths]
    data=ModelData.default(); numerical=identity(data)
    if any(r['numerical_identity']!=numerical for r in records):
        raise ValueError('force calculation code/data no longer match')
    model=Path(records[0]['model'])
    if records[1]['model']!=records[0]['model']:
        raise ValueError('quadrature check must use the same fixed populations')
    meta,counts,departures=saved_departures(model)
    model_records=[json.loads((Path(r['model'])/'metadata.json').read_text()) for r in records]
    for r in model_records:
        if any(r[key]!=meta[key] for key in ('host_parameters','abundances','host_directory')):
            raise ValueError('force sensitivity tests must share a host and abundances')
    host=load_frozen_host(Path(meta['host_directory']),data)
    arrays=[dict(np.load(p/'forces.npz',allow_pickle=False)) for p in paths]
    base,fine,small=arrays;m=base['column_mass'];g=float(base['gravity'])
    for result in arrays[1:]:
        np.testing.assert_array_equal(result['column_mass'],m)
        np.testing.assert_array_equal(result['gravity'],base['gravity'])
        np.testing.assert_array_equal(result['temperature'],base['temperature'])
    formation=line_formation_depths(host,meta,counts,departures)
    database=read_pg1159_atomic_database(data.stout,elements=tuple(counts))
    reference=fixed_electron_metal_reference(host.atmosphere,database,meta['abundances'])
    gradients={};checks={};sel=(m>=1e-4)&(m<=1.)
    with np.load(model/'populations.npz',allow_pickle=False) as p:
        for e in counts:
            keys=[(str(k[0]),int(k[1]),int(k[2])) for k in p[e+'_level_key']]
            q=mean_element_charge(reference,e,keys,p[e+'_population_density'],p[e+'_lte_population_density'])
            gradients[e+'_mean_charge']=q
            for name,force in [('off',np.zeros_like(m)),('on',fine[e+'_total'])]:
                gradients[e+'_'+name]=zero_drift_abundance_gradient(host.atmosphere,ATOMIC_MASS_U[e],q,force)
            gradients[e+'_homogeneous_support_acceleration']=homogeneous_support_acceleration(
                host.atmosphere,ATOMIC_MASS_U[e],q)
            checks[e]=dict(
                maximum_quadrature_relative_change=float(np.max(abs(fine[e+'_total'][sel]/base[e+'_total'][sel]-1))),
                maximum_atom_expansion_relative_change=float(np.max(abs(base[e+'_total'][sel]/small[e+'_total'][sel]-1))))
    for row in formation.values():
        e=row['element'];mass=row['core_mass']
        sample=lambda a:float(np.interp(np.log(mass),np.log(m),a))
        for name,force in [('compact',small),('expanded',base),('finer',fine)]:
            row[name+'_g_rad_over_g']=sample(force[e+'_total']/g)
        row['bound_bound_g_rad_over_g']=sample(fine[e+'_bb']/g)
        row['hst_band_g_rad_over_g']=sample(fine[e+'_hst_uv']/g)
        row['mean_charge']=sample(gradients[e+'_mean_charge'])
        row['homogeneous_support_acceleration_over_g']=sample(gradients[e+'_homogeneous_support_acceleration']/g)
        row['ion_contributions_to_g_rad_over_g']={
            str(q):sample(base[f'{e}_charge_{q}']/g) for q in counts[e]
            if f'{e}_charge_{q}' in base}
        row['required_dln_abundance_dln_mass_without_radiation']=sample(gradients[e+'_off'])
        row['required_dln_abundance_dln_mass_with_radiation']=sample(gradients[e+'_on'])
    output.mkdir(parents=True,exist_ok=True)
    report=dict(experimental=True,diffusion_equilibrium_solved=False,
        spectrum_with_diffusion_predicted=False,existing_spectrum_altered=False,
        host_parameters=meta['host_parameters'],abundances=meta['abundances'],
        numerical_identity=numerical,quadrature_wavelength_points=[len(a['wave']) for a in arrays],
        tested_atoms=[r['levels_per_charge'] for r in model_records],
        maximum_bulk_force_over_g=float(np.max(fine['total_acceleration']/g)),
        checks_region_g_cm2=[1e-4,1.],checks=checks,line_depth_estimates=formation,
        formation_caveat='monochromatic tau=1 interpolation; not a line contribution function',
        gradient_caveat='zero-drift force-balance diagnostic with electron-pressure electric field; '
            'no thermal diffusion, ion-specific mobilities, accretion or wind; '
            'forces must be recomputed before integrating an abundance profile',
        inputs={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [model/'populations.npz',model/'metadata.json']+[d/'forces.npz' for d in paths]},
        literature=[dict(reference='Rauch et al. 2013, sections 4.3–4.4',
            url='https://doi.org/10.1051/0004-6361/201322336',
            note='Their 60000 K/logg 7.60 diffusion model did not improve the UV metal-line fit; '
                 'this is context, not the parameters used for our Preval 52500 K/logg 7.53 benchmark.')])
    (output/'assessment.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez(output/'diffusion-gradient-diagnostic.npz',column_mass=m,**gradients)
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.6),sharey=True)
    for ax,e in zip(axes,counts):
        ax.semilogx(m,fine[e+'_total']/g,color='#1668a6',lw=2,label='Expanded atom: total force')
        ax.semilogx(m,small[e+'_total']/g,color='#1668a6',ls='--',label='Smaller atom')
        ax.semilogx(m,fine[e+'_hst_uv']/g,color='#bf711b',ls=':',label='1150–1700 Å contribution')
        ax.axhline(1,color='black',lw=.8,label='Gravity')
        depths=[row['core_mass'] for row in formation.values() if row['element']==e]
        ax.axvspan(min(depths),max(depths),color='#a95590',alpha=.25,label=r'UV cores: $\tau_\lambda=1$')
        ax.set(xlabel=r'Column mass (g cm$^{-2}$)',title=e,xlim=(1e-4,1),ylim=(0,5.15))
        ax.grid(alpha=.2);ax.legend(fontsize=8,loc='upper left')
    axes[0].set_ylabel(r'Radiative acceleration / gravity')
    fig.suptitle('G191-B2B: levitation force diagnostic at published uniform abundances\n'
                 '52,500 K · log g = 7.53 · fixed H/He atmosphere · no diffusion equilibrium',fontsize=11)
    fig.tight_layout();fig.savefig(output/'levitation-assessment.png',dpi=180)
    fig.savefig(output/'levitation-assessment.pdf');plt.close(fig)
    print(json.dumps(dict(checks=checks,line_depth_estimates=formation),indent=2))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--finer',type=Path,required=True)
    parser.add_argument('--compact',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();assess(args.baseline,args.finer,args.compact,args.output)
