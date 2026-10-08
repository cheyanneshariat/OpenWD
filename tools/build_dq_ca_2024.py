"""Matched-coverage C-A screen using Lino da Silva (2024) Einstein A values.

Slide 20 lower table: rows v_upper, columns v_lower, A in s^-1. Only the
existing 0..6 by 0..8 coverage is used. Nothing is fitted to WD spectra.
The inherited approximate rotational envelope and band origins are unchanged.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from build_dq_ca_historical import (
    band_constants, oscillator_strength, lower_states, FC, HC_K,
    geometry_for_band, exact_band_fractions,
)
from wd_spectra.constants import ELEMENTARY_CHARGE_ESU, ELECTRON_MASS, LIGHT_SPEED
from wd_spectra.carbon_molecular import read_c2_cross_section_table
from wd_spectra._dq.provenance import digest, source_hashes

SOURCE = 'https://indico.esa.int/event/466/contributions/9848/attachments/6164/10489/C2-deslandres-dazambuja.pdf'
A = np.array([
 [1.711e7,8.536e6,2.612e6,6.400e5,1.385e5,2.777e4,5.343e3,1.034e3,2.141e2],
 [1.242e7,2.599e6,7.423e6,4.413e6,1.631e6,4.781e5,1.228e5,2.948e4,6.984e3],
 [3.229e6,1.328e7,2.416e4,3.946e6,4.641e6,2.499e6,9.615e5,3.088e5,9.018e4],
 [3.542e5,6.784e6,9.732e6,1.699e6,1.219e6,3.688e6,2.903e6,1.444e6,5.705e5],
 [1.246e4,1.078e6,9.369e6,5.636e6,3.802e6,6.626e4,2.275e6,2.760e6,1.766e6],
 [3.070e-1,4.261e4,1.997e6,1.070e7,2.698e6,4.826e6,1.992e5,1.024e6,2.191e6],
 [3.306e1,2.804e2,6.633e4,2.778e6,1.112e7,1.226e6,4.656e6,8.738e5,2.873e5],
])


def f_from_A(aval, origin):
    # g_upper/g_lower = 1 for singlet Pi -> singlet Pi. Wavenumber in cm^-1.
    return ELECTRON_MASS*LIGHT_SPEED/(8*np.pi**2*ELEMENTARY_CHARGE_ESU**2)*aval/origin**2


def build(output, states_path, pdf):
    output.mkdir(parents=True, exist_ok=False)
    parent = Path('src/wd_spectra/data/dq/c2-ca-historical.npz')
    with np.load(parent, allow_pickle=False) as z:
        data = {k:z[k].copy() for k in z.files}
    states = lower_states(states_path)
    centers = (1e8/data['wavelength_angstrom'])[::-1]
    dl = np.diff(np.log(centers))
    assert np.allclose(dl, dl[0], rtol=1e-7)
    edges = np.r_[centers*np.exp(-dl[0]/2), centers[-1]*np.exp(dl[0]/2)]
    widths = np.diff(edges)
    bands=[]
    for vu in range(7):
        for vl in range(9):
            nu, bu, bl = band_constants(vu, vl)
            bands.append(dict(vu=vu, vl=vl, origin=nu, bl=bl, A=float(A[vu,vl]),
                f=float(f_from_A(A[vu,vl],nu)), old_f=oscillator_strength(nu,FC[vu,vl]),
                geometry=geometry_for_band(edges,nu,bu,bl)))
    candidate=np.zeros_like(data['cross_section'])
    reproduced=np.zeros_like(candidate)
    for it,t in enumerate(data['temperature_K']):
        q=np.exp(np.interp(np.log(t),np.log(data['partition_temperature_K']),np.log(data['partition_function'])))
        pop=np.bincount(states[:,2].astype(int),weights=states[:,1]*np.exp(-HC_K*states[:,0]/t),minlength=9)/q
        for b in bands:
            fractions,_=exact_band_fractions(b['geometry'],b['bl'],float(t),len(centers))
            unit=(np.pi*ELEMENTARY_CHARGE_ESU**2/(ELECTRON_MASS*LIGHT_SPEED**2)
                  *pop[b['vl']]*fractions/widths*centers/b['origin']*-np.expm1(-HC_K*centers/t))
            candidate[:,it] += (unit*b['f'])[::-1]
            reproduced[:,it] += (unit*b['old_f'])[::-1]
        if it%10==0: print(f'Table {it+1}/{len(data["temperature_K"])}',flush=True)
    err=float(np.max(abs(reproduced-data['cross_section']))/np.max(data['cross_section']))
    # The release table was extracted by subtracting two larger opacity tables;
    # negligible tails contain cancellation/zeros. Bound absolute error by peak.
    np.testing.assert_allclose(reproduced,data['cross_section'],rtol=2e-12,
                               atol=2e-12*np.max(data['cross_section']))
    assert np.all(np.isfinite(candidate)) and np.all(candidate>=0)
    for b in bands: del b['geometry']
    report=dict(scope=__doc__,source=SOURCE,pdf_sha256=digest(pdf),parent_sha256=digest(parent),
                states_sha256=digest(states_path),harness_sha256=digest(__file__),
                old_table_reproduction_max_absolute_over_peak=err,bands=bands,
                A00_lifetime_partial_ns=float(1e9/A[0].sum()),
                f00_new_over_old=bands[0]['f']/bands[0]['old_f'])
    data['cross_section']=candidate
    data['source']='Lino da Silva 2024 C-A A coefficients, matched 63-band approximate envelope'
    data['provenance_json']=json.dumps(report)
    np.savez_compressed(output/'ca-2024.npz',**data)
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='bands'},indent=2),flush=True)




if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--states', type=Path, required=True)
    p.add_argument('--pdf', type=Path, required=True)
    a = p.parse_args()
    if digest(a.pdf) != '414f7aa1e719f291ea46fcc3ff016a11d3765206ef0f53ab717f902e7cb2bccf':
        raise ValueError('C–A source PDF differs from the audited 2024 presentation')
    build(a.output, a.states, a.pdf)
