"""Locate structure/synthesis quadrature error without altering a saved state."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import time
import numpy as np
from wd_spectra import gray_helium_atmosphere
from wd_spectra._compat import trapezoid
from wd_spectra.models.common import ModelData
from wd_spectra.models.dq import DQConfig
from wd_spectra._dq.runtime import make_material, numerical_policy
from wd_spectra._dq.validation import independent_grid
from wd_spectra._dq.provenance import digest, source_hashes
from wd_spectra.constants import STEFAN_BOLTZMANN


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--state',type=Path,required=True)
    p.add_argument('--grid',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--infrared',action='store_true')
    p.add_argument('--double-ir-from',type=Path)
    p.add_argument('--ultraviolet',action='store_true')
    p.add_argument('--teff',type=float,default=5529.)
    p.add_argument('--logg',type=float,default=8.178)
    p.add_argument('--log-c-he',type=float,default=-5.27)
    args=p.parse_args()
    if args.infrared and args.ultraviolet:p.error('Select only one spectral interval')
    if args.double_ir_from and not args.infrared:p.error('--double-ir-from requires --infrared')
    args.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    config=DQConfig(args.teff,args.logg,args.log_c_he)
    provenance=dict(scope=__doc__,cold_start=False,atmosphere_reconverged=False,
        config=asdict(config),state_sha256=digest(args.state),grid_sha256=digest(args.grid),
        source_sha256=source_hashes())
    with numerical_policy(args.output):
        mat=make_material(config,ModelData.default(),args.output)
        with np.load(args.state) as z:
            for key in ('effective_temperature','logg'):
                if key in z and float(z[key])!=getattr(config,key):
                    raise ValueError('Saved state and requested '+key+' differ')
            seed=gray_helium_atmosphere(config.effective_temperature,config.logg,n_depth=len(z['temperature']))
            a=mat.chemistry(replace(seed,**{k:z[k].copy() for k in
                ('temperature','gas_pressure','column_mass','rosseland_optical_depth')}))[0]
        with np.load(args.grid) as z:original=z['wavelength'].copy()
        grids=dict(original=original,
            full_continuum=np.unique(np.r_[original,np.geomspace(1000,100000,4000)]),
            independent=independent_grid())
        if args.infrared:
            count=8*int(np.ceil(3000*np.log(100000/6800)))
            grids=dict(infrared=np.geomspace(6800,100000,count+1))
            if args.double_ir_from:
                previous=np.load(args.double_ir_from)
                grids=dict(infrared=np.sqrt(previous['wavelength'][:-1]*previous['wavelength'][1:]))
        if args.ultraviolet:
            uv_base=np.geomspace(1000,100000,4000)
            uv_base=np.r_[uv_base[uv_base<3800],3800.]
            uv_count=8*int(np.ceil(3000*np.log(3800/1000)))
            uv_fine=np.geomspace(1000,3800,uv_count+1)
            grids=dict(ultraviolet=np.unique(np.r_[uv_base,uv_fine]))
        output={}
        for name,w in grids.items():
            spectrum=mat.spectrum(a,w,3)
            f=spectrum.surface_flux_lambda;norm=STEFAN_BOLTZMANN*config.effective_temperature**4
            if args.double_ir_from:
                w,order=np.unique(np.r_[w,previous['wavelength']],return_index=True)
                f=np.r_[f,previous['flux']][order]
            intervals={}
            for lo,hi in ((1000,3800),(3800,6800),(6800,100000)):
                if args.infrared and lo<6800:continue
                if args.ultraviolet and hi>3800:continue
                q=np.unique(np.r_[lo,w[(w>lo)&(w<hi)],hi])
                intervals[f'{lo}:{hi}']=float(trapezoid(np.interp(q,w,f),q)/norm)
            output[name]=dict(points=len(w),ratio=float(trapezoid(f,w)/norm),intervals=intervals)
            if args.infrared:
                output[name]['nested_integrals']={str((48000 if args.double_ir_from else 24000)//stride):float(trapezoid(f[::stride],w[::stride])/norm)
                    for stride in (8,4,2,1)}
            if args.ultraviolet:
                output[name]['broadband_integral']=float(trapezoid(f[np.searchsorted(w,uv_base)],uv_base)/norm)
                output[name]['nested_integrals']={str(24000//stride):float(trapezoid(f[np.searchsorted(w,uv_fine[::stride])],uv_fine[::stride])/norm)
                    for stride in (8,4,2,1)}
            np.savez_compressed(args.output/(name+'.npz'),wavelength=w,flux=f)
            (args.output/'report.json').write_text(json.dumps(dict(provenance,results=output,seconds=time.monotonic()-started),indent=2)+'\n')
            print(json.dumps(output[name]),flush=True)


if __name__=='__main__':main()
