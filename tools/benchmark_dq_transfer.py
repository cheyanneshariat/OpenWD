"""Compare production transfer with the frozen pre-optimization test algebra.

Run in a separate process, e.g. OPENBLAS_NUM_THREADS=1 PYTHONPATH=src
python tools/benchmark_dq_transfer.py. Compilation is excluded. This measures
one analytic-response workload, not a full atmosphere calculation.
"""
import json
from pathlib import Path
import sys
import time
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tests'))
import dq_reference_kernels as reference
from wd_spectra._dq import dq_refractive_finite_volume as fv
from wd_spectra._dq import dq_refractive_fv_response as response


def main():
    mass=np.geomspace(1e-4,12,45);wave=np.linspace(3800,6800,100)
    t=np.linspace(.8,2.,len(mass));r=(wave/5000)[:,None]
    a=.3*r*t**.7;s=.5*r*t**-.3;b=r*t**4
    n=1+.12*r*(1-np.exp(-mass))*t**-.2
    inputs=(mass,wave,a,s,b,n,.7*a,-.3*s,4*b,-.2*(n-1))
    production_ray,production_known=fv.ray_solve,response.known_response
    outputs=[];times=[]
    try:
        for ray,known in ((reference.ray_solve,reference.known_response),
                          (production_ray,production_known)):
            fv.ray_solve=ray;response.known_response=known
            fv.operators.recompile()
            outputs.append(response.integrated_response(*inputs))
            samples=[]
            for _ in range(3):
                start=time.monotonic()
                response.integrated_response(*inputs)
                samples.append(time.monotonic()-start)
            times.append(float(np.median(samples)))
        for a,b in zip(*outputs):np.testing.assert_array_equal(a,b)
    finally:
        fv.ray_solve=production_ray;response.known_response=production_known
        fv.operators.recompile()
    print(json.dumps(dict(wavelengths=100,depths=45,bitwise_equal=True,
        median_reference_seconds=times[0],median_production_seconds=times[1],
        speedup=times[0]/times[1],compilation_included=False,
        full_atmosphere_speedup_measured=False),indent=2))


if __name__=='__main__':main()
