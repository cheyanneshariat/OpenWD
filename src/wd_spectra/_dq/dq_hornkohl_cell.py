from .data import line_root
"""DQ-local finite-cell Swan opacity experiment with the existing solver.

Temperature and molecular abundance are constant within each material cell.
The pressure varies hydrostatically between mass-midpoint cell faces; the
same coupled REOS/He/C chemistry determines the two face densities. Integrate
thermal Swan profiles over their resulting LINEAR density-shift interval.
This is a mesh-dependent spatial quadrature, not extra collision broadening.
Both convergence and synthesis use it. Independent depth tests are required.
"""
from dataclasses import replace

import numpy as np

from .dq_exact_opacity_cache import subset, ExactLayerOpacityCache
from .dq_hornkohl_consistent import ConsistentDQMaterial
from .dq_bounded_helium_lines import bounded_helium_lines
from .dq_swan_depth_integral import SpatiallyIntegratedSwan
from wd_spectra.c2_profiles import swan_density_shift_wavenumber


def pressure_faces(pressure):
    p=np.asarray(pressure,dtype=float)
    if p.ndim!=1 or len(p)<2 or np.any(~np.isfinite(p)) or np.any(p<=0) or np.any(np.diff(p)<=0):
        raise ValueError('Ordered positive pressures are required')
    middle=.5*(p[1:]+p[:-1])
    # Do not extrapolate the constitutive tables beyond the solved domain.
    # Outer unresolved tails retain the boundary-state convention.
    return np.r_[p[0],middle],np.r_[middle,p[-1]]


class FastConsistentDQMaterial(ConsistentDQMaterial):
    def helium_line_opacity(self,a,wavelength,background):
        value,info=bounded_helium_lines(a,wavelength,self.he_i,background)
        self.maximum_helium_opacity_bound=max(getattr(self,'maximum_helium_opacity_bound',0.),info['relative_bound'])
        return value


class CellIntegratedDQMaterial(FastConsistentDQMaterial):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.integrated_swan=SpatiallyIntegratedSwan(self.table,branches=line_root()/'hornkohl_calibrated.npz',cache_grids=192)
        self.cell_pressure_grid=None
        self.cell_shifts={}

    def prepare_opacity_state(self,a,carbon,c2):
        p=a.gas_pressure
        if self.cell_pressure_grid is None or not np.array_equal(p,self.cell_pressure_grid):
            # A remesh changes numerical cell geometry even at retained nodes.
            self.opacity_cache=ExactLayerOpacityCache()
            self.cell_pressure_grid=p.copy()
        left,right=pressure_faces(p)
        indices=np.repeat(np.arange(a.n_depth),2)
        face=subset(a,indices,a.n_depth)
        face=replace(face,gas_pressure=np.column_stack((left,right)).ravel())
        saved=self.cached
        state=self.chemistry(face)[0]
        self.cached=saved
        shifts=swan_density_shift_wavenumber(state.mass_density).reshape(-1,2)
        self.cell_shifts={float(pj):tuple(s) for pj,s in zip(p,shifts)}

    def swan_column(self,a,j,wavelength):
        left,right=self.cell_shifts[float(a.gas_pressure[j])]
        return self.integrated_swan.cell_average(wavelength,float(a.temperature[j]),left,right)


