"""Finite-EOS remainder in a DQ Newton trial model, not physical transfer.

Resolve all REOS temperature knots and finite thermodynamic-window edges
using the unchanged table. Freeze helium host pressure only in this cheap
predictor; match the complete coupled physical tangent exactly at the anchor.
Opacity, refractivity and composition retain their first-order responses.
The outer solve always reconstructs the real chemistry/opacity/transfer.
"""
from contextlib import contextmanager
from contextvars import ContextVar
import numpy as np
from .convective_consistency_experiment import MaterialCoefficients
from .dq_reos_secant_thermodynamics import reos_secant_thermodynamics
from .dq_reos_secant_response import thermodynamic_response

CURRENT_MATERIAL = ContextVar('dq_eos_trial_material', default=None)


@contextmanager
def eos_trial_material(material):
    token=CURRENT_MATERIAL.set(material)
    try:yield
    finally:CURRENT_MATERIAL.reset(token)


class AnchoredCoefficients:
    """Positive nonlinear remainder, with exact supplied origin and tangent."""
    def __init__(self, anchor, values, responses, raw):
        self.anchor=np.asarray(anchor).copy();self.values=values;self.responses=responses
        self.raw=raw
        self.base,self.slope=raw(self.anchor)
        if any(np.any(v<=0) for v in self.base):raise ValueError('Positive coefficient model required')
        self.correction=tuple(d/v[:,None]-r/b[:,None] for v,d,b,r in
                              zip(values,responses,self.base,self.slope))

    def __call__(self, point):
        if np.array_equal(point,self.anchor):return self.values,self.responses
        raw,derivative=self.raw(point);dx=point-self.anchor
        factors=tuple(v/b*np.exp(c@dx) for v,b,c in zip(self.values,self.base,self.correction))
        values=tuple(f*r for f,r in zip(factors,raw))
        responses=tuple(f[:,None]*(d+r[:,None]*c) for f,d,r,c in
                         zip(factors,derivative,raw,self.correction))
        return values,responses


def coefficient_model(system, state, ev, *, anchor_material=None):
    owner=system.eos_trial_owner
    if owner.helium_reos3 is None:raise ValueError('REOS trial model requires the declared table')
    p=ev.payload;anchor=state[:system.n].copy();a=p['atmosphere']
    pressure=np.asarray(a.metadata['dq_helium_host_pressure'])
    table=owner.helium_reos3
    thermal=(table.thermodynamics if hasattr(table,'thermodynamics') else
             lambda t,p:reos_secant_thermodynamics(table,t,p))
    base_rho=table.evaluate(pressure,a.temperature)[0]
    base_thermo=thermal(a.temperature,pressure)
    actual=p['convection_transport']
    # Exact table nonlinearity at the actual node's helium-host pressure.
    # Fixed normalization only maps host quantities to this mixture's anchor.
    scales=np.asarray([a.mass_density/base_rho,
        actual['specific_heat']/base_thermo.specific_heat_constant_pressure,
        actual['expansion']/base_thermo.density_temperature_derivative,
        actual['node_adiabatic_gradient']/base_thermo.adiabatic_temperature_gradient])
    assembly=MaterialCoefficients(None,None,None,owner.config.mixing_length_alpha)
    from dataclasses import replace
    def fields(x):
        t=np.exp(x);rho=table.evaluate(pressure,t)
        if not np.all(rho[2]):raise ValueError('Finite EOS trial leaves the table domain')
        th=thermal(t,pressure)
        f=np.asarray([rho[0],th.specific_heat_constant_pressure,
            th.density_temperature_derivative,th.adiabatic_temperature_gradient])*scales
        return replace(a,temperature=t,mass_density=f[0]),np.asarray([
            f[0],actual['rosseland'],f[1],f[2],f[3]])
    def raw(x):
        atmosphere,f=fields(x)
        response=thermodynamic_response(table,np.exp(x),pressure)*scales
        slopes=np.asarray([response[0],np.zeros_like(x),*response[1:]])
        return assembly.assemble(atmosphere,f,slopes)
    values,responses=(system.material(anchor,True) if anchor_material is None else anchor_material)
    return AnchoredCoefficients(anchor,values,responses,raw)
