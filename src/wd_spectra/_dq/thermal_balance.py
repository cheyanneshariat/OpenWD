"""Implicit thermal-relaxation diagnostic with unchanged equilibrium root.

At fixed column mass, Q=F_bottom-F_top heats a cell. Freeze its positive
cp*dm at the starting state and add cp*dm*(T-T_start)/dt to -Q. The bottom
flux and gradient compatibility remain algebraic equations. This is a
pseudo-time preconditioner, not a claimed time-dependent stellar model.
No transient residual is ever a steady-atmosphere convergence certificate.
"""
import numpy as np


def capacity(payload):
    a=payload['atmosphere']
    result=np.diff(a.column_mass)*payload['convection_transport']['specific_heat'][:-1]*a.temperature[:-1]
    if np.any(~np.isfinite(result)) or np.any(result<=0):
        raise ValueError('Positive finite cell thermal capacity required')
    return result


def thermal_rows(system,state,ev,anchor,heat_capacity,dt):
    if dt<=0 or not np.isfinite(dt):raise ValueError('Positive finite pseudo timestep required')
    n=system.n;p=ev.payload;target=system.target
    cv=p['dq_augmented_auxiliary_flux']
    q=p['radiative_cell_energy_defect']+np.diff(cv)
    d=heat_capacity/(target*dt)
    # Fixed row units for this time step, not iterate-dependent weights.
    scale=np.maximum(1.,d)
    x=state[:n-1]-anchor[:n-1]
    r=ev.residual.copy()
    r[:n-1]=(d*np.expm1(x)-q/target)/scale
    r[n-1]=(p['radiative_flux_interface'][-1]+cv[-1])/target-1
    j=None
    if ev.jacobian is not None:
        cj=p['dq_augmented_auxiliary_flux_state_jacobian']
        fj=np.column_stack([p['radiative_flux_log_temperature_jacobian'],np.zeros((n,n-1))])+cj
        qj=np.column_stack([p['radiative_cell_energy_log_temperature_jacobian'],np.zeros((n-1,n-1))])+np.diff(cj,axis=0)
        j=ev.jacobian.copy()
        j[:n-1]=-qj/target
        j[np.arange(n-1),np.arange(n-1)]+=d*np.exp(x)
        j[:n-1]/=scale[:,None]
        j[n-1]=fj[-1]/target
    return r,j
