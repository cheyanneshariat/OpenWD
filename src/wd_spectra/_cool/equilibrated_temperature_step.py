"""Unregularized square Newton direction with explicit physical T bounds.

No singular directions are discarded. Singular equations fail explicitly.
The full nonlinear driver must still accept the resulting physical trial.
"""
import numpy as np


def equilibrated_step(state,evaluation,jacobian,radius,method,**settings):
    mapping=evaluation.payload['log_temperature_from_state']
    matrix=np.linalg.solve(mapping.T,jacobian.T).T
    rows=np.max(abs(matrix),axis=1)
    if np.any(rows==0):raise ValueError('unconstrained energy equation')
    scaled=matrix/rows[:,None]
    columns=np.max(abs(scaled),axis=0)
    if np.any(columns==0):raise ValueError('unconstrained temperature variable')
    delta=np.linalg.solve(scaled/columns[None,:],-evaluation.residual/rows)/columns
    unbounded=float(np.max(abs(delta)))
    delta*=min(1.,radius/max(unbounded,np.finfo(float).tiny))
    print(f'Equilibrated square Newton: unbounded max dlnT={unbounded:.6g}; '
        f'bounded={np.max(abs(delta)):.6g}',flush=True)
    return np.linalg.solve(mapping,delta)
