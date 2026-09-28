"""Update energy weights between linearizations, never within a line search.

The stationary physical equations are unchanged. No derivative of an arbitrary
equation weight is added. Current, independently evaluated physical residuals
must still pass the outer convergence test. Intended for refreshed-Jacobian
proposals that use the supplied material/transfer payload, not stale secants.
"""
import numpy as np
from wd_spectra.constants import STEFAN_BOLTZMANN
from wd_spectra.nonlinear import NonlinearEvaluation


def frozen_energy_evaluator(evaluate):
    scale=None
    def weighted(state,need_jacobian):
        nonlocal scale
        ev=evaluate(state,need_jacobian);p=ev.payload
        if not p['energy_balance_is_physical_flux']:
            return ev
        target=STEFAN_BOLTZMANN*p['atmosphere'].effective_temperature**4
        if scale is None or need_jacobian:
            scale=np.array(p['cell_energy_scale'],copy=True)
            if np.any(~np.isfinite(scale)) or np.any(scale<=0):
                raise ValueError('energy weights must be finite positive')
        energy=p['radiative_cell_energy_defect']+np.diff(p['convective_flux_interface'])
        r=p['total_flux_interface']/target-1
        r[:-1]-=energy/scale
        j=None
        if need_jacobian:
            # Reuse the already evaluated face response directly; summing its
            # cell differences loses small couplings in efficient convection.
            convective=p['convective_flux_log_temperature_jacobian']
            j=(p['radiative_flux_log_temperature_jacobian']+convective)/target
            j[:-1]-=p['cell_energy_log_temperature_jacobian']/scale[:,None]
            j=j@p['log_temperature_from_state']
        payload={**p,'diagnostic_fixed_cell_scale':scale.copy()/target,
            'experimental_weights_frozen_within_linearization':True}
        return NonlinearEvaluation(r,j,payload)
    return weighted
