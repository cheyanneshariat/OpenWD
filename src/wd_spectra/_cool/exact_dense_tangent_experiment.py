"""Use the declared REOS interpolant's analytic thermal tangent in Newton.

No EOS values change. The old finite step can straddle a PCHIP knot, where
the derivative of Cp is one-sided. Preserve the opacity's explicit centered
response, but differentiate bulk rho, Cp, Q and adiabatic gradient exactly.
"""
from contextlib import contextmanager,ExitStack
from unittest.mock import patch
import numpy as np
from wd_spectra import adaptive_structure as adaptive
from wd_spectra.constants import STEFAN_BOLTZMANN
from wd_spectra.nonlinear import NonlinearEvaluation
from .convective_consistency_experiment import MaterialCoefficients
from .dense_helium_materials import DenseExactMaterials


def replace_material_tangent(evaluation,responses):
    p=dict(evaluation.payload);tr=p['convection_transport']
    coefficients=tuple(tr[k] for k in ('adiabatic_gradient','ml2_radiative_loss','ml2_flux_coefficient'))
    difference=tuple(new-old for new,old in zip(responses,p['ml2_coefficient_log_temperature_responses']))
    change=adaptive._ml2_flux_coefficient_response(p['temperature_gradient'],coefficients,difference)
    # Surface convection is a fixed boundary, regardless of its unused
    # dummy thermodynamic coefficients.
    change[0]=0.
    target=STEFAN_BOLTZMANN*p['atmosphere'].effective_temperature**4
    jacobian=evaluation.jacobian+change@p['log_temperature_from_state']/target
    p['ml2_coefficient_log_temperature_responses']=responses
    p['cell_energy_log_temperature_jacobian']=p['cell_energy_log_temperature_jacobian']+np.diff(change,axis=0)
    p['cell_energy_scale_log_temperature_jacobian']=p['cell_energy_scale_log_temperature_jacobian']+change[:-1]+change[1:]
    return NonlinearEvaluation(evaluation.residual,jacobian,p)


@contextmanager
def exact_dense_tangent_experiment(bulk):
    # This scope must be entered by the runner's late transfer scope, AFTER
    # it has installed its residual/proposal wrappers. We wrap the raw
    # physical evaluator before any invertible row transformations.
    original_solve=adaptive.solve_adaptive_lte_structure
    original_newton=adaptive.solve_trust_region_newton
    context={}
    def capture(seed,wave,**options):
        context['materials']=MaterialCoefficients(options['with_temperature'],options['thermodynamics'],
            options['rosseland_opacity'],options['mixing_length_alpha'])
        options['compute_local_energy_response']=True
        options['metadata']={**options.get('metadata',{}),'experimental_exact_dense_convection_tangent':True}
        return original_solve(seed,wave,**options)
    def solve(initial,evaluate,**settings):
        def exact_evaluate(state,need_jacobian):
            ev=evaluate(state,need_jacobian)
            if not need_jacobian or not ev.payload['energy_balance_is_physical_flux']:
                return ev
            if 'diagnostic_fixed_cell_scale' in ev.payload:
                raise ValueError('exact material tangent must precede transformed residual rows')
            p=ev.payload
            model=DenseExactMaterials(context['materials'],np.log(p['atmosphere'].temperature),.04,bulk=bulk)
            values,responses=model(np.zeros(p['atmosphere'].n_depth))
            for actual,key in zip(values,('adiabatic_gradient','ml2_radiative_loss','ml2_flux_coefficient')):
                np.testing.assert_allclose(actual,p['convection_transport'][key],rtol=2e-12,atol=0)
            result=replace_material_tangent(ev,responses)
            # Reuse this same exact material model for an explicitly
            # requested nonlinear-convection pseudo-time proposal.
            result.payload['diagnostic_dense_exact_material_model']=model
            return result
        return original_newton(initial,exact_evaluate,**settings)
    with ExitStack() as stack:
        stack.enter_context(patch.object(adaptive,'solve_adaptive_lte_structure',capture))
        stack.enter_context(patch.object(adaptive,'solve_trust_region_newton',solve))
        yield
