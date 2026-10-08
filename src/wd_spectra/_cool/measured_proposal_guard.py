"""Require a measured, non-trust-limited correction in strict research runs.

The generic nonlinear driver can stop after an accepted small step. A small
accepted step may be due to trust clipping or line search, rather than a small
computed correction. Keep that generic behavior unchanged, but do not use it
to certify these DAB experiments. This wrapper changes acceptance of a final
convergence claim, not the physical residual, Jacobian, or trial direction.
"""
import numpy as np
from wd_spectra.nonlinear import NonlinearProposal


def require_measured_proposal(settings):
    settings=dict(settings)
    build=settings['step_builder']
    physical_test=settings.get('convergence_test')
    measure=settings.get('step_measure')
    tolerance=settings.get('step_tolerance',2e-4)
    measured_small=False

    def proposal(state,evaluation,jacobian,radius):
        nonlocal measured_small
        measured_small=False
        direction=build(state,evaluation,jacobian,radius)
        limited=isinstance(direction,NonlinearProposal) and direction.limited
        step=direction.direction if isinstance(direction,NonlinearProposal) else direction
        size=(np.max(abs(step)) if measure is None
              else measure(state,state+step))
        # If the builder is bounded, a sub-tolerance region cannot establish
        # that the correction itself is small. Check before outer backtracking.
        model_solved=evaluation.payload.get('diagnostic_inner_proposal_root_solved',True)
        measured_small=bool(not limited and model_solved and radius>=tolerance and np.isfinite(size) and size<tolerance)
        return direction

    def converged(state,evaluation,accepted_size):
        return (measured_small and accepted_size<tolerance and
                (physical_test is None or physical_test(state,evaluation,accepted_size)))

    settings.update(step_builder=proposal,convergence_test=converged,
                    allow_initial_convergence=False)
    return settings
