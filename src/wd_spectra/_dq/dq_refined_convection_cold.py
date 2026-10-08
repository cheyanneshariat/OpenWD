"""DQ-only refined material correction BETWEEN shared Newton iterations.

Follow Hubeny (2017) §§5.3–5.4's ordering: local temperature/material
correction, actual formal transfer, material update, fresh global correction.
Retain established convection-zone history, but never substitute that mask
for the physical stability law. Independently backtrack the correction with
both actual coupled AND physical merits; it cannot certify stationarity.
No saved-atmosphere initialization, EOS replacement or physics fallback.
"""
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch
import numpy as np
from . import dq_augmented_convection as coupled
from . import dq_augmented_current_norm as current
from .dq_augmented_probe_reuse import ProbeReuseSystem
from .dq_refined_convection import refined_correction, ConvectionHistory
from wd_spectra.nonlinear import NonlinearCorrection, RecoverableEvaluationError


class RefinedController:
    def __init__(self, history=None):
        self.history = ConvectionHistory() if history is None else history

    def proposal(self, system, state, ev):
        n = system.n
        p = ev.payload
        active = self.history.observe(system.pressure, p['temperature_gradient'],
                                     p['convection_transport']['adiabatic_gradient'])
        if not np.any(active):
            return None
        original = system.physical(state[:n], False)
        physical_merit = .5*np.mean(original.residual**2)
        try:
            x, info = refined_correction(state[:n], system.pressure,
                p['radiative_flux_interface'], system.refined_material, system.target,
                active=active, through_bottom=True)
        except RecoverableEvaluationError as error:
            print('DQ refined correction unavailable: '+str(error), flush=True)
            return None
        print(f'DQ refined independent correction: dlnT={np.max(abs(x-state[:n])):.6g}; '
              f'material calls={info["material_calls"]}; layers={info["corrected_layers"]}', flush=True)

        latest = {}
        def trial(factor):
            point = state[:n]+factor*(x-state[:n])
            physical = system.physical(point, False)
            latest['physical'] = physical
            # Recompute compatible auxiliary coordinates at EACH backtracked
            # temperature, never interpolate an incompatible trial flux.
            return system.state_from_temperature(point, physical)

        def acceptable(trial_ev):
            actual = latest['physical']
            np.testing.assert_array_equal(actual.payload['atmosphere'].temperature,
                                          trial_ev.payload['atmosphere'].temperature)
            merit = .5*np.mean(actual.residual**2)
            accepted = bool(merit < physical_merit)
            print(f'DQ refined physical acceptance: {physical_merit:.6g}->{merit:.6g}; '
                  f'accepted={accepted}', flush=True)
            return accepted
        return NonlinearCorrection(trial, acceptable)


@contextmanager
def refined_proposals(history=None, system_class=ProbeReuseSystem):
    original_solver = coupled.adaptive.solve_trust_region_newton
    controller = RefinedController(history)

    def solve(initial, evaluate, **settings):
        system = getattr(evaluate, '__self__', None)
        if isinstance(system, system_class):
            settings['iteration_correction'] = lambda state, ev: controller.proposal(system, state, ev)
            if getattr(system, 'native_trust_geometry', False):
                measure = getattr(system, 'trust_step_size', None)
                settings['trust_step_measure'] = (
                    measure if measure is not None
                    else lambda old, new: float(np.max(abs(new-old)))
                )
        return original_solver(initial, evaluate, **settings)

    with patch.object(coupled.adaptive, 'solve_trust_region_newton', solve):
        yield


def refined_material(wavelengths, system_class=ProbeReuseSystem):
    class RefinedProbeSystem(system_class):
        def __init__(self, first, evaluate, material):
            self.refined_material = material
            super().__init__(first, evaluate, material)

    class RefinedDQ(current.current_norm_material(wavelengths, system_class=RefinedProbeSystem)):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.refined_history = ConvectionHistory()
            from .provenance import digest
            self.experiment_metadata.update(refined_convection_proposal=True,
                refined_auxiliary_system=system_class.__name__,
                refined_convection_source_sha256=digest(Path(__file__).with_name('dq_refined_convection.py')),
                refined_controller_source_sha256=digest(__file__),
                refined_nonlinear_source_sha256=digest(Path(coupled.adaptive.__file__).with_name('nonlinear.py')),
                refined_proposal_acceptance='independent between-Newton update; actual coupled and physical merit; original gates')

        def solve(self, *args, **kwargs):
            with refined_proposals(self.refined_history, RefinedProbeSystem):
                return super().solve(*args, **kwargs)
    return RefinedDQ


