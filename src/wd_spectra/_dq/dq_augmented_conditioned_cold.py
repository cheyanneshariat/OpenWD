"""Keep the established cold ML2 conditioner before the enlarged equations.

The direct physical-equation cold start can leave the useful convective
branch before its auxiliary/material compatibility has recovered. Retain
the existing cold initializer, then replace only the costly nested physical
phase with the simultaneous solve. No external atmosphere or model
continuation, and no provisional gradient equation can certify convergence.
"""
from unittest.mock import patch
from . import base as dq
from .dq_augmented_material_scaling import scaled_material


def conditioned_material(wavelengths=None, *, system_class=None):
    class ConditionedCoupledDQ(scaled_material(wavelengths, system_class=system_class)):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            from .provenance import digest
            self.experiment_metadata.update(
                coupled_initialization='existing cold ML2 gradient conditioner, then physical coupled equations',
                coupled_conditioner_source_sha256=digest(__file__))
        def solve(self,*args,**kwargs):
            original=dq.solve_adaptive_lte_structure
            def cold_conditioner(seed,wave,**options):
                # The screened extension and physical-phase handoff already
                # supply their own same-calculation initial structures. Never
                # re-enter the approximate conditioner for those calls.
                cold=not options.get('initial_temperature_was_supplied',False)
                return original(seed,wave,**dict(options,
                    use_convective_gradient_preconditioner=cold))
            with patch.object(dq,'solve_adaptive_lte_structure',cold_conditioner):
                return super().solve(*args,**kwargs)
    return ConditionedCoupledDQ


