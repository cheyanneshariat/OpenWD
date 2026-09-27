"""Explicit research scope for cancellation-resistant transfer arithmetic.

The tested implementation lives in wd_spectra; this wrapper never selects it
for a public production run implicitly.
"""
from contextlib import contextmanager, ExitStack
from unittest.mock import patch
from wd_spectra._stable_feautrier import (
    cancellation_safe_field as stable_field,
    cancellation_safe_scalar_field as stable_scalar_field,
    cancellation_safe_response as stable_response,
)


@contextmanager
def stable_transfer_experiment():
    """Scope experimental field, independent scalar check, and tangent together."""
    import wd_spectra.adaptive_structure as adaptive
    import wd_spectra.radiative_transfer as transfer
    with ExitStack() as stack:
        for module in (adaptive, transfer):
            stack.enter_context(patch.object(module, 'coherent_scattering_feautrier_field', stable_field))
            stack.enter_context(patch.object(module, 'feautrier_radiation_field', stable_scalar_field))
            stack.enter_context(patch.object(module, 'integrated_coherent_scattering_feautrier_state_response', stable_response))
        yield
