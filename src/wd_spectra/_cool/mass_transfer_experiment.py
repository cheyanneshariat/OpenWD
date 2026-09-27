"""Research-run adapter to the explicit, per-call mass-transfer option.

No radiation-field globals or last-evaluated mass/opacity arrays are patched.
Use separate processes for research scopes that still replace the entry point.
"""
from contextlib import contextmanager
from functools import wraps
from unittest.mock import patch


@contextmanager
def mass_transfer_experiment():
    from wd_spectra import adaptive_structure as adaptive
    original = adaptive.solve_adaptive_lte_structure

    @wraps(original)
    def solve(*args, **options):
        if options.get("transfer_discretization", "column-mass") != "column-mass":
            raise ValueError("conflicting transfer discretization in mass experiment")
        options["transfer_discretization"] = "column-mass"
        return original(*args, **options)

    with patch.object(adaptive, "solve_adaptive_lte_structure", solve):
        yield
