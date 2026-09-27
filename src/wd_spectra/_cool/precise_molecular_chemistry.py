"""Explicit accuracy experiment; no equilibrium equations or rates change."""
from contextlib import contextmanager
from functools import partial
from unittest.mock import patch
from wd_spectra import _mixed_molecules as mixture


@contextmanager
def precise_chemistry(tolerance):
    original=mixture.molecular_hydrogen_helium_lte
    with patch.object(mixture,'molecular_hydrogen_helium_lte',
                      partial(original,conservation_tolerance=tolerance)):
        yield
