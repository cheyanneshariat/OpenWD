"""Match the molecular research structure's explicit transfer and Lyman policy.

This scope affects only fixed-atmosphere exploratory synthesis in a research
worker. It neither relaxes an atmosphere nor changes its convergence status.
"""
from contextlib import contextmanager
from dataclasses import replace
from unittest.mock import patch
from wd_spectra.models import stellar


@contextmanager
def matched_synthesis(runner,angles):
    compute=runner.compute_dab
    synthesize=stellar.synthesize_hydrogen_helium_spectrum
    def fixed(config,*args,**kwargs):
        if kwargs.get('relax_atmosphere',True):
            raise ValueError('research synthesis scope requires an explicitly fixed atmosphere')
        return compute(replace(config,lyman_profile_source='stark'),*args,**kwargs)
    def spectrum(atmosphere,*args,**kwargs):
        # Match the existing mixed-structure wrapper's He II line selection
        # too; changing transfer alone is not a matched-physics comparison.
        kwargs.update(transfer_discretization='column-mass',n_angle=angles,
                      include_helium_ii_lines=atmosphere.effective_temperature>=15000.)
        return synthesize(atmosphere,*args,**kwargs)
    with patch.object(runner,'compute_dab',fixed), \
            patch.object(stellar,'synthesize_hydrogen_helium_spectrum',spectrum):
        yield
