"""Scoped recovery of the line physics before the excited-parent series fix.

This is a research/test helper, never a production configuration option.  Inside
previous_excited_parent_series() the frozen J1637 controls approved before
2026-10-25 (warm_v1 ... warm_v4) reproduce exactly:

- classical electron Stark widths measure every upper level from the first
  ionization limit, so levels above it receive no width;
- only ground-parent series qualify for the linear-Stark n-manifold, and the
  quasi-static bindings use the first limit;
- the O I 2p3(2D)3p -> 4d, 5s 3Do supplement is absent from the D6 atom.
"""
from contextlib import contextmanager, ExitStack
from unittest.mock import patch

import wd_spectra.metals as metals
import wd_spectra.models.d6 as public_d6


def _clear_stark_caches():
    metals._LINEAR_STARK_LEVEL_CACHE.clear()
    metals._STARK_MANIFOLD_CACHE.clear()


@contextmanager
def previous_excited_parent_series():
    classical = metals.classical_electron_stark_rate_coefficient
    series_parent = metals._series_parent_excitation_ev
    rydberg_level = metals._linear_stark_rydberg_level

    def previous_classical(ion, transition, parent_ion=None):
        return classical(ion, transition)

    def previous_rydberg_level(atomic_database, ion, level):
        # Excited-parent (or unidentified-parent) series were excluded.
        if series_parent(atomic_database, ion, level) != 0.0:
            return None
        return rydberg_level(atomic_database, ion, level)

    _clear_stark_caches()
    try:
        with ExitStack() as stack:
            stack.enter_context(patch.object(
                metals, "classical_electron_stark_rate_coefficient", previous_classical))
            stack.enter_context(patch.object(
                metals, "_series_parent_excitation_ev", lambda *args: 0.0))
            stack.enter_context(patch.object(
                metals, "_linear_stark_rydberg_level", previous_rydberg_level))
            stack.enter_context(patch.object(
                public_d6, "augment_oxygen_i_excited_core_3d_transitions", lambda database: database))
            yield
    finally:
        _clear_stark_caches()
