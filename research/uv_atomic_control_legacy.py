"""Scoped legacy physics for historical controls; never a production option."""
from contextlib import contextmanager, ExitStack
from dataclasses import replace
import sys
from unittest.mock import patch

import wd_spectra.metals as metals


@contextmanager
def legacy_uv_atomic_controls():
    """Restore pre-Fe-supplement/pre-trace-H-line-width fixed spectra.

    Keep the metal populations, EOS and charged microfield moment unchanged.
    Only the state passed into metal line opacity loses trace_hydrogen_state
    for a He reference host, reproducing the former omission of neutral H.
    Patch every already-loaded binding by identity, including imported aliases;
    the metals module binding covers imports that occur inside the scope.
    """
    augment = metals.augment_iron_ii_kurucz_uv_transitions
    opacity = metals.metal_line_mass_absorption_coefficient

    def without_fe_supplement(database):
        return database

    def without_trace_h_width(atmosphere, wavelength_angstrom, atomic_database,
                              metal_state, *args, **kwargs):
        if (metal_state.reference_species == "He"
                and metal_state.trace_hydrogen_state is not None):
            metal_state = replace(metal_state, trace_hydrogen_state=None)
        return opacity(atmosphere, wavelength_angstrom, atomic_database,
                       metal_state, *args, **kwargs)

    bindings = []
    for module in tuple(sys.modules.values()):
        if module is None or not getattr(module, "__name__", "").startswith("wd_spectra"):
            continue
        for name, value in tuple(vars(module).items()):
            if value is augment:
                bindings.append((module, name, without_fe_supplement))
            elif value is opacity:
                bindings.append((module, name, without_trace_h_width))
    try:
        with ExitStack() as stack:
            for module, name, replacement in bindings:
                stack.enter_context(patch.object(module, name, replacement))
            yield
    finally:
        # Imports inside the scope can create aliases that ExitStack did not
        # see at entry. Restore only this scope's wrappers by identity, after
        # restoring the original bindings (also preserving nested scopes).
        for module in tuple(sys.modules.values()):
            if module is None or not getattr(module, "__name__", "").startswith("wd_spectra"):
                continue
            for name, value in tuple(vars(module).items()):
                if value is without_fe_supplement:
                    setattr(module, name, augment)
                elif value is without_trace_h_width:
                    setattr(module, name, opacity)
