"""Recover the pre-2026-10-27 hydrogen neutral defaults for immutable controls.

Research/test scope only. Before 2026-10-27 the Hummer--Mihalas neutral
hydrogen hard-sphere radius scale defaulted to 1 (r_n = n^2 a_0), and the
public hydrogen self-broadening policy used Ali--Griem below 10,000 K and
Barklem at and above it. The current defaults are the Montreal radius
scale 0.5 and Ali--Griem at every temperature. Nothing else is restored:
explicitly requested radii and prescriptions are honored unchanged.
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
from inspect import Parameter, signature
import sys
from unittest.mock import patch

from wd_spectra import eos, magnetic_eos
from wd_spectra.models import stellar

PREVIOUS_HYDROGEN_RADIUS_SCALE = 1.0
PREVIOUS_ALI_GRIEM_TEMPERATURE_CUTOFF_K = 10_000.0

# Every default bound to eos.HM_NEUTRAL_HYDROGEN_RADIUS_SCALE when defined.
_RADIUS_DEFAULTS = (
    (eos.hydrogen_occupation_probability, "neutral_radius_scale"),
    (eos.hydrogen_level_distribution, "neutral_radius_scale"),
    (eos.hummer_mihalas_hydrogen_lte, "neutral_radius_scale"),
    (eos.helium_occupation_probability, "hydrogen_neutral_radius_scale"),
    (eos._helium_partition_functions, "hydrogen_neutral_radius_scale"),
    (eos.HydrogenLTEState.__init__, "neutral_radius_scale"),
    (magnetic_eos.magnetic_hydrogen_level_distribution, "neutral_radius_scale"),
    (magnetic_eos.magnetic_hummer_mihalas_hydrogen_lte, "neutral_radius_scale"),
)


def previous_self_broadening_prescription(effective_temperature, requested):
    """The pre-2026-10-27 public hydrogen neutral-broadening policy."""
    if requested is not None:
        return requested
    return (
        "ali-griem"
        if effective_temperature < PREVIOUS_ALI_GRIEM_TEMPERATURE_CUTOFF_K
        else "barklem"
    )


def _replace_default(function, name, value):
    """Set one parameter default in place; return a callable that restores it."""
    parameter = signature(function).parameters[name]
    if parameter.kind is Parameter.KEYWORD_ONLY:
        old = dict(function.__kwdefaults__)
        function.__kwdefaults__ = {**old, name: value}
        return lambda: setattr(function, "__kwdefaults__", old)
    positional = [
        p for p in signature(function).parameters.values()
        if p.kind in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)
    ]
    defaults = list(function.__defaults__)
    index = len(defaults) - len(positional) + positional.index(parameter)
    old = tuple(defaults)
    defaults[index] = value
    function.__defaults__ = tuple(defaults)
    return lambda: setattr(function, "__defaults__", old)


@contextmanager
def previous_hydrogen_neutral_defaults():
    """Restore radius scale 1 and the 10,000 K Ali--Griem/Barklem switch."""
    restore = []
    try:
        for function, name in _RADIUS_DEFAULTS:
            current = signature(function).parameters[name].default
            if current != eos.HM_NEUTRAL_HYDROGEN_RADIUS_SCALE:
                raise RuntimeError(f"{function.__qualname__}.{name} is not the shared default")
            restore.append(_replace_default(function, name, PREVIOUS_HYDROGEN_RADIUS_SCALE))
        with ExitStack() as stack:
            for module in tuple(sys.modules.values()):
                if module is None or not getattr(module, "__name__", "").startswith("wd_spectra"):
                    continue
                for attribute, value in tuple(vars(module).items()):
                    if attribute == "HM_NEUTRAL_HYDROGEN_RADIUS_SCALE":
                        stack.enter_context(patch.object(
                            module, attribute, PREVIOUS_HYDROGEN_RADIUS_SCALE))
                    elif value is stellar._da_self_broadening_prescription:
                        stack.enter_context(patch.object(
                            module, attribute, previous_self_broadening_prescription))
            yield
    finally:
        for undo in reversed(restore):
            undo()
