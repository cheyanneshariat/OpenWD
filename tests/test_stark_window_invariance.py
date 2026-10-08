"""Quasi-static Stark line opacity must not depend on the requested wavelengths."""

import numpy as np
import pytest

from wd_spectra.d6 import (
    atmosphere_with_bulk_metal_state,
    bulk_metal_lte_state,
    gray_d6_atmosphere,
)
from wd_spectra.metals import metal_line_mass_absorption_coefficient
from wd_spectra.models.common import ModelData
from wd_spectra.models.d6 import _atomic_inputs

ABUNDANCE = {"O": 0.0, "Ne": 0.3, "Mg": -0.6, "Na": -1.1}


@pytest.fixture(scope="module")
def plasma():
    database = _atomic_inputs(ModelData.default(), tuple(ABUNDANCE))[0]
    atmosphere = gray_d6_atmosphere(
        12_500.0, 5.75, database, ABUNDANCE, reference_element="O", n_depth=8
    )
    state = bulk_metal_lte_state(atmosphere, database, ABUNDANCE, reference_element="O")
    return database, atmosphere_with_bulk_metal_state(atmosphere, state), state


def _mg_ii_key(database, upper_label):
    ion = database.ions[("Mg", 1)]
    levels = {level.index: level.label for level in ion.levels}
    for transition in ion.transitions:
        if levels[transition.upper_index] == upper_label and "4f" in levels[transition.lower_index]:
            return ("Mg", 1, transition.lower_index, transition.upper_index)
    raise LookupError(upper_label)


@pytest.mark.parametrize("profile", ["manifold", "two-level"])
@pytest.mark.parametrize("upper", ["2p6.7g.(2G<9/2>)", "2p6.10g.(2G<9/2>)"])
def test_stark_opacity_is_independent_of_the_requested_window(plasma, profile, upper):
    database, atmosphere, state = plasma
    key = _mg_ii_key(database, upper)
    broad = np.arange(3600.0, 6000.0, 0.05)
    center = 1.0e8 / (
        next(l.energy_wavenumber for l in database.ions[("Mg", 1)].levels if l.index == key[3])
        - next(l.energy_wavenumber for l in database.ions[("Mg", 1)].levels if l.index == key[2])
    )
    red_wing = broad[(broad > center + 1.0) & (broad < center + 40.0)]

    def opacity(wavelength):
        return metal_line_mass_absorption_coefficient(
            atmosphere, wavelength, database, state,
            minimum_oscillator_strength=1.0e-8, maximum_lines=None,
            include_classical_electron_stark=True,
            include_linear_stark_quasistatic=True, linear_stark_profile=profile,
            transition_keys=(key,),
        )

    full = opacity(broad)
    partial = opacity(red_wing)
    common = np.isin(broad, red_wing)
    assert np.max(full[common]) > 0.0
    np.testing.assert_allclose(partial, full[common], rtol=1.0e-9, atol=0.0)
