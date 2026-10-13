"""Isolate neutral-H damping with fixed metal populations and electrons."""

from dataclasses import replace
from types import MappingProxyType

import numpy as np
import pytest

from wd_spectra import gray_helium_atmosphere
from wd_spectra.eos import HydrogenLTEState
from wd_spectra.metals import (
    AtomicDatabase,
    AtomicIon,
    AtomicLevel,
    AtomicTransition,
    MetalLTEState,
    metal_line_mass_absorption_coefficient,
)


@pytest.fixture
def line_case():
    atmosphere = gray_helium_atmosphere(10000.0, 8.0, n_depth=3)
    shape = atmosphere.temperature.shape
    atmosphere = replace(
        atmosphere,
        temperature=np.full(shape, 10000.0),
        helium_lte_state=replace(
            atmosphere.helium_lte_state,
            neutral_he_density=np.full(shape, 1.0e17),
        ),
    )
    # A single E1 line avoids changes in line selection or partition sums.
    line = AtomicTransition(
        1, 2, 1.0e8, "E1", 2000.0, 0.1,
        electron_stark_rate_coefficient_cm3_s=2.0e-6,
        neutral_h_vdw_rate_coefficient_cm3_s=1.0e-8,
    )
    ion = AtomicIon(
        "C", 1, 12.011, 24.383,
        (AtomicLevel(1, 0.0, 2.0, "lower"),
         AtomicLevel(2, 50000.0, 2.0, "upper")),
        (line,),
    )
    neutral_ion = AtomicIon(
        "C", 0, 12.011, 11.2603,
        (AtomicLevel(1, 0.0, 1.0, "neutral ground"),), (),
    )
    database = AtomicDatabase(MappingProxyType({
        ("C", 0): neutral_ion, ("C", 1): ion,
    }))
    carbon = np.full(shape, 1.0e8)
    state = MetalLTEState(
        reference_species="He",
        log_number_abundance={"C": -9.0},
        element_number_density={"C": carbon},
        ion_number_density={"C": np.stack((np.zeros(shape), carbon))},
        partition_function={
            ("C", 0): np.ones(shape), ("C", 1): np.full(shape, 2.0),
        },
        electron_density=np.full(shape, 1.0e13),
        metal_electron_density=carbon,
    )
    return atmosphere, database, state


def trace_hydrogen(shape, neutral_density, proton_density=0.0):
    neutral = np.full(shape, neutral_density)
    protons = np.full(shape, proton_density)
    nuclei = neutral + protons
    return HydrogenLTEState(
        mass_density=1.6735575e-24 * nuclei,
        hydrogen_nuclei_density=nuclei,
        neutral_h_density=neutral,
        proton_density=protons,
        electron_density=protons,
        ionization_fraction=np.divide(
            protons, nuclei, out=np.zeros(shape), where=nuclei > 0.0,
        ),
    )


def opacity(atmosphere, database, state):
    return metal_line_mass_absorption_coefficient(
        atmosphere, np.array([1999.8, 2000.0, 2000.2]), database, state,
        maximum_lines=None, transition_keys=(("C", 1, 1, 2),),
    )


def test_zero_neutral_trace_h_preserves_pure_he_opacity(line_case):
    atmosphere, database, state = line_case
    baseline = opacity(atmosphere, database, state)
    # A large ionized-H density must not broaden a neutral collision profile.
    ionized = replace(
        state, trace_hydrogen_state=trace_hydrogen(
            atmosphere.temperature.shape, 0.0, 2.0e17,
        ),
    )
    np.testing.assert_array_equal(opacity(atmosphere, database, ionized), baseline)


def test_trace_h_broadens_wings_without_changing_metal_populations(line_case):
    atmosphere, database, state = line_case
    hydrogen = trace_hydrogen(atmosphere.temperature.shape, 2.0e17)
    trace = replace(state, trace_hydrogen_state=hydrogen)
    baseline = opacity(atmosphere, database, state)
    broadened = opacity(atmosphere, database, trace)
    assert np.all(broadened[[0, 2]] > baseline[[0, 2]])
    assert np.all(broadened[1] < baseline[1])
    # The same neutral atoms have the same coefficient in either host path.
    hydrogen_host = replace(state, reference_species="H")
    np.testing.assert_array_equal(
        broadened,
        opacity(replace(atmosphere, neutral_h_density=hydrogen.neutral_h_density),
                database, hydrogen_host),
    )
    # Stale atmospheric H fields do not replace the metal-state charge closure.
    np.testing.assert_array_equal(
        broadened,
        opacity(replace(atmosphere, neutral_h_density=np.full(
            atmosphere.temperature.shape, 9.0e19,
        )), database, trace),
    )


def test_unified_he_profile_retains_neutral_h_impact_damping(line_case):
    atmosphere, database, state = line_case
    calls = []

    class CapturingProfile:
        window_angstrom = 5.0

        def profile(self, detuning, temperature, helium_density, gamma_other):
            calls.append(np.array(gamma_other, copy=True))
            return np.zeros((detuning.size, temperature.size))

    ion = database.ions[("C", 1)]
    line = replace(ion.transitions[0], helium_unified_profile=CapturingProfile())
    unified = AtomicDatabase(MappingProxyType({
        **database.ions, ("C", 1): replace(ion, transitions=(line,)),
    }))
    opacity(atmosphere, unified, state)
    neutral = trace_hydrogen(atmosphere.temperature.shape, 2.0e17)
    opacity(atmosphere, unified, replace(state, trace_hydrogen_state=neutral))
    opacity(atmosphere, unified, replace(state, trace_hydrogen_state=
        trace_hydrogen(atmosphere.temperature.shape, 0.0, 2.0e17)))
    natural_and_electron = 1.0e8 + 2.0e-6 * state.electron_density
    np.testing.assert_allclose(calls[0], natural_and_electron)
    np.testing.assert_allclose(calls[1], natural_and_electron + 2.0e9)
    np.testing.assert_array_equal(calls[2], calls[0])
