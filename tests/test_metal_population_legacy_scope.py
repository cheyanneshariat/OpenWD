"""Historical population recovery must be scoped and keep release profiles."""
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'research'))
from metal_population_control_legacy import previous_metal_opacity_populations
from wd_spectra import d6, metals
from test_topbase_population_consistency import _fixture, _section


@pytest.mark.parametrize('exceptional_exit', [False, True])
def test_historical_scope_restores_aliases_and_preserves_occupation_and_rap(exceptional_exit):
    old = (metals.metal_line_mass_absorption_coefficient,
           d6.metal_line_mass_absorption_coefficient,
           d6.topbase_lte_level_population,
           metals.metal_neutral_hard_sphere_occupation_probability,
           d6.read_tlusty_rap_lte_photoionization)
    class ExpectedExit(Exception):
        pass
    try:
        with previous_metal_opacity_populations():
            assert metals.metal_line_mass_absorption_coefficient is not old[0]
            assert d6.metal_line_mass_absorption_coefficient is metals.metal_line_mass_absorption_coefficient
            assert d6.topbase_lte_level_population is not old[2]
            assert metals.metal_neutral_hard_sphere_occupation_probability is old[3]
            assert d6.read_tlusty_rap_lte_photoionization is old[4]
            if exceptional_exit:
                raise ExpectedExit
    except ExpectedExit:
        assert exceptional_exit
    assert old == (metals.metal_line_mass_absorption_coefficient,
                   d6.metal_line_mass_absorption_coefficient,
                   d6.topbase_lte_level_population,
                   metals.metal_neutral_hard_sphere_occupation_probability,
                   d6.read_tlusty_rap_lte_photoionization)


def test_historical_topbase_recovers_the_ideal_lower_weight_in_nonideal_state():
    # A highly excited term makes the missing occupation factor visible;
    # the current helper must recover conservation after leaving the scope.
    ground = metals.AtomicLevel(8, 0., 1., '3s2.(1S<0>)')
    excited = metals.AtomicLevel(19, 7. * metals.EV_TO_WAVENUMBER, 3., '7p.(1Po<1>)')
    atmosphere, state, atom = _fixture([ground, excited])
    section = _section(atom.ions[('Mg', 0)], excited)
    current = d6.topbase_lte_level_population(section, state, atmosphere.temperature)
    with previous_metal_opacity_populations():
        previous = d6.topbase_lte_level_population(section, state, atmosphere.temperature)
        ideal_state = replace(state, metal_level_dissolution=False)
        expected = d6.topbase_lte_level_population(section, ideal_state, atmosphere.temperature)
    np.testing.assert_array_equal(previous, expected)
    assert np.all(current < previous)
    np.testing.assert_array_equal(current, d6.topbase_lte_level_population(section, state, atmosphere.temperature))
