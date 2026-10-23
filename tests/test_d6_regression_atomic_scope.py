"""Actual cached D6 atoms must not cross historical regression boundaries."""

import pytest

from d6_regression_support import atomic_input_scope, atomic_inputs


@pytest.mark.parametrize("policy,old_rap", [
    (dict(legacy=True), True),
    (dict(previous=True), True),
    (dict(legacy_rap=True), True),
    ({}, False),
])
@pytest.mark.parametrize("exceptional_exit", [False, True])
def test_rap_identity_and_cached_atoms_restore_after_historical_scopes(policy, old_rap, exceptional_exit):
    from wd_spectra import d6

    class HistoricalExit(Exception):
        pass

    reader = d6.read_tlusty_rap_lte_photoionization

    def first_fe():
        return next(section for section in atomic_inputs()[2].sections
                    if section.element == "Fe" and section.charge == 1)

    try:
        with atomic_input_scope():
            current = first_fe()
            assert current.absolute_photon_grid
            assert current.excitation_energy_ev == pytest.approx(
                416.299 / d6.EV_TO_WAVENUMBER, abs=1e-14)
        try:
            with atomic_input_scope(**policy):
                assert atomic_inputs.cache_info().currsize == 0
                scoped = first_fe()
                assert scoped.absolute_photon_grid is (not old_rap)
                if old_rap:
                    assert scoped.excitation_energy_ev > current.excitation_energy_ev
                else:
                    assert scoped.excitation_energy_ev == current.excitation_energy_ev
                if exceptional_exit:
                    raise HistoricalExit
        except HistoricalExit:
            assert exceptional_exit
        assert d6.read_tlusty_rap_lte_photoionization is reader
        assert atomic_inputs.cache_info().currsize == 0
        with atomic_input_scope():
            restored = first_fe()
            assert restored.absolute_photon_grid
            assert restored.excitation_energy_ev == current.excitation_energy_ev
            assert restored.cross_section_cm2 is not scoped.cross_section_cm2
    finally:
        atomic_inputs.cache_clear()


@pytest.mark.parametrize("exceptional_exit", [False, True])
def test_scoped_historical_atoms_restore_default_fe_even_after_exception(exceptional_exit):
    class HistoricalExit(Exception):
        pass

    try:
        # Populate the default cache before entering the historical scope.
        default = atomic_inputs()[0].ions[("Fe", 1)]
        default_keys = {(line.lower_index, line.upper_index) for line in default.transitions}
        assert len(default.transitions) == 6724 + 4613
        assert atomic_inputs.cache_info().currsize == 1
        try:
            with atomic_input_scope(legacy=True):
                assert atomic_inputs.cache_info().currsize == 0
                historical = atomic_inputs()[0].ions[("Fe", 1)]
                old_keys = {
                    (line.lower_index, line.upper_index) for line in historical.transitions
                }
                assert len(historical.transitions) == 6724
                assert old_keys < default_keys
                assert len(default_keys - old_keys) == 4613
                assert len(default.levels) == len(historical.levels) + 1
                assert "Kurucz Fe II" not in historical.source
                if exceptional_exit:
                    raise HistoricalExit
        except HistoricalExit:
            assert exceptional_exit
        assert atomic_inputs.cache_info().currsize == 0
        with atomic_input_scope():
            assert atomic_inputs.cache_info().currsize == 0
            restored = atomic_inputs()[0].ions[("Fe", 1)]
            assert restored is not default and restored is not historical
            assert restored.transitions == default.transitions
            assert restored.levels == default.levels
            assert restored.source == default.source
            assert atomic_inputs.cache_info().currsize == 1
        assert atomic_inputs.cache_info().currsize == 0
    finally:
        atomic_inputs.cache_clear()
