"""Actual cached D6 atoms must not cross historical regression boundaries."""

import pytest

from d6_regression_support import atomic_input_scope, atomic_inputs


@pytest.mark.parametrize("exceptional_exit", [False, True])
def test_scoped_historical_atoms_restore_default_fe_even_after_exception(exceptional_exit):
    class HistoricalExit(Exception):
        pass

    try:
        # Populate the default cache before entering the historical scope.
        default = atomic_inputs()[0].ions[("Fe", 1)]
        default_keys = {(line.lower_index, line.upper_index) for line in default.transitions}
        assert len(default.transitions) == 6869
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
                assert len(default_keys - old_keys) == 145
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
