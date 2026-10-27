import numpy as np
import pytest

from wd_spectra.metals import (
    _O_I_EXCITED_CORE_3D_TRANSITIONS,
    augment_oxygen_i_excited_core_3d_transitions,
    read_stout_atomic_database,
)
from wd_spectra.models import ModelData


@pytest.fixture(scope="module")
def oxygen():
    return read_stout_atomic_database(
        ModelData.default().stout, elements=("O",), maximum_charge=2
    )


def _added(original, augmented):
    old = {(t.lower_index, t.upper_index) for t in original.ions[("O", 0)].transitions}
    return [t for t in augmented.ions[("O", 0)].transitions
            if (t.lower_index, t.upper_index) not in old]


def test_excited_core_3d_multiplets_are_added_once(oxygen):
    augmented = augment_oxygen_i_excited_core_3d_transitions(oxygen)
    added = _added(oxygen, augmented)
    # Stout lists the 4d' and 5s' 3Do levels but no transitions to them.
    assert len(added) == len(_O_I_EXCITED_CORE_3D_TRANSITIONS) == 24
    assert augment_oxygen_i_excited_core_3d_transitions(augmented) is augmented
    wavelengths = sorted(t.wavelength_vacuum_angstrom for t in added)
    assert wavelengths[0] == pytest.approx(6094.53, abs=0.01)
    assert wavelengths[-1] == pytest.approx(6554.20, abs=0.01)
    assert all(t.radiative_damping_rate_s is None for t in added)
    # Existing Stout lines (for example 3p' 3D3 -> 4d' 3Fo4 at 6107.96 A) are kept.
    assert len(augmented.ions[("O", 0)].transitions) - len(added) == len(
        oxygen.ions[("O", 0)].transitions
    )


def test_ls_fine_structure_obeys_the_lower_level_sum_rule(oxygen):
    """For each lower J, sum_J' S(J, J') = (2J + 1) S_multiplet / ((2S+1)(2L+1))."""
    augmented = augment_oxygen_i_excited_core_3d_transitions(oxygen)
    ion = augmented.ions[("O", 0)]
    levels = {level.index: level for level in ion.levels}
    sums: dict[int, float] = {}
    for t in _added(oxygen, augmented):
        lower, upper = levels[t.lower_index], levels[t.upper_index]
        if ".(2Do).3p.(3D<" not in lower.label or ".(2Do).4d.(3Do<" not in upper.label:
            continue
        # f * lambda is proportional to S / g_lower.
        sums[t.lower_index] = sums.get(t.lower_index, 0.0) + (
            t.absorption_oscillator_strength * t.wavelength_vacuum_angstrom
        )
    assert len(sums) == 3
    values = np.array(list(sums.values()))
    np.testing.assert_allclose(values, values.mean(), rtol=1e-3)
