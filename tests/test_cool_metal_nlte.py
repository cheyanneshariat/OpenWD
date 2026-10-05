from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

from wd_spectra import gray_helium_atmosphere
from wd_spectra.cool_metal_nlte import (
    ca_ii_resonance_scattering_probabilities,
)
from wd_spectra.metals import (
    ATOMIC_MASS_U,
    AtomicDatabase,
    AtomicIon,
    AtomicLevel,
    AtomicTransition,
)


def _calcium_database() -> AtomicDatabase:
    levels = (
        AtomicLevel(1, 0.0, 2.0, "4s 2S1/2"),
        AtomicLevel(2, 13_650.0, 4.0, "3d 2D3/2"),
        AtomicLevel(3, 13_711.0, 6.0, "3d 2D5/2"),
        AtomicLevel(4, 25_192.0, 2.0, "4p 2P1/2"),
        AtomicLevel(5, 25_414.0, 4.0, "4p 2P3/2"),
    )
    transitions = (
        AtomicTransition(1, 4, 100.0, "E1", 3969.6, 0.3),
        AtomicTransition(2, 4, 20.0, "E1", 8664.0, 0.02),
        AtomicTransition(1, 5, 200.0, "E1", 3934.8, 0.6),
        AtomicTransition(2, 5, 10.0, "E1", 8500.0, 0.01),
        AtomicTransition(3, 5, 30.0, "E1", 8544.0, 0.03),
    )
    ion = AtomicIon(
        "Ca", 1, ATOMIC_MASS_U["Ca"], 11.87, levels, transitions
    )
    return AtomicDatabase(MappingProxyType({("Ca", 1): ion}))


def _write_scups(path: Path) -> None:
    path.write_text(
        """1 4 2.296e-1 0.0 1.0 2 2 1.0
0.0 1.0
2.0 2.0
1 5 2.316e-1 0.0 1.0 2 2 1.0
0.0 1.0
4.0 4.0
-1
""",
        encoding="ascii",
    )


def test_ca_ii_scattering_includes_radiative_branching_and_electron_destruction(
    tmp_path: Path,
):
    scups = tmp_path / "ca_2.scups"
    _write_scups(scups)
    atmosphere = gray_helium_atmosphere(10_000.0, 8.0, n_depth=8)
    collisionless = replace(
        atmosphere, electron_density=np.zeros(atmosphere.n_depth)
    )
    database = _calcium_database()

    low_density = ca_ii_resonance_scattering_probabilities(
        collisionless, database, scups
    )
    assert np.all(low_density[(1, 4)].probability == pytest.approx(100.0 / 120.0))
    assert np.all(low_density[(1, 5)].probability == pytest.approx(200.0 / 240.0))

    high_density = ca_ii_resonance_scattering_probabilities(
        replace(
            atmosphere,
            electron_density=np.full(atmosphere.n_depth, 1.0e16),
        ),
        database,
        scups,
    )
    assert np.all(
        high_density[(1, 4)].probability < low_density[(1, 4)].probability
    )
    assert np.all(
        high_density[(1, 5)].probability < low_density[(1, 5)].probability
    )


def test_ca_ii_fine_structure_transfer_is_not_counted_as_destruction(tmp_path: Path):
    scups = tmp_path / "ca_2.scups"
    scups.write_text(
        """1 4 2.296e-1 0.0 1.0 2 2 1.0
0.0 1.0
2.0 2.0
4 5 1.0e-2 0.0 1.0 2 2 1.0
0.0 1.0
5.6 5.6
-1
""",
        encoding="ascii",
    )
    atmosphere = replace(
        gray_helium_atmosphere(10_000.0, 8.0, n_depth=8),
        electron_density=np.full(8, 1.0e16),
    )
    records = ca_ii_resonance_scattering_probabilities(
        atmosphere, _calcium_database(), scups
    )
    for key, record in records.items():
        assert record.fine_structure_partner in records
        assert record.fine_structure_partner != key
        assert np.all(record.fine_structure_transfer > 0.0)
        np.testing.assert_allclose(
            record.probability + record.fine_structure_transfer
            + record.destruction_probability,
            1.0,
        )


def _isothermal_crd_inputs(probability: float):
    from wd_spectra.cool_metal_nlte import ResonanceScatteringProbability

    atmosphere = gray_helium_atmosphere(8000.0, 8.0, n_depth=40)
    atmosphere = replace(atmosphere, temperature=np.full(atmosphere.n_depth, 8000.0))
    wavelength = np.linspace(3800.0, 4100.0, 1501)
    centers = {(1, 4): 3969.6, (1, 5): 3934.8}
    background = np.full((wavelength.size, atmosphere.n_depth), 0.01)

    def line_extinction(grid):
        return {
            key: 1.0e3 * 0.5 / np.pi / ((grid[:, None] - center) ** 2 + 0.25)
            * np.ones(atmosphere.n_depth)[None, :]
            for key, center in centers.items()
        }

    probabilities = {
        key: ResonanceScatteringProbability(
            probability=np.full(atmosphere.n_depth, probability),
            resonant_einstein_a=1.0,
            total_radiative_rate=1.0,
            source="test",
            fine_structure_partner=other,
            fine_structure_transfer=np.zeros(atmosphere.n_depth),
        )
        for key, other in (((1, 4), (1, 5)), ((1, 5), (1, 4)))
    }
    absorption = background + sum(line_extinction(wavelength).values())
    return atmosphere, wavelength, absorption, line_extinction, probabilities


def test_ca_ii_crd_source_recovers_lte_without_scattering():
    from wd_spectra.cool_metal_nlte import (
        _planck_frequency,
        ca_ii_crd_resonance_source_functions,
    )

    atmosphere, wavelength, absorption, lines, probabilities = _isothermal_crd_inputs(0.0)
    sources = ca_ii_crd_resonance_source_functions(
        atmosphere, wavelength, absorption, np.zeros_like(absorption),
        lines, probabilities, n_angle=3,
    )
    planck = _planck_frequency(np.asarray([3934.8]), atmosphere.temperature)[0]
    np.testing.assert_allclose(sources[(1, 5)], planck, rtol=1e-12)


def test_ca_ii_crd_source_follows_sqrt_epsilon_surface_law():
    from wd_spectra.cool_metal_nlte import (
        _planck_frequency,
        ca_ii_crd_resonance_source_functions,
    )

    epsilon = 1.0e-2
    atmosphere, wavelength, absorption, lines, probabilities = (
        _isothermal_crd_inputs(1.0 - epsilon)
    )
    sources = ca_ii_crd_resonance_source_functions(
        atmosphere, wavelength, absorption, np.zeros_like(absorption),
        lines, probabilities, n_angle=4,
    )
    planck = _planck_frequency(np.asarray([3934.8]), atmosphere.temperature)[0]
    ratio = sources[(1, 5)] / planck
    # Isothermal CRD: S(0)/B is of order sqrt(epsilon) and S -> B at depth.
    assert 0.3 * np.sqrt(epsilon) < ratio[0] < 3.0 * np.sqrt(epsilon)
    assert ratio[-1] == pytest.approx(1.0, rel=0.02)
    assert np.all(np.diff(ratio) >= -1e-12)
