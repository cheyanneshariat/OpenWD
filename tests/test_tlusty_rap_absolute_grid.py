"""TLUSTY RAP photoionization: absolute excitation energies and photon grids.

From OpenWD PR #37 (cheyanneshariat, "Normalize metal populations and correct
RAP photoionization"): only the RAP reader/cross-section fix is included here;
PR #37's population-matching helpers are not, so their assertions are omitted.
"""
from dataclasses import replace
from types import MappingProxyType

import numpy as np
import pytest

from wd_spectra.constants import LIGHT_SPEED, PLANCK
from wd_spectra.d6 import (
    TOPbaseLevelPhotoionization,
    read_tlusty_rap_lte_photoionization,
    resonance_average_topbase_photoionization,
)
from wd_spectra.metals import (
    AtomicDatabase, EV_TO_ERG, EV_TO_WAVENUMBER, ground_term_levels, read_stout_atomic_ion,
)
from wd_spectra.models.common import ModelData


def test_bundled_rap_feii_excitation_is_already_above_lowest_j():
    """Actual RAP absolute energies must not receive a second LS offset."""
    data = ModelData.default()
    ion = read_stout_atomic_ion(data.stout, "Fe", 1)
    # Native indices deliberately differ from source-local model indices.
    ion = replace(ion, levels=tuple(replace(level, index=level.index+1000)
                                   for level in ion.levels), transitions=())
    atom = AtomicDatabase(MappingProxyType({("Fe", 1): ion}))
    path = data.tlusty_atoms / "fe2p_14+11lev.rap"
    loaded = read_tlusty_rap_lte_photoionization([(path, "Fe", 1)], atom)
    np.testing.assert_allclose(
        [section.excitation_energy_ev for section in loaded.sections[:3]],
        np.array([416.299, 2416.571, 8320.469]) / EV_TO_WAVENUMBER,
        rtol=0., atol=1e-14,
    )
    assert [section.statistical_weight for section in loaded.sections[:3]] == [30., 28., 20.]
    ground = loaded.sections[0]
    assert ground.threshold_energy_ev == pytest.approx(
        ion.ionization_energy_ev - 416.299 / EV_TO_WAVENUMBER, abs=1e-13,
    )

    # Keep every channel and the original absolute photon/cross-section grid.
    # A companion atom has a slightly older IP; the first positive sample is
    # not redefined as I-E here and no level-identity tolerance is relaxed.
    lines = path.read_text().splitlines()
    count = int(lines[0].split()[2])
    assert count == len(loaded.sections) == 25
    cursor = 1
    for section in loaded.sections:
        index, energy, weight, points = lines[cursor].split()
        cursor += 1
        samples = np.array([tuple(map(float, row.split()))
                            for row in lines[cursor:cursor+int(points)]])
        cursor += int(points)
        first = int(np.flatnonzero(samples[:, 1] > 0.)[0])
        start = max(0, first-1)
        assert section.level_index == int(index)
        assert section.statistical_weight == float(weight)
        assert section.excitation_energy_ev == float(energy) / EV_TO_WAVENUMBER
        np.testing.assert_array_equal(section.photon_energy_ev,
                                      PLANCK * samples[start:, 0] / EV_TO_ERG)
        np.testing.assert_array_equal(section.cross_section_cm2,
                                      samples[start:, 1] * 1e-18)
        assert section.absolute_photon_grid
        positive = section.cross_section_cm2 > 0.
        np.testing.assert_allclose(
            section.cross_section(section.photon_energy_ev[positive]),
            section.cross_section_cm2[positive], rtol=3e-13, atol=0.,
        )
        opening = section.photon_energy_ev[np.flatnonzero(positive)[0]]
        assert section.sampling_edge_energy_ev == opening
        assert section.cross_section(np.nextafter(section.photon_energy_ev[0], 0.)) == 0.
        first = int(np.flatnonzero(positive)[0])
        if first:
            midpoint = .5 * (section.photon_energy_ev[first-1] + opening)
            # The narrowest ramps make the represented midpoint differ
            # from an exact half by ~1e-10 of the interval. Evaluate the
            # reference at that actual floating-point coordinate.
            expected_midpoint = np.interp(midpoint,
                section.photon_energy_ev[first-1:first+1],
                section.cross_section_cm2[first-1:first+1])
            np.testing.assert_allclose(section.cross_section(midpoint),
                expected_midpoint, rtol=3e-13, atol=0.)
        # Nominal population bindings cannot clip or shift an aggregate's
        # absolute photon table, including below the mean lower binding.
        shifted = replace(section, threshold_energy_ev=100.,
                          source_threshold_energy_ev=101.)
        np.testing.assert_array_equal(
            shifted.cross_section(section.photon_energy_ev),
            section.cross_section(section.photon_energy_ev),
        )

    # Actual Fe II aggregate6 has positive support below its mean I-E.
    section = loaded.sections[5]
    photon = PLANCK * LIGHT_SPEED / (958.9365366146347e-8) / EV_TO_ERG
    assert photon < section.threshold_energy_ev
    positive = section.cross_section_cm2 > 0.
    expected = np.exp(np.interp(np.log(photon),
        np.log(section.photon_energy_ev[positive]),
        np.log(section.cross_section_cm2[positive])))
    np.testing.assert_allclose(section.cross_section(photon), expected,
                               rtol=3e-13, atol=0.)
    assert expected > 2e-18
    averaged = resonance_average_topbase_photoionization(loaded, .01)
    assert all(s.absolute_photon_grid for s in averaged.sections)


def test_absolute_table_interpolates_zero_pairs_without_inventing_coverage():
    section = TOPbaseLevelPhotoionization(
        "Fe", 1, 1, 3., 78., 100., 101.,
        np.array([2., 4., 6., 8., 10.]),
        np.array([1., 0., 0., 4., 2.]) * 1e-18,
        absolute_photon_grid=True,
    )
    np.testing.assert_allclose(section.cross_section([1., 2., 3., 5., 7., 8., 20.]),
        np.array([0., 1., .5, 0., 2., 4., .25]) * 1e-18,
        rtol=3e-13, atol=0.)
    assert section.structure_edge_energies_ev == (2.,)


def test_rap_feiii_primary_header_has_nonzero_absolute_ground_centroid(tmp_path):
    """Fe III is a source-format control, not an extra bundled opacity ion.

    Header and first samples: TLUSTY database/atom/fe3p_22+7lev.rap, linked
    from https://tlusty.oca.eu/tlusty/Tlusty2002/data-iron.html. The fixture
    excerpts one four-point section; the declared counts match that excerpt.
    """
    ion = read_stout_atomic_ion(ModelData.default().stout, "Fe", 2)
    atom = AtomicDatabase(MappingProxyType({("Fe", 2): ion}))
    rap = tmp_path / "feiii-ground-format-control.rap"
    rap.write_text(
        "26 3 1\n"
        "1 422.896 25.0 4\n"
        "9.99999996e11 0.0\n"
        "6.99692532e15 0.0\n"
        "6.99693230e15 3.942\n"
        "7.20841434e15 5.454\n",
        encoding="ascii",
    )
    members = ground_term_levels(ion)
    assert sum(level.statistical_weight for level in members) == 25.
    assert min(level.energy_wavenumber for level in members) == 0.
    assert max(level.energy_wavenumber for level in members) > 1000.
    section, = read_tlusty_rap_lte_photoionization([(rap, "Fe", 2)], atom).sections
    assert section.excitation_energy_ev == pytest.approx(422.896 / EV_TO_WAVENUMBER, abs=1e-14)
    assert section.threshold_energy_ev == pytest.approx(
        ion.ionization_energy_ev-422.896 / EV_TO_WAVENUMBER, abs=1e-13,
    )
    np.testing.assert_array_equal(section.cross_section_cm2,
                                  np.array([0., 3.942, 5.454]) * 1e-18)
