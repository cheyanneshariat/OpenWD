from __future__ import annotations

from zipfile import ZipFile
from pathlib import Path

import numpy as np
from wd_spectra.models import ModelData

from wd_spectra import gray_helium_atmosphere, synthesize_helium_spectrum
from wd_spectra.multilevel_nlte import read_ccc_hydrogen_collision_data
from wd_spectra.helium_collisions import read_tlusty_helium_collision_data
from wd_spectra.helium_nlte import (
    helium_ii_excitation_collision_rate_coefficient,
    helium_ii_ionization_collision_rate_coefficient,
    helium_ii_photoionization_cross_section,
    helium_ii_shell_transition,
    remap_coupled_helium_state,
    solve_neutral_helium_statistical_equilibrium,
    solve_hydrogenic_helium_statistical_equilibrium,
    solve_coupled_helium_statistical_equilibrium,
    synthesize_hydrogenic_helium_spectrum,
)
from wd_spectra.helium import _helium_i_op_shell_cross_section
from wd_spectra.helium_nlte import _prepare_helium_line_transfer_problem
from wd_spectra.constants import (
    HELIUM_SECOND_IONIZATION_ENERGY,
    PLANCK,
)


def _write_small_ccc_archive(path, maximum_level: int = 3) -> None:
    with ZipFile(path, "w") as archive:
        for lower in range(1, maximum_level + 1):
            ionization = 13.5984346 / lower**2
            archive.writestr(
                f"TICS.{lower}",
                "\n".join(
                    f"{energy:.8f} {cross_section:.8f}"
                    for energy, cross_section in (
                        (ionization, 0.0),
                        (ionization + 0.5, 0.4 * lower),
                        (ionization + 3.0, 0.8 * lower),
                        (ionization + 30.0, 0.15 * lower),
                    )
                ),
            )
            for upper in range(lower + 1, maximum_level + 1):
                threshold = 13.5984346 * (
                    1.0 / lower**2 - 1.0 / upper**2
                )
                archive.writestr(
                    f"{upper}.{lower}",
                    "\n".join(
                        f"{energy:.8f} {cross_section:.8f}"
                        for energy, cross_section in (
                            (threshold, 0.0),
                            (threshold + 0.2, 2.0 * upper),
                            (threshold + 2.0, 4.0 * upper),
                            (threshold + 30.0, 0.5 * upper),
                        )
                    ),
                )


def test_helium_ii_hydrogenic_atomic_scaling():
    transition = helium_ii_shell_transition(2, 3)
    assert np.isclose(transition.wavelength_vacuum_angstrom, 1640.4, atol=0.3)
    assert np.isclose(transition.absorption_oscillator_strength, 0.640747044864)

    threshold = HELIUM_SECOND_IONIZATION_ENERGY / PLANCK
    frequency = np.asarray([0.999 * threshold, threshold, 1.001 * threshold])
    cross_section = helium_ii_photoionization_cross_section(1, frequency)
    assert cross_section[0] == 0.0
    assert np.all(cross_section[1:] > 0.0)


def test_population_transfer_accepts_extended_heii_transition():
    atmosphere = gray_helium_atmosphere(100_000.0, 7.0, n_depth=4)
    line = helium_ii_shell_transition(1, 9)
    problem = _prepare_helium_line_transfer_problem(
        atmosphere, line, 14, stark_table=None
    )
    assert problem.line.lower_level == 1
    assert problem.line.upper_level == 9
    assert np.all(np.isfinite(problem.lte_line_opacity))
    assert np.any(problem.lte_line_opacity > 0.0)


def test_tlusty_helium_superlevel_photoionization_keeps_multiplicity():
    frequency = np.asarray([5.0e14, 1.0e15, 2.0e15])
    singlet = _helium_i_op_shell_cross_section(
        frequency, 3, multiplicity=1
    )
    triplet = _helium_i_op_shell_cross_section(
        frequency, 3, multiplicity=3
    )
    combined = _helium_i_op_shell_cross_section(frequency, 3)
    np.testing.assert_allclose(combined, (singlet + 3.0 * triplet) / 4.0)
    assert np.any(np.abs(singlet - triplet) > 0.05 * combined)


def test_optional_tlusty_storey_hummer_helium_collision_reader(tmp_path):
    source = ModelData.default().tlusty_source
    atom = ModelData.default().tlusty_helium_atom
    if not source.exists() or not atom.exists():
        import pytest

        pytest.skip("optional official TLUSTY source is not downloaded")
    collision_data = read_tlusty_helium_collision_data(source, atom)
    np.testing.assert_allclose(
        collision_data.ionization_scale[:5],
        [1.64, 16.8, 22.9, 17.5, 20.4],
    )
    rates = collision_data.term_rate_matrix(
        np.asarray([10_000.0, 40_000.0, 100_000.0, 300_000.0])
    )
    assert rates.shape == (4, 14, 14)
    assert np.all(rates >= 0.0)
    np.testing.assert_allclose(
        rates[:2, 0, 4],
        [2.86805794e-20, 4.38318570e-12],
        rtol=2.0e-8,
    )
    archive = tmp_path / "ccc.zip"
    _write_small_ccc_archive(archive)
    hydrogenic_collisions = read_ccc_hydrogen_collision_data(
        archive, maximum_level=3
    )
    atmosphere = gray_helium_atmosphere(60_000.0, 8.0, n_depth=8)
    state = solve_coupled_helium_statistical_equilibrium(
        atmosphere,
        hydrogenic_collisions,
        maximum_helium_ii_level=3,
        helium_i_collision_data=collision_data,
    )
    np.testing.assert_allclose(
        state.neutral_departure_coefficient,
        1.0,
        rtol=2.0e-11,
        atol=2.0e-11,
    )


def test_neutral_helium_planck_field_recovers_lte():
    atmosphere = gray_helium_atmosphere(40_000.0, 8.0, n_depth=12)
    state = solve_neutral_helium_statistical_equilibrium(atmosphere)
    assert state.population_density.shape == (atmosphere.n_depth, 14)
    np.testing.assert_allclose(
        state.departure_coefficient, 1.0, rtol=5.0e-12, atol=5.0e-12
    )
    np.testing.assert_allclose(
        state.continuum_departure_coefficient,
        1.0,
        rtol=5.0e-12,
        atol=5.0e-12,
    )


def test_coupled_helium_planck_field_recovers_lte_and_conserves_particles(tmp_path):
    archive = tmp_path / "ccc.zip"
    _write_small_ccc_archive(archive)
    collisions = read_ccc_hydrogen_collision_data(archive, maximum_level=3)
    atmosphere = gray_helium_atmosphere(60_000.0, 8.0, n_depth=10)
    state = solve_coupled_helium_statistical_equilibrium(
        atmosphere, collisions, maximum_helium_ii_level=3
    )
    np.testing.assert_allclose(
        state.neutral_departure_coefficient, 1.0, rtol=2.0e-11, atol=2.0e-11
    )
    np.testing.assert_allclose(
        state.singly_ionized_departure_coefficient,
        1.0,
        rtol=2.0e-11,
        atol=2.0e-11,
    )
    np.testing.assert_allclose(
        state.doubly_ionized_departure_coefficient,
        1.0,
        rtol=2.0e-11,
        atol=2.0e-11,
    )
    actual = (
        np.sum(state.neutral_population_density, axis=1)
        + np.sum(state.singly_ionized_population_density, axis=1)
        + state.doubly_ionized_he_density
    )
    reference = (
        np.sum(state.lte_neutral_population_density, axis=1)
        + np.sum(state.lte_singly_ionized_population_density, axis=1)
        + state.lte_doubly_ionized_he_density
    )
    np.testing.assert_allclose(actual, reference, rtol=3.0e-13)

    nearby = gray_helium_atmosphere(62_000.0, 8.0, n_depth=10)
    mapped = remap_coupled_helium_state(nearby, state)
    mapped_actual = (
        np.sum(mapped.neutral_population_density, axis=1)
        + np.sum(mapped.singly_ionized_population_density, axis=1)
        + mapped.doubly_ionized_he_density
    )
    mapped_reference = (
        np.sum(mapped.lte_neutral_population_density, axis=1)
        + np.sum(mapped.lte_singly_ionized_population_density, axis=1)
        + mapped.lte_doubly_ionized_he_density
    )
    np.testing.assert_allclose(mapped_actual, mapped_reference, rtol=3.0e-13)
    assert mapped.metadata["state_remap"].startswith("frozen coupled")


def test_tlusty_mihalas_heii_collision_rates_and_lte_balance(tmp_path):
    archive = tmp_path / "ccc.zip"
    _write_small_ccc_archive(archive)
    collisions = read_ccc_hydrogen_collision_data(archive, maximum_level=3)
    temperature = np.asarray([60_000.0])
    np.testing.assert_allclose(
        helium_ii_excitation_collision_rate_coefficient(
            collisions, temperature, 1, 2, model="tlusty-mihalas"
        ),
        [3.998895532e-12],
        rtol=2.0e-10,
    )
    np.testing.assert_allclose(
        helium_ii_ionization_collision_rate_coefficient(
            collisions, temperature, 1, model="tlusty-mihalas"
        ),
        [3.268374148e-14],
        rtol=2.0e-10,
    )
    atmosphere = gray_helium_atmosphere(60_000.0, 8.0, n_depth=8)
    state = solve_coupled_helium_statistical_equilibrium(
        atmosphere,
        collisions,
        maximum_helium_ii_level=3,
        hydrogenic_collision_model="tlusty-mihalas",
    )
    np.testing.assert_allclose(
        state.singly_ionized_departure_coefficient,
        1.0,
        rtol=2.0e-11,
        atol=2.0e-11,
    )
    assert state.metadata["hydrogenic_collision_model"] == "tlusty-mihalas"


def test_coupled_helium_supports_tmap_sized_n32_top_closure(tmp_path):
    archive = tmp_path / "ccc.zip"
    _write_small_ccc_archive(archive)
    collisions = read_ccc_hydrogen_collision_data(archive, maximum_level=3)
    atmosphere = gray_helium_atmosphere(60_000.0, 8.0, n_depth=4)
    state = solve_coupled_helium_statistical_equilibrium(
        atmosphere, collisions, maximum_helium_ii_level=32
    )
    assert state.singly_ionized_population_density.shape == (4, 32)
    np.testing.assert_allclose(
        state.singly_ionized_departure_coefficient,
        1.0,
        rtol=3.0e-10,
        atol=3.0e-10,
    )


def test_helium_ii_planck_field_recovers_lte(tmp_path):
    archive = tmp_path / "ccc.zip"
    _write_small_ccc_archive(archive)
    collisions = read_ccc_hydrogen_collision_data(archive, maximum_level=3)
    atmosphere = gray_helium_atmosphere(80_000.0, 8.0, n_depth=12)
    state = solve_hydrogenic_helium_statistical_equilibrium(
        atmosphere,
        collisions,
        maximum_level=3,
    )
    np.testing.assert_allclose(
        state.departure_coefficient, 1.0, rtol=2.0e-12, atol=2.0e-12
    )
    np.testing.assert_allclose(
        state.continuum_departure_coefficient,
        1.0,
        rtol=2.0e-12,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        np.sum(state.population_density, axis=1)
        + state.doubly_ionized_he_density,
        np.sum(state.lte_population_density, axis=1)
        + state.lte_doubly_ionized_he_density,
        rtol=2.0e-13,
    )

    wavelength = np.geomspace(100.0, 20_000.0, 80)
    # Compare the atomic coefficients in the exact LTE limit. The legacy
    # synthesizer uses four scattering iterations; the released LTE formal
    # solver now closes scattering directly and is not that numerical method.
    from wd_spectra.helium_nlte import helium_nlte_transfer_coefficients
    from wd_spectra.helium import helium_continuum_mass_absorption_coefficient
    from wd_spectra.spectrum import planck_lambda_angstrom
    c = helium_nlte_transfer_coefficients(
        atmosphere, wavelength, state, maximum_level=3,
        include_helium_i_lines=False, include_helium_i_resonance_lines=False,
        include_helium_ii_lines=False)
    lte = helium_continuum_mass_absorption_coefficient(
        atmosphere, wavelength, include_electron_scattering=False,
        include_rayleigh_scattering=False)
    np.testing.assert_allclose(c.true_absorption, lte, rtol=3e-12)
    np.testing.assert_allclose(c.thermal_emissivity,
        lte*planck_lambda_angstrom(wavelength[:, None], atmosphere.temperature[None, :]),
        rtol=3e-12)


def test_helium_collision_extrapolation_preserves_positive_strength_and_balance():
    from wd_spectra.helium_collisions import (
        TlustyHeliumCollisionData, _FINE_ENERGY_EV, _FINE_STATISTICAL_WEIGHT)
    # A positive polynomial on its native interval that becomes negative
    # under unrestricted extrapolation; no optional atomic archive needed.
    data=TlustyHeliumCollisionData(np.arange(172)*3+1,
        np.tile([1.,0.,-.1],171),'synthetic-test')
    t=np.array([100.,1000.,10000.,50000.,150000.,1e6])
    rate=data.fine_structure_rate_matrix(t)
    assert np.all(np.isfinite(rate)) and np.all(rate>=0)
    for lower,upper in [(0,1),(2,4),(8,18)]:
        downward=rate[:,upper,lower]
        np.testing.assert_allclose(downward[0]*np.sqrt(t[0]),downward[1]*np.sqrt(t[1]),rtol=1e-14)
        np.testing.assert_allclose(downward[3:]*np.sqrt(t[3:]),np.full(3,downward[3]*np.sqrt(t[3])),rtol=1e-14)
        expected=downward*_FINE_STATISTICAL_WEIGHT[upper]/_FINE_STATISTICAL_WEIGHT[lower]*np.exp(
            (_FINE_ENERGY_EV[lower]-_FINE_ENERGY_EV[upper])/(8.62e-5*t))
        np.testing.assert_allclose(rate[:,lower,upper],expected,rtol=1e-14)


def test_pg1159_helium_continuum_cache_reweights_new_populations(tmp_path):
    from wd_spectra._pg1159_helium import CoupledHeliumNLTEModel,PG1159HeliumTransferCache
    path=tmp_path/'ccc.zip';_write_small_ccc_archive(path)
    data=read_ccc_hydrogen_collision_data(path,maximum_level=3)
    a=gray_helium_atmosphere(90000.,7.,n_depth=8)
    model=CoupledHeliumNLTEModel(data,maximum_helium_ii_level=3,
        helium_i_stark_table=object(),include_helium_i_lines=False,
        include_helium_i_resonance_lines=False,include_helium_ii_lines=False)
    first=solve_coupled_helium_statistical_equilibrium(a,data,maximum_helium_ii_level=3)
    second=solve_coupled_helium_statistical_equilibrium(a,data,maximum_helium_ii_level=3,
        helium_ii_line_mean_intensity_nu={(1,2):np.full(a.n_depth,1e-3)})
    wave=np.geomspace(50.,10000.,160)
    cache=PG1159HeliumTransferCache(a,wave)
    model.transfer_coefficients(a,wave,first,_cache=cache)
    cached=model.transfer_coefficients(a,wave,second,_cache=cache)
    fresh=model.transfer_coefficients(a,wave,second)
    for name in ('true_absorption','thermal_emissivity','scattering'):
        np.testing.assert_array_equal(getattr(cached,name),getattr(fresh,name))
    assert len(cache.values)==2
