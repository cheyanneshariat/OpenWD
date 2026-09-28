"""Fast checks of the magnetic hydrogen (DAH) physics and public surface."""

import numpy as np
import pytest
from types import SimpleNamespace
import weakref

from wd_spectra import DAHConfig, compute_dah, select_physics
from wd_spectra.atmosphere import gray_hydrogen_atmosphere
from wd_spectra._compat import trapezoid
from wd_spectra.constants import BOLTZMANN, ELECTRON_MASS, ELEMENTARY_CHARGE_ESU, HYDROGEN_IONIZATION_ENERGY, LIGHT_SPEED, PI, PLANCK
from wd_spectra.eos import hummer_mihalas_hydrogen_lte
from wd_spectra.magnetic import (
    MagneticPhysics,
    BalmerLineTemplate,
    PolarizedOpacity,
    dipole_surface_cells,
    frequency_hilbert_dispersion,
    h2db_balmer_manifolds,
    linear_zeeman_triplet,
    magnetic_continuum,
    synthesize_magnetic_hydrogen_spectrum,
    uniform_field_surface_cells,
    weak_field_balmer_manifolds,
)
from wd_spectra.magnetic_atomic import (
    read_h2db_energy_database,
    read_h2db_transition_database,
    H2dbBalmerComponents,
)
from wd_spectra.magnetic_continuum import (
    hydrogen_free_free_mass_absorption_coefficient,
    isotropic_free_electron_opacity,
    magnetized_free_electron_manifolds,
    _explicit_rwa_atomic_kernels,
    explicit_rwa_hydrogen_bound_free_mass_absorption_coefficient,
)
from wd_spectra.opacity import electron_scattering_mass_coefficient
from wd_spectra.magnetic_eos import magnetic_hummer_mihalas_hydrogen_lte
from wd_spectra.models.common import ModelData
from wd_spectra.opacity import (
    hydrogen_continuum_mass_absorption_coefficient,
    hydrogen_dissolved_level_pseudocontinuum_mass_absorption_coefficient,
)
from wd_spectra.spectrum import synthesize_hydrogen_spectrum
from wd_spectra.validation.magnetic_da import (
    DAH_VALIDATION_TARGETS,
    load_dah_observation,
    score_dah_spectrum,
)

H2DB = ModelData.default().h2db_balmer_subset


@pytest.fixture(scope="module")
def transitions():
    return read_h2db_transition_database(H2DB)


@pytest.fixture(scope="module")
def energies():
    return read_h2db_energy_database(H2DB)


@pytest.fixture(scope="module")
def atmosphere():
    return gray_hydrogen_atmosphere(15_000.0, 8.0, n_depth=24, correlated_microfields=True)


def test_h2db_subset_restores_the_mirrored_pi_components(transitions):
    # 15 Halpha branches: the archive stores 14; the loosely bound 2p(m=+1)
    # pi branch is its exact mirror.
    assert len(transitions.transitions_by_upper_level[3]) == 15
    temperature = np.array([12_000.0])
    for level, rest, tolerance in ((3, 6564.636, 2e-5), (4, 4862.694, 5e-5), (6, 4102.898, 2e-4)):
        components = transitions.balmer_components(level, rest, 0.05, temperature)
        strength = components.normalized_strength[:, 0]
        assert np.sum(strength[components.delta_m == 0]) == pytest.approx(1 / 3, abs=2e-3)
        assert components.line_strength_scale[0] == pytest.approx(1.0, abs=2e-3)
        # Delta m = +1 is the blue component, displaced by the normal triplet
        # (checked at 0.5 MG: at 0.05 MG the shift is at H2db's 1e-5 Ry
        # interpolation accuracy; the residual is the quadratic shift).
        components = transitions.balmer_components(level, rest, 0.5, temperature)
        strength = components.normalized_strength[:, 0]
        blue = components.delta_m > 0
        centroid = np.sum(components.wavelength_angstrom[blue] * strength[blue]) / np.sum(strength[blue])
        assert centroid == pytest.approx(
            linear_zeeman_triplet(rest, 0.5).sigma_blue_angstrom, rel=tolerance
        )


def test_h2db_components_exist_through_the_high_field_track_endpoints(transitions):
    temperature = np.array([20_000.0])
    for level in range(3, 13):
        components = transitions.balmer_components(level, 4000.0, 2_000.0, temperature)
        assert np.all(np.isfinite(components.wavelength_angstrom))


def test_magnetic_eos_has_the_exact_zero_field_limit(energies):
    temperature = np.array([6_000.0, 15_000.0, 25_000.0])
    pressure = np.full(3, 1.0e5)
    reference = hummer_mihalas_hydrogen_lte(temperature, pressure, include_molecules=False)
    weak = magnetic_hummer_mihalas_hydrogen_lte(temperature, pressure, 1.0e-3, energies)
    np.testing.assert_allclose(weak.ionization_fraction, reference.ionization_fraction, rtol=3e-4)
    np.testing.assert_allclose(
        weak.level_population_density[:, :4], reference.level_population_density[:, :4], rtol=3e-4
    )
    assert energies.ground_state_binding_energy_erg(1.0e-4) == pytest.approx(
        HYDROGEN_IONIZATION_ENERGY, rel=1e-6
    )


def test_free_electrons_recover_zero_field_free_free_and_thomson(atmosphere):
    wavelength = np.geomspace(1_000.0, 50_000.0, 400)
    absorption, scattering = isotropic_free_electron_opacity(atmosphere, wavelength, 1.0e-7)
    np.testing.assert_allclose(
        absorption, hydrogen_free_free_mass_absorption_coefficient(atmosphere, wavelength), rtol=1e-8
    )
    np.testing.assert_allclose(
        scattering,
        np.broadcast_to(electron_scattering_mass_coefficient(atmosphere)[None, :], scattering.shape),
        rtol=1e-8,
    )


def test_cyclotron_resonance_has_the_classical_strength_and_far_field_limit(atmosphere):
    field = 200.0
    omega_c = ELEMENTARY_CHARGE_ESU * field * 1e6 / (ELECTRON_MASS * LIGHT_SPEED)
    center = 2.0 * PI * LIGHT_SPEED / omega_c * 1e8
    wavelength = center * np.linspace(0.97, 1.03, 40001)
    manifolds = magnetized_free_electron_manifolds(atmosphere, wavelength, field, 0.7)
    depth = 12
    extinction = manifolds.absorption[2][:, depth] + manifolds.scattering[2][:, depth]
    baseline = manifolds.absorption[1][:, depth] + manifolds.scattering[1][:, depth]
    angular = 2.0 * PI * LIGHT_SPEED / (wavelength * 1e-8)
    per_electron = -trapezoid(extinction - baseline, angular) * (
        atmosphere.mass_density[depth] / atmosphere.electron_density[depth]
    )
    assert per_electron == pytest.approx(
        4.0 * PI**2 * ELEMENTARY_CHARGE_ESU**2 / (ELECTRON_MASS * LIGHT_SPEED), rel=0.03
    )
    # Far from resonance every mode tends to the zero-field value.
    far = magnetized_free_electron_manifolds(atmosphere, np.array([500.0, 600.0]), 0.1, 0.5)
    for triple in (far.absorption, far.scattering):
        np.testing.assert_allclose(triple[0], triple[2], rtol=1e-3)


def test_all_dispersion_profiles_share_one_sign_convention(atmosphere):
    wavelength = np.linspace(4000.0, 6000.0, 4001)
    line = 1.0 / (1.0 + ((wavelength - 5000.0) / 2.0) ** 2)
    psi = frequency_hilbert_dispersion(wavelength, line[:, np.newaxis])[:, 0]
    # Positive on the low-frequency (red) side.
    assert psi[np.searchsorted(wavelength, 5003.0)] > 0.0 > psi[np.searchsorted(wavelength, 4997.0)]
    omega_c = ELEMENTARY_CHARGE_ESU * 200e6 / (ELECTRON_MASS * LIGHT_SPEED)
    center = 2.0 * PI * LIGHT_SPEED / omega_c * 1e8
    grid = center * np.linspace(0.95, 1.05, 2001)
    dispersion = magnetized_free_electron_manifolds(atmosphere, grid, 200.0, 0.5).dispersion[2]
    assert dispersion[np.searchsorted(grid, center * 1.003), 12] > 0.0
    assert dispersion[np.searchsorted(grid, center * 0.997), 12] < 0.0


def test_stokes_coefficients_reduce_to_the_isotropic_mean():
    shape = (5, 3)
    rng = np.random.default_rng(1)
    manifolds = PolarizedOpacity(*(rng.random(shape) for _ in range(3)))
    nodes, weights = np.polynomial.legendre.leggauss(8)
    average = sum(
        0.5 * w * manifolds.stokes(0.5 * (x + 1.0))[0] for x, w in zip(nodes, weights)
    )
    np.testing.assert_allclose(average, manifolds.isotropic(), rtol=1e-12)
    eta_i, eta_q, eta_v = manifolds.stokes(0.3)
    assert np.all(np.sqrt(eta_q**2 + eta_v**2) <= eta_i + 1e-15)


def test_rwa_continuum_tends_to_the_zero_field_continuum(atmosphere, energies):
    wavelength = np.geomspace(1_000.0, 12_000.0, 800)
    reference = hydrogen_continuum_mass_absorption_coefficient(
        atmosphere, wavelength, include_electron_scattering=False,
        include_rayleigh_scattering=False, include_molecular_absorption=False,
    ) + hydrogen_dissolved_level_pseudocontinuum_mass_absorption_coefficient(atmosphere, wavelength)
    unpolarized, rwa = magnetic_continuum(
        atmosphere, wavelength, 0.01, energies, include_rwa_photoionization=True
    )
    total = unpolarized + rwa.isotropic()
    relative = np.abs(total / reference - 1.0)
    assert np.mean(relative) < 1e-3


def test_zero_field_synthesis_is_the_da_spectrum(atmosphere, transitions, energies):
    wavelength = np.arange(4700.0, 5000.0, 1.0)
    da = synthesize_hydrogen_spectrum(
        atmosphere, wavelength, include_series_pseudocontinuum=True,
        include_molecular_absorption=False,
        balmer_self_broadening_truncation_closure="stark-core",
    )
    physics = MagneticPhysics(transitions, energies)
    for transfer, tolerance in (("scalar-stokes-i", 1e-6), ("full-stokes-iquv", 1e-6)):
        dah = synthesize_magnetic_hydrogen_spectrum(
            atmosphere, wavelength, uniform_field_surface_cells(0.0), physics,
            polarized_transfer=transfer,
        )
        np.testing.assert_allclose(dah.surface_flux_lambda, da.surface_flux_lambda, rtol=tolerance)


def test_weak_field_triplet_conserves_the_line_opacity(atmosphere):
    wavelength = np.arange(4700.0, 5030.0, 0.05)
    manifolds = weak_field_balmer_manifolds(atmosphere, wavelength, 0.5)
    zero = weak_field_balmer_manifolds(atmosphere, wavelength, 0.0)
    np.testing.assert_allclose(
        np.sum(manifolds.isotropic()[:, 12]), np.sum(zero.isotropic()[:, 12]), rtol=5e-3
    )


def test_dipole_cells_are_normalized_and_polar_field_is_the_pole():
    cells = dipole_surface_cells(100.0, inclination_deg=0.0, n_field_bins=None)
    assert np.sum(cells.projected_weight) == pytest.approx(1.0)
    assert np.max(cells.field_strength_megagauss) == pytest.approx(100.0, rel=0.02)
    assert np.min(cells.field_strength_megagauss) >= 50.0 - 1e-9
    compressed = dipole_surface_cells(100.0, inclination_deg=60.0, offset_vector_radius=(0.0, 0.1, 0.2))
    assert compressed.projected_weight.size == 21
    assert np.sum(compressed.projected_weight) == pytest.approx(1.0)


def test_dah_configuration_and_selection():
    assert select_physics(DAHConfig()).workflow == "dah"
    with pytest.raises(ValueError, match="dipole_offset_radius"):
        compute_dah(DAHConfig(dipole_offset_radius=(0.0, 0.0, 0.1)))
    with pytest.raises(ValueError, match="field_angle_deg"):
        compute_dah(DAHConfig(field_geometry="dipole", field_angle_deg=30.0))
    with pytest.raises(ValueError, match="3400"):
        compute_dah(DAHConfig(), np.arange(3000.0, 5000.0))


def test_bundled_validation_targets_load_and_score_a_flat_model():
    for target in DAH_VALIDATION_TARGETS:
        observation = load_dah_observation(target.key)
        assert observation.wavelength_angstrom.size > 500
        assert np.all(observation.flux_nu > 0.0)
    wavelength = np.arange(3600.0, 7000.0, 1.0)
    score = score_dah_spectrum("j2149-0728", wavelength, wavelength**-2.0)
    assert np.isfinite(score["broad_rms"]) and "hardy" in score


@pytest.mark.parametrize("component_wave", [4142.1894, 7838.4694])
def test_h2db_net_component_strength_uses_its_photon_energy(atmosphere, component_wave):
    # An isolated transition with a constant absorption cross section makes
    # the LTE net coefficient analytic, independent of interpolation/widths.
    rest = 6564.636
    h_over_kT = PLANCK / (BOLTZMANN * atmosphere.temperature)
    template_opacity = 2.3 * -np.expm1(-LIGHT_SPEED / (rest * 1e-8) * h_over_kT)
    values = np.broadcast_to(template_opacity, (2, atmosphere.n_depth)).copy()
    templates = {3: BalmerLineTemplate(np.array([rest - 100, rest + 100]), values, 2 * values)}
    components = H2dbBalmerComponents(
        np.array([component_wave]), np.ones((1, atmosphere.n_depth)),
        np.ones(atmosphere.n_depth), np.array([1]),
    )
    database = SimpleNamespace(
        transitions_by_upper_level={3: ()}, balmer_components=lambda *args: components
    )
    absorption, dispersion = h2db_balmer_manifolds(
        atmosphere, np.array([component_wave, component_wave + 0.01]), 100.0,
        database, templates=templates, dispersion=True,
    )
    expected = 3 * 2.3 * -np.expm1(-LIGHT_SPEED / (component_wave * 1e-8) * h_over_kT)
    np.testing.assert_allclose(absorption.plus, np.broadcast_to(expected, values.shape), rtol=1e-13)
    np.testing.assert_allclose(dispersion.plus, 2 * absorption.plus, rtol=1e-13)
    assert not np.any(absorption.pi) and not np.any(absorption.minus)


def test_rwa_cache_releases_previous_fields_and_grids(energies):
    cache = _explicit_rwa_atomic_kernels
    cache.cache_clear()
    root = str(energies.root.resolve())
    previous = []
    # Successive surface cells and a second model must not accumulate kernels.
    for field in (10., 30., 100., 120.):
        for n_wave in (16, 24):
            current = []
            grid = tuple(np.linspace(4000., 8000., n_wave))
            for q in (-1, 0, 1):
                kernels = cache(root, grid, field, q, 2)
                assert cache(root, grid, field, q, 2) is kernels
                current.extend(weakref.ref(shell[2]) for shell in kernels)
            del kernels
            assert all(reference() is None for reference in previous)
            assert cache.cache_info().currsize == 3
            previous = current
    cache.cache_clear()
    assert all(reference() is None for reference in previous)


def test_stationary_rwa_uses_boltzmann_substate_populations(atmosphere, energies):
    from wd_spectra.opacity import _atmosphere_level_distribution

    wave = np.linspace(3400., 8000., 40)
    field = 100.0
    kernels = _explicit_rwa_atomic_kernels(
        str(energies.root.resolve()), tuple(wave), field, 1, 3, False
    )
    levels = _atmosphere_level_distribution(atmosphere, maximum_level=40)
    expected = np.zeros((wave.size, atmosphere.n_depth))
    for n, (energy, mass, kernel, _, _) in enumerate(kernels):
        np.testing.assert_array_equal(mass, np.ones_like(mass))
        weight = np.exp(
            -(energy - energy.min())[:, None] * HYDROGEN_IONIZATION_ENERGY
            / (BOLTZMANN * atmosphere.temperature[None, :])
        )
        population = levels.population_density[:, n] * weight / weight.sum(axis=0)
        expected += kernel.T @ population
    expected *= -np.expm1(
        -PLANCK * LIGHT_SPEED / (wave[:, None] * 1e-8 * BOLTZMANN * atmosphere.temperature)
    ) / atmosphere.mass_density
    actual = explicit_rwa_hydrogen_bound_free_mass_absorption_coefficient(
        atmosphere, wave, field, 1, energies, maximum_level=3,
        include_dissolved_levels=False, include_centered_motion=False,
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-13)
    moving = explicit_rwa_hydrogen_bound_free_mass_absorption_coefficient(
        atmosphere, wave, field, 1, energies, maximum_level=3,
        include_dissolved_levels=False, include_centered_motion=True,
    )
    absorbing = actual > 0
    assert np.max(np.abs(moving[absorbing] / actual[absorbing] - 1)) > 1e-6


def test_crossing_dipole_uses_same_structure_and_synthesis_regime(monkeypatch, atmosphere, transitions):
    import wd_spectra.models.dah as dah
    from wd_spectra.magnetic import magnetic_continuum
    from wd_spectra.spectrum import Spectrum

    config = DAHConfig(
        field_geometry="dipole", magnetic_field_megagauss=1.1,
        include_cyclotron_absorption=False, include_centered_motion=False,
    )
    cells = dah.dah_surface_cells(config)
    field = dah.structure_field_megagauss(cells)
    assert field < 1 < cells.field_strength_megagauss.max()
    wave = np.linspace(3900., 7000., 20)
    captured = {}

    def structure(*args, **kwargs):
        captured.update(kwargs)
        return atmosphere

    def synthesis(atm, wavelength, cells, physics, **kwargs):
        assert physics.regime(field) == physics.regime(cells.field_strength_megagauss.max()) == "h2db"
        assert not captured["include_series_pseudocontinuum"]
        np.testing.assert_allclose(
            captured["balmer_opacity_function"](atm, wavelength),
            h2db_balmer_manifolds(atm, wavelength, field, transitions).isotropic(),
        )
        unpolarized, rwa = magnetic_continuum(
            atm, wavelength, field, physics.energies,
            include_rwa_photoionization=True, include_centered_motion=False,
        )
        np.testing.assert_allclose(
            captured["continuum_opacity_function"](atm, wavelength),
            unpolarized + rwa.isotropic(),
        )
        return Spectrum(wavelength, np.ones_like(wavelength), {
            "magnetic_line_regime": "h2db", "polarized_transfer": "test",
        })

    monkeypatch.setattr(dah, "radiative_equilibrium_hydrogen_atmosphere", structure)
    monkeypatch.setattr(dah, "synthesize_magnetic_hydrogen_spectrum", synthesis)
    monkeypatch.setattr(dah, "warn_if_atmosphere_not_converged", lambda *args: "unknown")
    result = compute_dah(config, wave)
    assert result.metadata["line_physics"] == "h2db"


def test_centered_motion_switch_reaches_absorption_and_dispersion(monkeypatch, atmosphere):
    import wd_spectra.magnetic as magnetic
    import wd_spectra.models.dah as dah

    calls = []

    def continuum(atm, wave, field, polarization, energies, **kwargs):
        calls.append((wave.size, polarization, kwargs.get("include_centered_motion")))
        return np.zeros((wave.size, atm.n_depth))

    monkeypatch.setattr(magnetic, "explicit_rwa_hydrogen_bound_free_mass_absorption_coefficient", continuum)
    monkeypatch.setattr(dah, "warn_if_atmosphere_not_converged", lambda *args: "unknown")
    wave = np.linspace(4000., 7000., 20)
    compute_dah(
        DAHConfig(magnetic_field_megagauss=100., field_angle_deg=40.,
                  include_centered_motion=False, include_magnetic_eos=False,
                  include_cyclotron_absorption=False),
        wave, initial_atmosphere=atmosphere, relax_atmosphere=False,
    )
    assert len(calls) == 6
    assert {n for n, _, _ in calls} == {wave.size, wave.size + 1000}
    assert all(flag is False for _, _, flag in calls)


def test_dah_physics_revision_invalidates_old_fixed_structure_claim(monkeypatch, atmosphere):
    from wd_spectra.models import common

    config = DAHConfig(effective_temperature=atmosphere.effective_temperature, logg=atmosphere.logg)
    data = ModelData.default()
    with monkeypatch.context() as historical:
        historical.setattr(common, "_MODEL_FAMILY_PHYSICS_REVISIONS", {})
        old = common.model_request_fingerprint("DAH", config, data)
    current = common.model_request_fingerprint("DAH", config, data)
    checkpoint = common.atmosphere_with_model_request_fingerprint(atmosphere, old)
    fixed = common.fixed_synthesis_atmosphere(checkpoint, current)
    assert old["sha256"] != current["sha256"]
    assert not fixed.metadata["fixed_synthesis_request_verified"]


@pytest.mark.parametrize("centered_motion", [False, True])
def test_fixed_checkpoint_restores_mean_field_eos_before_stark_templates(
    monkeypatch, tmp_path, atmosphere, energies, centered_motion
):
    import wd_spectra.models.dah as dah
    from wd_spectra.magnetic_eos import atmosphere_with_magnetic_hydrogen_eos
    from wd_spectra.models.common import load_atmosphere_checkpoint
    from wd_spectra.spectrum import Spectrum

    config = DAHConfig(
        effective_temperature=atmosphere.effective_temperature, logg=atmosphere.logg,
        magnetic_field_megagauss=111.49, field_geometry="dipole",
        include_centered_motion=centered_motion,
    )
    mean_field = dah.structure_field_megagauss(dah.dah_surface_cells(config))
    expected = atmosphere_with_magnetic_hydrogen_eos(
        atmosphere, mean_field, energies, include_centered_motion=centered_motion
    )
    path = tmp_path / "checkpoint.npz"
    np.savez(path, **{k: getattr(expected, k) for k in (
        "temperature", "gas_pressure", "column_mass", "rosseland_optical_depth",
        "mass_density", "electron_density",
    )})
    restored = load_atmosphere_checkpoint(
        path, config.effective_temperature, config.logg, "hydrogen"
    )
    assert not np.allclose(restored.neutral_h_density, expected.neutral_h_density, rtol=1e-4)

    def synthesis(reference, wavelength, cells, physics, **kwargs):
        # These are the actual reference densities consumed by the shared
        # Stark templates; local n=2 rescaling cannot repair their widths.
        for name in ("mass_density", "electron_density", "neutral_h_density"):
            np.testing.assert_allclose(getattr(reference, name), getattr(expected, name), rtol=1e-12)
        for name in ("temperature", "gas_pressure", "column_mass"):
            np.testing.assert_array_equal(getattr(reference, name), getattr(restored, name))
        assert not reference.metadata["fixed_synthesis_request_verified"]
        return Spectrum(wavelength, np.ones_like(wavelength), {
            "magnetic_line_regime": "h2db", "polarized_transfer": "test",
        })

    monkeypatch.setattr(dah, "synthesize_magnetic_hydrogen_spectrum", synthesis)
    monkeypatch.setattr(dah, "warn_if_atmosphere_not_converged", lambda *args: "unconverged")
    result = compute_dah(config, np.array([4000., 5000.]), initial_atmosphere=restored, relax_atmosphere=False)
    assert result.metadata["atmosphere_convergence_status"] == "unconverged"


def test_strong_field_restart_honors_requested_depth_resolution(monkeypatch, atmosphere):
    import wd_spectra.models.dah as dah

    class CapturedRestart(Exception):
        pass

    def structure(*args, **kwargs):
        assert kwargs["n_depth"] == 100
        assert kwargs.get("initial_atmosphere") is None
        np.testing.assert_array_equal(kwargs["initial_temperature"], atmosphere.temperature)
        np.testing.assert_array_equal(kwargs["initial_column_mass"], atmosphere.column_mass)
        raise CapturedRestart

    monkeypatch.setattr(dah, "radiative_equilibrium_hydrogen_atmosphere", structure)
    config = DAHConfig(effective_temperature=atmosphere.effective_temperature,
                       logg=atmosphere.logg, magnetic_field_megagauss=100., quality="production")
    with pytest.raises(CapturedRestart):
        compute_dah(config, np.array([4000., 5000.]), initial_atmosphere=atmosphere)


def test_refined_restart_still_rejects_mismatched_stellar_parameters(atmosphere):
    with pytest.raises(ValueError, match="must match effective temperature"):
        compute_dah(DAHConfig(effective_temperature=16000., magnetic_field_megagauss=100.,
                              quality="production"), np.array([4000., 5000.]),
                    initial_atmosphere=atmosphere)


@pytest.mark.parametrize("wave", [[], [4000.], [np.nan, 5000.], [4000., np.inf],
                                  [5000., 4000.], [4000., 4000.]])
def test_invalid_dah_grid_is_rejected_before_structure_work(monkeypatch, wave):
    import wd_spectra.models.dah as dah

    def forbidden(*args, **kwargs):
        pytest.fail("Invalid grids must fail before solving an atmosphere")

    monkeypatch.setattr(dah, "compute_da", forbidden)
    with pytest.raises(ValueError, match="finite, increasing 1D wavelength grid"):
        compute_dah(DAHConfig(), wave)
