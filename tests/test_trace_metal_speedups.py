"""Exactness/accuracy of the trace-metal NLTE speed-ups (no observed-star claims)."""
import numpy as np
import pytest
from wd_spectra import light_metal_nlte as L
from wd_spectra.atmosphere import gray_hydrogen_atmosphere
from wd_spectra.hot_trace_metals import (_relaxed, fixed_electron_metal_reference,
    solve_hot_trace_metals)
from wd_spectra.light_metal_nlte import (ReducedLightMetalLevelState,
    light_metal_bound_free_nlte_coefficients, reduced_light_metal_wavelength,
    solve_reduced_light_metal_levels_nlte)
from wd_spectra.metals import read_pg1159_atomic_database, read_verner_photoionization_database
from wd_spectra.models.common import ModelData
from wd_spectra.nlte_core import NLTETransferCoefficients
from wd_spectra.spectrum import planck_lambda_angstrom

compiled = pytest.mark.skipif(
    L._rt is None or not hasattr(L._rt, "expand_metal_line_blocks"),
    reason="compiled block kernels unavailable")


@pytest.fixture(scope="module")
def atoms():
    data = ModelData.default()
    return (read_pg1159_atomic_database(data.stout, elements=("C", "Si")),
            read_verner_photoionization_database(data.verner_photoionization, elements=("C", "Si")))


@pytest.fixture
def atmosphere():
    return gray_hydrogen_atmosphere(50000., 7.75, n_depth=6, tau_max=30.)


def _forest(n_depth=8, n_line=300, seed=1):
    """Dense line-sampled grid with Doppler cores and deep Stark-broadened wings."""
    rng = np.random.default_rng(seed)
    centers = np.sort(rng.uniform(1000., 1100., n_line))
    velocity = np.array([-1200, -300, -80, -20, -10, 0, 10, 20, 80, 300, 1200]) / 299792.458
    wave = np.unique(np.concatenate([np.arange(980., 1120., 0.02)]
                                    + [c * (1 + velocity) for c in centers]))
    sigma = rng.uniform(0.006, 0.012, (n_line, 1)) * np.ones((n_line, n_depth))
    gamma = 10 ** np.linspace(-6., 0., n_depth)[None, :] * 10 ** rng.uniform(-1, 0.3, (n_line, 1))
    return rng, wave, centers, sigma, gamma


@compiled
def test_line_means_blocks_exact_at_zero_and_accurate():
    rng, wave, centers, sigma, gamma = _forest()
    smooth = (1. + 0.3 * np.sin(wave / 7.))[:, None] * np.ones(sigma.shape[1])
    forest = smooth * rng.uniform(0.2, 1., (wave.size, sigma.shape[1]))
    constant = np.ones_like(forest)
    exact = L._profile_weighted_line_field_means(wave, (forest, smooth, constant), centers, sigma, gamma)
    legacy_j, legacy_lambda = L._profile_weighted_line_means(wave, forest, smooth, centers, sigma, gamma)
    np.testing.assert_array_equal(exact[0], legacy_j)
    np.testing.assert_array_equal(exact[1], legacy_lambda)
    blocked = L._profile_weighted_line_field_means(
        wave, (forest, smooth, constant), centers, sigma, gamma, block_tolerance=1e-4)
    np.testing.assert_allclose(blocked[0], exact[0], rtol=1e-5)
    np.testing.assert_allclose(blocked[1], exact[1], rtol=1e-6)
    # Shared quadrature: a constant field is returned exactly.
    np.testing.assert_allclose(blocked[2], 1., rtol=1e-12)


@compiled
def test_line_opacity_block_deposit_is_pointwise_accurate():
    rng, wave, centers, sigma, gamma = _forest(seed=2)
    n_line, n_depth = sigma.shape
    temperature = np.linspace(30000., 90000., n_depth)
    planck = planck_lambda_angstrom(wave[:, None], temperature[None, :])
    planck_depth_major = np.ascontiguousarray(planck.T)
    zero = np.zeros((n_line, n_depth))
    population = 10 ** rng.uniform(-3, 3, (n_line, 1)) * np.ones((n_line, n_depth))
    lower = rng.uniform(.5, 1.5, (n_line, n_depth))
    upper = rng.uniform(.5, 1.5, (n_line, n_depth))
    exponential = rng.uniform(0., .1, (n_line, n_depth))
    strength = rng.uniform(.01, 1., n_line) * 0.0265

    def accumulate(tolerance):
        absorption = np.zeros((n_depth, wave.size))
        emissivity = np.zeros_like(absorption)
        options = {}
        if tolerance:
            blocks = [np.zeros((n_depth, L._metal_block_count(wave.size), 2)) for _ in range(2)]
            options = dict(block_absorption=blocks[0], block_emissivity=blocks[1],
                           block_tolerance=tolerance)
        L._accumulate_metal_line_profiles(
            wave, planck, centers, strength, sigma, gamma, np.zeros(n_line), zero, zero, zero,
            population, lower, upper, exponential, absorption, emissivity, True,
            depth_major=True, planck_depth_major=planck_depth_major, **options)
        if tolerance:
            L._rt.expand_metal_line_blocks(wave, planck_depth_major, blocks[0], blocks[1],
                                           absorption, emissivity)
        return absorption, emissivity

    exact_a, exact_j = accumulate(0.)
    for tolerance in (1e-3, 1e-4):
        a, j = accumulate(tolerance)
        inside = exact_a > 0  # outside every window both are exactly zero
        assert np.all(a[~inside] == 0) and np.all(j[~inside] == 0)
        assert np.max(abs(a - exact_a)[inside] / exact_a[inside]) <= 1.01 * tolerance
        assert np.max(abs(j - exact_j)[inside] / exact_j[inside]) <= 1.01 * tolerance


@pytest.mark.parametrize("inverted", [False, True])
def test_cumulative_kramers_bound_free_matches_per_level(atmosphere, atoms, inverted):
    database, photo = atoms
    reference = fixed_electron_metal_reference(atmosphere, database, {"C": -7.})
    counts = {2: 20, 3: 15, 4: 1}
    wave = reduced_light_metal_wavelength(database, "C", counts, n_continuum_wavelength=200)
    unity = {("C", ion.charge): np.ones(atmosphere.n_depth) for ion in database.ion_stages("C")}
    rng = np.random.default_rng(3)
    departures = {("C", q, level.index): rng.uniform(.2, 3., atmosphere.n_depth)
                  for q in (2, 3, 4) for level in database.ions["C", q].levels}
    if inverted:
        # Upper-stage overpopulation makes the per-level opacity clip act.
        for level in database.ions["C", 3].levels:
            departures["C", 3, level.index] *= 1e3
    options = dict(level_departure_coefficient=departures, elements=("C",),
                   levels_per_charge=counts, include_explicit_kramers=True)
    direct = light_metal_bound_free_nlte_coefficients(
        atmosphere, wave, database, reference, photo, unity, **options)
    cumulative = light_metal_bound_free_nlte_coefficients(
        atmosphere, wave, database, reference, photo, unity, kramers_cumulative=True, **options)
    for a, b in zip(direct, cumulative):
        np.testing.assert_allclose(b, a, rtol=1e-12, atol=1e-14 * np.max(abs(a)))


def test_cumulative_kramers_rates_cache_and_blocks_keep_populations(atmosphere, atoms):
    database, photo = atoms
    reference = fixed_electron_metal_reference(atmosphere, database, {"Si": -7.})
    counts = {2: 20, 3: 15, 4: 1}
    wave = reduced_light_metal_wavelength(database, "Si", counts, n_continuum_wavelength=200)
    intensity = 0.3 * planck_lambda_angstrom(wave[:, None], atmosphere.temperature[None, :])
    base = solve_reduced_light_metal_levels_nlte(
        atmosphere, database, reference, photo, wave, intensity, "Si", counts)
    cache = {}
    for _ in range(2):
        fast = solve_reduced_light_metal_levels_nlte(
            atmosphere, database, reference, photo, wave, intensity, "Si", counts,
            kramers_cumulative=True, rate_cache=cache, profile_block_tolerance=1e-4)
        np.testing.assert_allclose(fast.population_density, base.population_density, rtol=1e-6)
    assert "bound_collisions" in cache


def test_relaxed_update_moves_halfway_in_log_and_conserves_particles():
    lte = np.array([[1.], [1.]])
    old = ReducedLightMetalLevelState("C", (("C", 2, 1), ("C", 2, 2)), np.array([[1.], [1.]]),
                                      lte, {}, 0., {})
    new = ReducedLightMetalLevelState("C", old.level_key, np.array([[2. - 1e-6], [1e-6]]),
                                      lte, {}, 0., {})
    relaxed = _relaxed(old, new, .5)
    np.testing.assert_allclose(relaxed.population_density.sum(axis=0), 2.)
    # The falling level drops to ~sqrt(1e-6) of its value (halfway in log),
    # not merely to one half as an arithmetic mean would.
    assert relaxed.population_density[1, 0] < 2e-3


def test_cold_start_takes_first_solution_undamped(atoms):
    atmosphere = gray_hydrogen_atmosphere(50000., 7.75, n_depth=8, tau_max=30.)

    def background(wave):
        b = planck_lambda_angstrom(wave[:, None], atmosphere.temperature[None, :])
        return NLTETransferCoefficients(wave, np.ones_like(b), b, np.zeros_like(b), {})
    states = []

    def callback(iteration, defect, evaluated, synthesize):
        states.append({e: s.population_density.copy() for e, s in evaluated.items()})
        return iteration == 2
    result = solve_hot_trace_metals(
        atmosphere, background, {"Si": -6.}, np.linspace(1390., 1405., 60),
        atomic_database=atoms[0], photoionization_database=atoms[1],
        levels_per_charge={"Si": {2: 5, 3: 5, 4: 1}}, require_convergence=False,
        state_callback=callback, maximum_iterations=5)
    assert result.metadata["cold_start_first_step"] == "undamped"
    assert result.metadata["profile_block_tolerance"] == 1e-4
    assert len(states) == 2 and not np.allclose(states[0]["Si"], states[1]["Si"])


def test_flux_criterion_converges_to_the_tight_spectrum(atoms):
    atmosphere = gray_hydrogen_atmosphere(50000., 7.75, n_depth=12, tau_max=100.)

    def background(wave):
        b = planck_lambda_angstrom(wave[:, None], atmosphere.temperature[None, :])
        return NLTETransferCoefficients(wave, np.ones_like(b), b, np.zeros_like(b), {})
    common = dict(atomic_database=atoms[0], photoionization_database=atoms[1],
                  levels_per_charge={"C": {2: 8, 3: 6, 4: 1}, "Si": {2: 5, 3: 5, 4: 1}},
                  maximum_iterations=150, n_angle=2, require_convergence=True,
                  accelerated_lambda=True)
    wave = np.linspace(1170., 1410., 301)
    tight = solve_hot_trace_metals(atmosphere, background, {"C": -5., "Si": -5.}, wave,
                                   convergence_criterion="opacity", tolerance=1e-7, **common)
    flux = solve_hot_trace_metals(atmosphere, background, {"C": -5., "Si": -5.}, wave,
                                  convergence_criterion="flux", tolerance=1e-3, **common)
    assert flux.converged and flux.iterations < tight.iterations
    assert flux.metadata["flux_defect_history"][-1] < 1e-3
    np.testing.assert_allclose(flux.spectrum.surface_flux_lambda,
                               tight.spectrum.surface_flux_lambda, rtol=2e-3)


def test_population_test_skips_negligible_levels_and_outer_layers(atoms, monkeypatch):
    import wd_spectra.hot_trace_metals as trace
    atmosphere = gray_hydrogen_atmosphere(50000., 7.75, n_depth=8, tau_max=30.)

    def background(wave):
        b = planck_lambda_angstrom(wave[:, None], atmosphere.temperature[None, :])
        return NLTETransferCoefficients(wave, np.ones_like(b), b, np.zeros_like(b), {})
    common = dict(atomic_database=atoms[0], photoionization_database=atoms[1],
                  levels_per_charge={"Si": {2: 5, 3: 5, 4: 1}}, require_convergence=False,
                  maximum_iterations=1)
    tested = solve_hot_trace_metals(atmosphere, background, {"Si": -6.}, np.linspace(1390., 1405., 40), **common)
    assert tested.metadata["population_defect_limits"] == dict(
        element_fraction_floor=trace.POPULATION_DEFECT_FLOOR,
        minimum_rosseland_depth=trace.POPULATION_DEFECT_MINIMUM_TAU)
    assert tested.population_defect > 0
    # Layers above the cut are not tested: with every layer excluded the defect is zero.
    monkeypatch.setattr(trace, "POPULATION_DEFECT_MINIMUM_TAU", 2 * float(np.max(atmosphere.rosseland_optical_depth)))
    skipped = solve_hot_trace_metals(atmosphere, background, {"Si": -6.}, np.linspace(1390., 1405., 40), **common)
    assert skipped.population_defect == 0
