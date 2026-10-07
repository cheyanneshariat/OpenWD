"""Conservation/transfer checks; not an observed-star qualification."""
import numpy as np
import pytest
from wd_spectra.atmosphere import gray_hydrogen_atmosphere
from wd_spectra.hot_nlte import transfer_field
from wd_spectra.hot_trace_metals import (TraceMetalConvergenceWarning,
    fixed_electron_metal_reference, solve_hot_trace_metals, synthesize_dao_trace_metals)
from wd_spectra.light_metal_nlte import (light_metal_bound_free_nlte_coefficients,
    reduced_light_metal_wavelength, solve_reduced_light_metal_levels_nlte)
from wd_spectra.metals import read_pg1159_atomic_database, read_verner_photoionization_database
from wd_spectra.models.common import ModelData
from wd_spectra.nlte_core import NLTETransferCoefficients
from wd_spectra.spectrum import planck_lambda_angstrom

@pytest.fixture(scope="module")
def atoms():
    data = ModelData.default()
    return (read_pg1159_atomic_database(data.stout, elements=("C", "Si")),
            read_verner_photoionization_database(data.verner_photoionization, elements=("C", "Si")))

@pytest.fixture
def atmosphere():
    return gray_hydrogen_atmosphere(50000., 7.75, n_depth=5, tau_max=10.)

def thermal_background(atmosphere):
    def background(wave):
        b = planck_lambda_angstrom(wave[:, None], atmosphere.temperature[None, :])
        return NLTETransferCoefficients(wave, np.ones_like(b), b, np.zeros_like(b), {})
    return background

def test_fixed_reference_uses_h_nuclei_and_preserves_ne(atmosphere, atoms):
    old = atmosphere.electron_density.copy()
    ref = fixed_electron_metal_reference(atmosphere, atoms[0], {"C": -7., "Si": -6.5})
    for e, abundance in ref.log_number_abundance.items():
        np.testing.assert_allclose(ref.ion_number_density[e].sum(axis=0),
            atmosphere.hydrogen_lte_state.hydrogen_nuclei_density * 10**abundance, rtol=2e-15)
    np.testing.assert_array_equal(old, ref.electron_density)
    np.testing.assert_array_equal(old, atmosphere.electron_density)

@pytest.mark.parametrize("element,counts", [
    ("C", {2: 8, 3: 3, 4: 1}), ("Si", {2: 3, 3: 3, 4: 1}),
    ("C", {2: 8, 3: 3, 4: 3, 5: 1}), ("Si", {2: 3, 3: 3, 4: 3, 5: 1})])
def test_planck_rates_recover_lte_and_particle_conservation(atmosphere, atoms, element, counts):
    database, photo = atoms
    reference = fixed_electron_metal_reference(atmosphere, database, {element: -7.})
    wave = reduced_light_metal_wavelength(database, element, counts, n_continuum_wavelength=120)
    intensity = planck_lambda_angstrom(wave[:, None], atmosphere.temperature[None, :])
    state = solve_reduced_light_metal_levels_nlte(
        atmosphere, database, reference, photo, wave, intensity, element, counts)
    np.testing.assert_allclose(state.population_density, state.lte_population_density, rtol=3e-6)
    diluted = solve_reduced_light_metal_levels_nlte(
        atmosphere, database, reference, photo, wave, intensity * 0.4, element, counts)
    np.testing.assert_allclose(diluted.population_density.sum(axis=0),
                               state.lte_population_density.sum(axis=0), rtol=1e-12)
    assert np.max(abs(diluted.population_density / state.lte_population_density - 1)) > 0.05

def test_excited_continua_kirchhoff_and_zero_count_exclusion(atmosphere, atoms):
    database, photo = atoms
    reference = fixed_electron_metal_reference(atmosphere, database, {"C": -7.})
    wave = np.geomspace(20., 5000., 240)
    unity = {("C", ion.charge): np.ones(atmosphere.n_depth) for ion in database.ion_stages("C")}
    counts = {ion.charge: 0 for ion in database.ion_stages("C")}
    a, j = light_metal_bound_free_nlte_coefficients(atmosphere, wave, database,
        reference, photo, unity, levels_per_charge=counts, include_explicit_kramers=True)
    assert np.count_nonzero(a) == np.count_nonzero(j) == 0
    counts[2] = 8
    a, j = light_metal_bound_free_nlte_coefficients(atmosphere, wave, database,
        reference, photo, unity, levels_per_charge=counts, include_explicit_kramers=True)
    assert np.max(a) > 0
    b = planck_lambda_angstrom(wave[:, None], atmosphere.temperature[None, :])
    np.testing.assert_allclose(j, a * b, rtol=3e-14, atol=1e-100)


def test_real_carbon_op_tables_preserve_detailed_balance(atmosphere, atoms):
    from wd_spectra.light_metal_nlte import read_tlusty_photoionization_threshold_data
    database,photo=atoms
    data=ModelData.default()
    tables={2:read_tlusty_photoionization_threshold_data(data.tlusty_atoms/'c3.dat'),
            3:read_tlusty_photoionization_threshold_data(data.tlusty_atoms/'c4_35+2lev.dat')}
    counts={2:12,3:8,4:1}
    ref=fixed_electron_metal_reference(atmosphere,database,{'C':-7.})
    wave=reduced_light_metal_wavelength(database,'C',counts,n_continuum_wavelength=120,
                                        photoionization_threshold_data=tables)
    b=planck_lambda_angstrom(wave[:,None],atmosphere.temperature)
    state=solve_reduced_light_metal_levels_nlte(atmosphere,database,ref,photo,wave,b,'C',counts,
                                               photoionization_threshold_data=tables)
    np.testing.assert_allclose(state.population_density,state.lte_population_density,rtol=3e-6)
    unity={('C',s.charge):np.ones(atmosphere.n_depth) for s in database.ion_stages('C')}
    selected={s.charge:counts.get(s.charge,0) if s.charge<4 else 0 for s in database.ion_stages('C')}
    a,j=light_metal_bound_free_nlte_coefficients(atmosphere,wave,database,ref,photo,unity,
        elements=('C',),levels_per_charge=selected,include_explicit_kramers=True,
        photoionization_threshold_data=tables)
    np.testing.assert_allclose(j,a*b,rtol=3e-14,atol=1e-100)
    approximate,_=light_metal_bound_free_nlte_coefficients(atmosphere,wave,database,ref,photo,unity,
        elements=('C',),levels_per_charge=selected,include_explicit_kramers=True)
    assert np.max(abs(a-approximate))/np.max(a)>.01

def test_zero_metals_is_exact_background(atmosphere):
    wave = np.linspace(1174., 1177., 101)
    background = thermal_background(atmosphere)
    expected = transfer_field(atmosphere, background(wave))[1].interface_flux[:, 0]
    result = solve_hot_trace_metals(atmosphere, background, {}, wave)
    np.testing.assert_array_equal(result.spectrum.surface_flux_lambda, expected)
    assert result.converged and result.iterations == 0 and result.population_defect == 0


def test_warm_start_and_callback_keep_evaluated_state(atmosphere, atoms):
    from wd_spectra.hot_trace_metals import _initial_state
    database,photo=atoms
    counts={2:3,3:3,4:1}
    ref=fixed_electron_metal_reference(atmosphere,database,{'Si':-7.})
    wave=reduced_light_metal_wavelength(database,'Si',counts,n_continuum_wavelength=120)
    p=solve_reduced_light_metal_levels_nlte(atmosphere,database,ref,photo,wave,
        .4*planck_lambda_angstrom(wave[:,None],atmosphere.temperature),'Si',counts)
    mapped=_initial_state(atmosphere,database,ref,'Si',{2:5,3:5,4:1},p)
    np.testing.assert_allclose(mapped.population_density.sum(axis=0),mapped.lte_population_density.sum(axis=0))
    snapshots=[]
    def callback(i,d,states,synthesize):
        snapshots.append((d,synthesize().surface_flux_lambda.copy()))
        return True
    with pytest.warns(TraceMetalConvergenceWarning):
        result=solve_hot_trace_metals(atmosphere,thermal_background(atmosphere),{'Si':-7.},
            np.linspace(1393.,1404.,80),atomic_database=database,photoionization_database=photo,
            levels_per_charge={'Si':counts},initial_populations={'Si':p},state_callback=callback,
            require_convergence=False,tolerance=1e-10)
    assert not result.converged and result.iterations==1 and result.metadata['stopped_by_callback']
    assert result.population_defect==snapshots[0][0]
    assert result.population_defect==max(result.metadata['element_population_defects'].values())
    assert result.metadata['worst_population_defect']['element']=='Si'
    assert len(result.metadata['element_population_defect_history'])==1
    np.testing.assert_array_equal(result.spectrum.surface_flux_lambda,snapshots[0][1])

def test_nonconvergence_is_not_hidden_by_tiny_damping(atmosphere, atoms):
    wave = np.linspace(1393., 1404., 51)
    background = thermal_background(atmosphere)
    options = dict(atomic_database=atoms[0], photoionization_database=atoms[1],
                   levels_per_charge={"Si": {2: 3, 3: 3, 4: 1}},
                   maximum_iterations=2, damping=1e-10, tolerance=1e-6)
    with pytest.raises(RuntimeError, match="undamped defect"):
        solve_hot_trace_metals(atmosphere, background, {"Si": -7.}, wave, **options)
    with pytest.warns(TraceMetalConvergenceWarning):
        result = solve_hot_trace_metals(atmosphere, background, {"Si": -7.}, wave,
                                        require_convergence=False, **options)
    assert not result.converged and result.population_defect > 1e-3

def test_trace_limit_rejects_bulk_mixture(atmosphere, atoms):
    with pytest.raises(ValueError, match="trace limits"):
        solve_hot_trace_metals(atmosphere, thermal_background(atmosphere), {"C": -1.},
                               [1174., 1177.], atomic_database=atoms[0], photoionization_database=atoms[1])

def test_lte_da_is_not_accepted_as_nlte_host():
    from types import SimpleNamespace
    from wd_spectra.models.stellar import DAConfig
    with pytest.raises(ValueError, match="DAO result"):
        synthesize_dao_trace_metals(SimpleNamespace(config=DAConfig()), {"C": -7.}, [1174.,1177.])


def test_acceleration_conserves_particles_and_matches_picard(atmosphere, atoms):
    wave = np.linspace(1393., 1404., 201)
    background = thermal_background(atmosphere)
    options = dict(atomic_database=atoms[0], photoionization_database=atoms[1],
                   levels_per_charge={"Si": {2: 3, 3: 3, 4: 1}},
                   maximum_iterations=120, tolerance=1e-5, n_angle=2)
    accelerated = solve_hot_trace_metals(atmosphere, background, {"Si": -10.}, wave,
                                        acceleration_depth=6, **options)
    plain = solve_hot_trace_metals(atmosphere, background, {"Si": -10.}, wave,
                                  acceleration_depth=0, **options)
    assert accelerated.converged and plain.converged
    assert accelerated.metadata['accelerated_updates'] > 0
    assert accelerated.metadata['maximum_particle_conservation_error'] < 1e-12
    np.testing.assert_allclose(accelerated.populations['Si'].population_density,
                               plain.populations['Si'].population_density, rtol=5e-5)
    np.testing.assert_allclose(accelerated.spectrum.surface_flux_lambda,
                               plain.spectrum.surface_flux_lambda, rtol=1e-6)
    assert np.min(accelerated.spectrum.surface_flux_lambda /
                   accelerated.background_spectrum.surface_flux_lambda) < 1.


def test_host_adapter_preserves_angular_quadrature_and_rejects_stale_state(atmosphere, monkeypatch):
    from dataclasses import replace
    from types import SimpleNamespace
    from wd_spectra.hot_nlte import HotPopulationState
    from wd_spectra.models.hot import DAOConfig
    from wd_spectra.multilevel_nlte import atmosphere_structure_fingerprint
    background = thermal_background(atmosphere)
    host = SimpleNamespace(config=DAOConfig(50000.,7.75,quality='production'),
        atmosphere=atmosphere, metadata={'atmosphere_convergence_status':'converged'},
        population_state=HotPopulationState(None, SimpleNamespace(metadata={
            'atmosphere_structure_sha256':atmosphere_structure_fingerprint(atmosphere)}),converged=True))
    model = SimpleNamespace(n_angle=4, transfer_coefficients=lambda a,w,s: background(w))
    monkeypatch.setattr('wd_spectra.models.hot._model_from_config', lambda *args:model)
    wave=np.linspace(1174.,1177.,41)
    result=synthesize_dao_trace_metals(host,{},wave)
    expected=transfer_field(atmosphere,background(wave),n_angle=4)[1].interface_flux[:,0]
    np.testing.assert_array_equal(result.spectrum.surface_flux_lambda,expected)
    host.atmosphere=replace(atmosphere,temperature=atmosphere.temperature*1.01)
    with pytest.raises(ValueError,match='different atmosphere'):
        synthesize_dao_trace_metals(host,{},wave)


def _large_alpha(temperature):
    # Far above any explicit-atom radiative recombination, so the top-up is active.
    return np.full_like(np.asarray(temperature, dtype=float), 1e-10)


def test_total_recombination_top_up_preserves_planck_lte(atmosphere, atoms):
    database, photo = atoms
    counts = {2: 8, 3: 3, 4: 1}
    reference = fixed_electron_metal_reference(atmosphere, database, {"C": -7.})
    wave = reduced_light_metal_wavelength(database, "C", counts, n_continuum_wavelength=120)
    intensity = planck_lambda_angstrom(wave[:, None], atmosphere.temperature[None, :])
    state = solve_reduced_light_metal_levels_nlte(
        atmosphere, database, reference, photo, wave, intensity, "C", counts,
        total_recombination_rate_coefficients={3: _large_alpha, 4: _large_alpha})
    np.testing.assert_allclose(state.population_density, state.lte_population_density, rtol=3e-6)
    top = state.metadata["recombination_top_up"]
    assert set(top) == {3, 4} and top[3]["explicit_fraction_maximum"] < 1.0


def test_total_recombination_top_up_recombines_in_strong_field(atmosphere, atoms):
    database, photo = atoms
    counts = {2: 8, 3: 3, 4: 1}
    reference = fixed_electron_metal_reference(atmosphere, database, {"C": -7.})
    wave = reduced_light_metal_wavelength(database, "C", counts, n_continuum_wavelength=120)
    intensity = 3.0 * planck_lambda_angstrom(wave[:, None], atmosphere.temperature[None, :])
    plain = solve_reduced_light_metal_levels_nlte(
        atmosphere, database, reference, photo, wave, intensity, "C", counts)
    same = solve_reduced_light_metal_levels_nlte(
        atmosphere, database, reference, photo, wave, intensity, "C", counts,
        total_recombination_rate_coefficients={})
    np.testing.assert_array_equal(plain.population_density, same.population_density)
    topped = solve_reduced_light_metal_levels_nlte(
        atmosphere, database, reference, photo, wave, intensity, "C", counts,
        total_recombination_rate_coefficients={3: _large_alpha})
    charge = np.array([key[1] for key in plain.level_key])
    lower_plain = plain.population_density[charge == 2].sum(axis=0)
    lower_topped = topped.population_density[charge == 2].sum(axis=0)
    assert np.all(lower_topped > lower_plain)
    np.testing.assert_allclose(topped.population_density.sum(axis=0),
                               plain.population_density.sum(axis=0), rtol=1e-12)


def test_total_recombination_requires_both_stages(atmosphere, atoms):
    database, photo = atoms
    counts = {2: 8, 3: 3, 4: 1}
    reference = fixed_electron_metal_reference(atmosphere, database, {"C": -7.})
    wave = reduced_light_metal_wavelength(database, "C", counts, n_continuum_wavelength=120)
    intensity = planck_lambda_angstrom(wave[:, None], atmosphere.temperature[None, :])
    with pytest.raises(ValueError):
        solve_reduced_light_metal_levels_nlte(
            atmosphere, database, reference, photo, wave, intensity, "C", counts,
            total_recombination_rate_coefficients={2: _large_alpha})


def _thick_case(atoms, **options):
    atmosphere = gray_hydrogen_atmosphere(50000., 7.75, n_depth=12, tau_max=100.)
    wave = np.linspace(1170., 1410., 401)
    common = dict(atomic_database=atoms[0], photoionization_database=atoms[1],
                  levels_per_charge={"C": {2: 4, 3: 3, 4: 1}, "Si": {2: 3, 3: 3, 4: 1}},
                  maximum_iterations=200, n_angle=2, require_convergence=False)
    common.update(options)
    return solve_hot_trace_metals(atmosphere, thermal_background(atmosphere),
                                  {"C": -5., "Si": -5.}, wave, **common)


@pytest.mark.canary
def test_accelerated_lambda_changes_path_not_fixed_point(atoms):
    plain = _thick_case(atoms, tolerance=1e-6, acceleration_depth=6)
    ali = _thick_case(atoms, tolerance=1e-6, acceleration_depth=6, accelerated_lambda=True)
    assert plain.converged and ali.converged
    assert ali.metadata["accelerated_lambda"] and ali.iterations < plain.iterations
    for element in ("C", "Si"):
        np.testing.assert_allclose(ali.populations[element].population_density,
                                   plain.populations[element].population_density, rtol=2e-4)
    np.testing.assert_allclose(ali.spectrum.surface_flux_lambda,
                               plain.spectrum.surface_flux_lambda, rtol=1e-5)


def test_cross_element_overlaps_pairs_only_different_elements(atoms):
    from wd_spectra.hot_trace_metals import _cross_element_overlaps
    lines = {("S", 2, 3, 21): 702.779, ("S", 2, 3, 20): 702.818, ("O", 2, 2, 12): 702.838,
             ("C", 2, 1, 5): 977.02, ("C", 2, 2, 6): 977.03}
    # 15 km/s at 702.8 A is 0.035 A: O III reaches S III 702.818 only.
    assert _cross_element_overlaps(lines, 15.) == {("S", 2, 3, 20), ("O", 2, 2, 12)}
    assert _cross_element_overlaps(lines, 30.) == {("S", 2, 3, 21), ("S", 2, 3, 20), ("O", 2, 2, 12)}
    with pytest.raises(ValueError):
        _thick_case(atoms, mali_overlap_velocity=0.)


@pytest.mark.canary
def test_mali_overlap_exclusion_changes_path_not_fixed_point(atoms):
    ali = _thick_case(atoms, tolerance=1e-6, acceleration_depth=6, accelerated_lambda=True)
    # A velocity spanning the whole window withholds preconditioning from every
    # line: the plain iteration's fixed point must be recovered.
    excluded = _thick_case(atoms, tolerance=1e-6, acceleration_depth=6, accelerated_lambda=True,
                           mali_overlap_velocity=1e5)
    assert ali.converged and excluded.converged
    assert ali.metadata["mali_overlap_excluded_lines"] == 0
    assert excluded.metadata["mali_overlap_excluded_lines"] > 0
    for element in ("C", "Si"):
        np.testing.assert_allclose(excluded.populations[element].population_density,
                                   ali.populations[element].population_density, rtol=2e-4)


@pytest.mark.canary
def test_opacity_criterion_tracks_tight_population_solution(atoms):
    reference = _thick_case(atoms, tolerance=1e-8, acceleration_depth=6, accelerated_lambda=True)
    assert reference.converged
    previous = np.inf
    for tolerance in (1e-3, 1e-5):
        result = _thick_case(atoms, tolerance=tolerance, acceleration_depth=6,
                             accelerated_lambda=True, convergence_criterion="opacity")
        assert result.converged
        assert result.metadata["opacity_defect_history"][-1] < tolerance
        error = np.max(abs(result.spectrum.surface_flux_lambda
                           / reference.spectrum.surface_flux_lambda - 1))
        assert error < 10 * tolerance and error < previous
        previous = error
    with pytest.raises(ValueError):
        _thick_case(atoms, tolerance=1e-3, convergence_criterion="spectrum")
