from types import SimpleNamespace
from dataclasses import dataclass
import numpy as np
import pytest
from wd_spectra._cool.dense_mesh_hierarchy import nested_indices, hierarchy_solve


@dataclass
class Result:
    gas_pressure: np.ndarray
    temperature: np.ndarray
    metadata: dict


def test_nested_grids_keep_exact_endpoints_and_parent_nodes():
    for size in (80, 159, 160, 317, 633):
        levels = nested_indices(size)
        assert len(levels[0]) <= 80
        for indices in levels:
            assert indices[0] == 0 and indices[-1] == size-1
        for coarse, fine in zip(levels, levels[1:]):
            assert set(coarse) <= set(fine)
    assert [len(i) for i in nested_indices(317,41)] == [41,80,159,317]


@pytest.mark.parametrize('success', [False, True])
def test_hierarchy_recertifies_each_grid_without_failed_coarse_substitution(success):
    calls = []
    p = np.geomspace(1, 100, 317)
    def solve(*args, **options):
        calls.append(options)
        return Result(options['initial_gas_pressure'], options['initial_temperature'],
            dict(radiative_equilibrium_converged=success,
                 maximum_all_depth_total_flux_residual=1e-8,
                 maximum_relative_cell_energy_balance_residual=1e-6,
                 radiative_equilibrium_maximum_log_temperature_correction=1e-5))
    options = dict(initial_gas_pressure=p, initial_column_mass=p/1e8,
                   initial_temperature=4000*p**.2, initial_rosseland_optical_depth=p,
                   n_depth=317)
    if not success:
        with pytest.raises(RuntimeError, match='no coarse-result substitution'):
            hierarchy_solve(solve, 5000, 8, **options)
        assert len(calls) == 1
    else:
        result = hierarchy_solve(solve, 5000, 8, **options)
        assert [c['n_depth'] for c in calls] == [80, 159, 317]
        assert [c['resume_supplied_structure_in_formal_flux_phase'] for c in calls] == [False, True, True]
        np.testing.assert_allclose(result.temperature, options['initial_temperature'], rtol=1e-14)
        assert len(result.metadata['experimental_mesh_hierarchy']) == 3


def test_hierarchy_does_not_trust_a_convergence_label_without_physical_checks():
    p=np.geomspace(1,10,80)
    def solve(*args,**options):
        return Result(p,options['initial_temperature'],dict(
            radiative_equilibrium_converged=True,maximum_all_depth_total_flux_residual=.1,
            maximum_relative_cell_energy_balance_residual=1e-6,
            radiative_equilibrium_maximum_log_temperature_correction=1e-5))
    with pytest.raises(RuntimeError,match='solve failed'):
        hierarchy_solve(solve,5000,8,initial_temperature=p**.2*4000,
            initial_gas_pressure=p,initial_column_mass=p/1e8,initial_rosseland_optical_depth=p)


@pytest.mark.parametrize('finest_passes',[False,True])
def test_explicit_incomplete_coarse_stages_never_replace_finest_convergence(finest_passes):
    p=np.geomspace(1,10,159)
    def solve(*args,**options):
        success=finest_passes and options['n_depth']==159
        return Result(options['initial_gas_pressure'],options['initial_temperature'],dict(
            radiative_equilibrium_converged=success,maximum_all_depth_total_flux_residual=1e-5 if success else .1,
            maximum_relative_cell_energy_balance_residual=1e-5,
            radiative_equilibrium_maximum_log_temperature_correction=1e-5))
    options=dict(initial_temperature=p**.2*4000,initial_gas_pressure=p,
        initial_column_mass=p/1e8,initial_rosseland_optical_depth=p)
    if finest_passes:
        result=hierarchy_solve(solve,5000,8,allow_incomplete_coarse=True,**options)
        history=result.metadata['experimental_mesh_hierarchy']
        assert not history[0]['converged'] and history[-1]['converged']
    else:
        with pytest.raises(RuntimeError,match='159-layer solve failed'):
            hierarchy_solve(solve,5000,8,allow_incomplete_coarse=True,**options)
