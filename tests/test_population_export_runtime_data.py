"""Population exports retain physical arrays without expanding runtime atoms."""
from dataclasses import replace
import json

import numpy as np

from wd_spectra.atmosphere import gray_helium_atmosphere
from wd_spectra.metals import (
    AtomicDatabase, AtomicIon, AtomicLevel, AtomicTransition, MetalLTEState,
)
from wd_spectra.models.common import ModelResult, _jsonable, save_model_result
from wd_spectra.models.pg1159 import PG1159Config, PG1159PopulationResult
from wd_spectra.spectrum import Spectrum


class RuntimeCache(dict):
    def items(self):
        raise AssertionError('population export traversed the runtime atomic cache')


def _result(cache_lines=8, *, forbid_cache_walk=False):
    levels = tuple(AtomicLevel(i, float(i - 1) * 100., 1., f'level-{i}')
                   for i in range(1, cache_lines + 2))
    lines = tuple(AtomicTransition(1, i + 2, 1e6, 'E1', 4000. + i, .1)
                  for i in range(cache_lines))
    ion = AtomicIon('C', 0, 12., 11.26, levels, lines)
    cache = {(('C',), 3900., 5200., 1e-6): tuple((ion, line, levels[0]) for line in lines)}
    atom = AtomicDatabase({('C', 0): ion},
                          _line_selection_cache=RuntimeCache(cache) if forbid_cache_walk else cache)
    atmosphere = gray_helium_atmosphere(110000., 7., n_depth=4)
    density = np.arange(1., 5.)
    material = MetalLTEState(
        'He', {'C': 0.}, {'C': density}, {'C': density[None, :]},
        {('C', 0): np.full(4, 2.)}, density, density, atomic_database=atom)
    formal = replace(material, electron_density=2. * density,
                     ion_number_density={'C': 3. * density[None, :]})
    state = PG1159PopulationResult(material, formal, atmosphere)
    wave = np.array([4000., 5000.])
    return ModelResult('PG1159', atmosphere, Spectrum(wave, np.ones(2), {}),
                       PG1159Config(), {'atomic_data_sha256': {'fixture': 'recorded'}}, state)


def test_json_conversion_does_not_traverse_runtime_atom():
    result = _result(forbid_cache_walk=True)
    material = result.population_state.structure
    record = _jsonable(material)
    assert 'atomic_database' not in record
    assert record['ion_number_density']['C'] == material.ion_number_density['C'].tolist()
    assert material.atomic_database is not None
    assert len(material.atomic_database._line_selection_cache) == 1


def test_nested_pg1159_export_preserves_all_physical_arrays_and_provenance(tmp_path):
    result = _result(forbid_cache_walk=True)
    structure = result.population_state.structure
    atom = structure.atomic_database
    save_model_result(result, tmp_path)
    with np.load(tmp_path / 'populations.npz', allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['structure.ion_number_density.C'], structure.ion_number_density['C'])
        np.testing.assert_array_equal(saved['structure.electron_density'], structure.electron_density)
        np.testing.assert_array_equal(saved["structure.partition_function.('C', 0)"], structure.partition_function[('C', 0)])
        np.testing.assert_array_equal(saved['line_formation.ion_number_density.C'], 3. * structure.ion_number_density['C'])
        np.testing.assert_array_equal(saved['line_formation.electron_density'], 2. * structure.electron_density)
        np.testing.assert_array_equal(saved['line_formation_atmosphere.temperature'], result.atmosphere.temperature)
        assert all('atomic_database' not in key for key in saved.files)
        metadata = json.loads(str(saved['metadata_json']))
        assert 'atomic_database' not in metadata['structure']
        assert 'atomic_database' not in metadata['line_formation']
    assert structure.atomic_database is atom
    assert result.population_state.line_formation.atomic_database is atom
    assert len(atom._line_selection_cache) == 1
    record = json.loads((tmp_path / 'metadata.json').read_text())
    assert record['model_metadata']['atomic_data_sha256'] == result.metadata['atomic_data_sha256']


def test_export_size_is_independent_of_atomic_line_cache_size(tmp_path):
    records = []
    for count in (8, 64):
        result = _result(cache_lines=count)
        output = tmp_path / str(count)
        save_model_result(result, output)
        with np.load(output / 'populations.npz', allow_pickle=False) as saved:
            records.append(str(saved['metadata_json']))
    assert records[0] == records[1]
    assert len(records[1]) < 10000
