"""Collision mapping and channel-counting tests independent of observed fits."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'research'))
from hot_trace_collisions import map_chianti_levels, carbon_collisions
from wd_spectra.atmosphere import gray_hydrogen_atmosphere
from wd_spectra.hot_trace_metals import fixed_electron_metal_reference
from wd_spectra.light_metal_nlte import (ConstantEffectiveCollisionStrength,
    reduced_light_metal_wavelength,solve_reduced_light_metal_levels_nlte)
from wd_spectra.metals import (AtomicTransition,read_pg1159_atomic_database,
                               read_verner_photoionization_database)
from wd_spectra.models.common import ModelData
from wd_spectra.spectrum import planck_lambda_angstrom


@pytest.fixture(scope='module')
def database():
    return read_pg1159_atomic_database(ModelData.default().stout,elements=('C',))


def level_row(i,configuration,spin,orbital,j,energy):
    return f'{i:7d}{configuration:30s}{"":5s}{spin:5d}{orbital:5s}{j:5.1f}{energy:15.3f}{energy:15.3f}\n'


def test_collision_reader_renumbers_and_checks_input_identity(tmp_path,database):
    (tmp_path/'c_3.elvlc').write_text(level_row(9,'2s2',1,'S',0,0)
        +level_row(7,'2s 2p',3,'P',1,52390.75)+'-1\n')
    (tmp_path/'c_3.scups').write_text('9 7 0.48 0 0 3 2 1\n0 0.5 1\n2 2 2\n-1\n')
    manifest={p.name:{'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in tmp_path.iterdir()}
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    records,audit=carbon_collisions(tmp_path,database,{2:4},charges=(2,))
    assert set(records)=={('C',2,1,3)}
    assert records['C',2,1,3].effective_collision_strength(50000.)==pytest.approx(2.)
    assert audit['2']['selected_pairs']==1
    (tmp_path/'c_3.scups').write_text('changed data')
    with pytest.raises(ValueError,match='checksum'):carbon_collisions(tmp_path,database,{2:4},charges=(2,))


def test_mapping_requires_energy_and_rejects_ambiguous_levels(tmp_path,database):
    path=tmp_path/'levels'
    path.write_text(level_row(1,'2s2',1,'S',0,0)+level_row(2,'2s 2p',3,'P',1,62000)+'-1\n')
    ion=database.ions['C',2]
    mapped,_=map_chianti_levels(path,ion)
    assert mapped=={1:1}
    bad=replace(ion,levels=ion.levels+(replace(ion.levels[0],index=999),))
    with pytest.raises(ValueError,match='ambiguous'):map_chianti_levels(path,bad)


def test_oxygen_collision_reader_matches_diagnostic_levels_not_indices(tmp_path):
    from hot_trace_oxygen_collisions import oxygen_collisions
    db=read_pg1159_atomic_database(ModelData.default().stout,elements=('O',))
    # Deliberately use unrelated source indices for the O IV 1338 pair.
    (tmp_path/'o_4.elvlc').write_text(
        level_row(73,'2s 2p2',2,'P',.5,180480.8)
        +level_row(19,'2p3',2,'D',1.5,255184.9)+'-1\n')
    (tmp_path/'o_4.scups').write_text('73 19 0.68 0 0 3 2 1\n0 0.5 1\n3 3 3\n-1\n')
    manifest={p.name:{'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in tmp_path.iterdir()}
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    records,audit=oxygen_collisions(tmp_path,db,{3:13},charges=(3,))
    assert set(records)=={('O',3,9,13)}
    assert records['O',3,9,13].effective_collision_strength(50000.)==pytest.approx(3.)
    assert audit['3']['selected_pairs']==1
    with pytest.raises(ValueError,match='no selected'):
        oxygen_collisions(tmp_path,db,{3:12},charges=(3,))
    (tmp_path/'o_4.scups').write_text('changed data')
    with pytest.raises(ValueError,match='checksum'):
        oxygen_collisions(tmp_path,db,{3:13},charges=(3,))


@pytest.mark.parametrize('explicit',[False,True])
def test_two_radiative_channels_do_not_double_electron_collisions(database,explicit):
    atmosphere=gray_hydrogen_atmosphere(50000.,7.75,n_depth=4,tau_max=3.)
    ion=database.ions['C',2]
    gap=ion.levels[1].energy_wavenumber-ion.levels[0].energy_wavenumber
    line=AtomicTransition(1,2,3e5,'M1',1e8/gap,0.)
    single=replace(database,ions={**database.ions,('C',2):replace(ion,transitions=(line,))})
    doubled=replace(database,ions={**database.ions,('C',2):replace(ion,transitions=(
        replace(line,einstein_a=1.5e5),replace(line,einstein_a=1.5e5,transition_type='E2')))})
    counts={2:4,3:2,4:1}
    ref=fixed_electron_metal_reference(atmosphere,single,{'C':-7.})
    wave=reduced_light_metal_wavelength(single,'C',counts,n_continuum_wavelength=120)
    photo=read_verner_photoionization_database(ModelData.default().verner_photoionization,elements=('C',))
    collisions={('C',2,1,2):ConstantEffectiveCollisionStrength(.8,'synthetic')} if explicit else None
    def solve(db,factor):
        return solve_reduced_light_metal_levels_nlte(atmosphere,db,ref,photo,wave,
            factor*planck_lambda_angstrom(wave[:,None],atmosphere.temperature),'C',counts,
            collision_data=collisions)
    a,b=solve(single,.4),solve(doubled,.4)
    np.testing.assert_allclose(a.population_density,b.population_density,rtol=2e-11)
    thermal=solve(doubled,1.)
    np.testing.assert_allclose(thermal.population_density,thermal.lte_population_density,rtol=3e-6)
