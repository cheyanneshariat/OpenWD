"""Portable snapshot bookkeeping; these checks never load model arrays."""
import copy
import importlib.util
from pathlib import Path

import pytest


SNAPSHOT = Path(__file__).resolve().parents[1] / 'docs/grids/sdb-2026-10-25'
SPEC = importlib.util.spec_from_file_location('sdb_snapshot', SNAPSHOT/'verify_snapshot.py')
SNAPSHOT_CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SNAPSHOT_CHECK)


@pytest.fixture
def metadata():
    return SNAPSHOT_CHECK.load_metadata(SNAPSHOT)


def test_exact_coordinate_and_reference_partitions(metadata):
    assert SNAPSHOT_CHECK.validate_metadata(*metadata) == dict(
        coordinates=1573,HHe_qualified=1381,reference_qualified=696,native_files=5535)


def test_a_duplicate_coordinate_cannot_replace_a_missing_point(metadata):
    rows = copy.deepcopy(metadata[0])
    for key in ['teff_K','logg_cgs','log10_nHe_nH','log10_nH_nHe']:
        rows[1][key] = rows[0][key]
    with pytest.raises(ValueError,match='coordinate membership'):
        SNAPSHOT_CHECK.validate_metadata(rows,*metadata[1:])


def test_a_failed_reference_cannot_supply_a_fitting_spectrum(metadata):
    rows = copy.deepcopy(metadata[0])
    row = next(r for r in rows if r['reference_status']=='REFERENCE_TIMEOUT')
    row['reference_spectrum_path'] = 'invented/spectrum.npz'
    with pytest.raises(ValueError,match='Excluded model has a spectrum'):
        SNAPSHOT_CHECK.validate_metadata(rows,*metadata[1:])


def test_a_reference_mask_cannot_include_an_unqualified_host(metadata):
    masks = copy.deepcopy(metadata[2])
    masks['reference_fit_host_ids'][0] = next(r['coordinate_id'] for r in metadata[0]
        if r['HHe_status']!='Verified')
    with pytest.raises(ValueError,match='Fitting masks'):
        SNAPSHOT_CHECK.validate_metadata(metadata[0],metadata[1],masks,*metadata[3:])


def test_reference_conversion_retains_helium_dependence(metadata):
    qualification = copy.deepcopy(metadata[3])
    item = next(v['reference-metals'] for v in qualification['models'].values()
                if 'reference-metals' in v)
    item['composition']['NLTE']['C'] += .01
    with pytest.raises(ValueError,match='X/H conversion'):
        SNAPSHOT_CHECK.validate_metadata(*metadata[:3],qualification,metadata[4])


def test_saved_qualification_coordinates_must_match_index(metadata):
    qualification = copy.deepcopy(metadata[3])
    next(iter(qualification['models'].values()))['HHe']['coordinate']['teff_K'] += 2000
    with pytest.raises(ValueError,match='Qualification coordinate'):
        SNAPSHOT_CHECK.validate_metadata(*metadata[:3],qualification,metadata[4])


@pytest.mark.parametrize('name',['/absolute/file.npz','../file.npz','HHe/../../file.npz',
                                  'HHe\\file.npz','.',''])
def test_archive_member_paths_cannot_escape_the_snapshot(name):
    with pytest.raises(ValueError,match='Unsafe data path'):
        SNAPSHOT_CHECK.safe_path(name)
