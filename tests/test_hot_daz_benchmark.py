"""Small synthetic archive checks; no network or external observations needed."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('hot_daz_benchmark',
    Path(__file__).resolve().parents[1]/'research/hot_daz_benchmark.py')
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def test_archive_reader_rejects_air_and_bad_flux_errors(tmp_path):
    fits = pytest.importorskip('astropy.io.fits')
    path = tmp_path/'test.fits'
    primary = fits.PrimaryHDU()
    primary.header['AIRORVAC'] = 'AIR'
    table = fits.BinTableHDU.from_columns([
        fits.Column(name='WAVE', format='D', array=[1174.,1175.,1176.,1177.]),
        fits.Column(name='FLUX', format='D', array=[1.,np.nan,2.,3.]),
        fits.Column(name='ERROR', format='D', array=[.1,.1,0.,.2])])
    fits.HDUList([primary,table]).writeto(path)
    with pytest.raises(ValueError,match='vacuum'):
        benchmark.read_observation(path)
    primary.header['AIRORVAC'] = 'VAC'
    fits.HDUList([primary,table]).writeto(path,overwrite=True)
    w,f,e=benchmark.read_observation(path)
    np.testing.assert_array_equal(w,[1174.,1177.])
    np.testing.assert_array_equal(f,[1.,3.])
    np.testing.assert_array_equal(e,[.1,.2])


def test_checksum_mismatch_does_not_silently_accept_archive(tmp_path):
    for rel,_ in benchmark.FILES.values():
        (tmp_path/Path(rel).name).write_bytes(b'corrupt data')
    with pytest.raises(ValueError,match='checksum mismatch'):
        benchmark.checked_files(tmp_path)


def test_catalogue_preserves_blends_instead_of_summing_duplicate_features(tmp_path):
    path=tmp_path/'lines.txt'
    path.write_text('# metadata\n1175.024 1.0 24.663 1.397 "C" "III" 1174.933 1.2 23.13 .26 .40 "PHOT"\n'
                    '1175.024 1.0 24.663 1.397 "Fe" "IV" 1174.932 8.7 23.38 .26 2.23 "PHOT"\n')
    rows=benchmark.diagnostic_lines(path)
    assert [r['element'] for r in rows]==['C','Fe']
    assert rows[0]['observed_wavelength']==rows[1]['observed_wavelength']
