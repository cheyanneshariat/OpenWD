"""A real g7 transfer row must retain nonnegative source and scalar closure."""
import json
from pathlib import Path
import numpy as np
import pytest
from wd_spectra._mass_feautrier import mass_field


def fixture():
    row=json.loads((Path(__file__).parent/'data/transfer_regressions/db5000-g7-source-roundoff.json').read_text())
    arrays={k:np.asarray(row[k],dtype=float)[None,:] for k in ('tau','planck','absorption','scattering')}
    return arrays,np.asarray(row['column_mass'],dtype=float),row['n_angle']


def test_native_positive_reconstruction_closes_real_dense_source():
    a,m,angles=fixture()
    source,field=mass_field(a['tau'],a['planck'],a['absorption'],a['scattering'],
                            column_mass=m,n_angle=angles,reconstruct_intensity=True)
    assert np.all(np.isfinite(source)) and np.all(source>=0)
    _,independent=mass_field(a['tau'],source,a['absorption']+a['scattering'],
                            np.zeros_like(a['scattering']),column_mass=m,n_angle=angles,
                            reconstruct_intensity=True)
    rhs=(a['absorption']*a['planck']+a['scattering']*independent.mean_intensity)/(a['absorption']+a['scattering'])
    assert np.max(abs(source-rhs))/np.max(source)<1e-10


def test_reconstruction_does_not_accept_negative_physical_input():
    a,m,angles=fixture();a['planck']=a['planck'].copy();a['planck'][0,-1]=-1.
    with pytest.raises(ValueError,match='nonnegative'):
        mass_field(a['tau'],a['planck'],a['absorption'],a['scattering'],
                   column_mass=m,n_angle=angles,reconstruct_intensity=True)


def test_dense_synthesis_accepts_the_saved_roundoff_row(monkeypatch):
    """Exercise synthesis and its independent scalar check, not just the kernel."""
    from dataclasses import dataclass
    from wd_spectra._cool import dense_direct_spectrum as synthesis

    @dataclass
    class Atmosphere:
        column_mass: np.ndarray
        temperature: np.ndarray
        metadata: dict

    @dataclass
    class Spectrum:
        wavelength_angstrom: np.ndarray
        surface_flux_lambda: np.ndarray
        metadata: dict
        bolometric_flux: float = 0.

    @dataclass
    class Result:
        atmosphere: Atmosphere
        spectrum: Spectrum
        metadata: dict

    arrays, mass, angles = fixture()
    atmosphere = Atmosphere(mass, np.full(mass.size, 5000.), {
        'experimental_dense_helium': True,
        'experimental_mass_conservative_transfer': True,
    })
    result = Result(atmosphere, Spectrum(np.array([102.03570935337449]),
                                       np.zeros(1), {}), {})
    monkeypatch.setattr(synthesis, 'pure_helium_opacities',
                        lambda atmosphere, wave: (arrays['absorption'], arrays['scattering']))
    monkeypatch.setattr(synthesis, 'optical_depth_from_mass_opacity',
                        lambda mass, extinction: arrays['tau'])
    monkeypatch.setattr(synthesis, 'planck_lambda_angstrom',
                        lambda wave, temperature: arrays['planck'])
    output = synthesis.direct_result(result, angles, mass_conservative=True)
    assert np.all(np.isfinite(output.spectrum.surface_flux_lambda))
    assert np.all(output.spectrum.surface_flux_lambda >= 0)
    assert output.spectrum.metadata['experimental_positive_intensity_reconstruction'] is True
    assert output.spectrum.metadata['experimental_spectrum_radiation_scale_source_error'] < 1e-10
    np.testing.assert_array_equal(output.atmosphere.temperature, atmosphere.temperature)
    np.testing.assert_array_equal(output.atmosphere.column_mass, atmosphere.column_mass)
