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
