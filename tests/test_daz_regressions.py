"""Fixed-state paper-spectrum checks; these never certify a cold atmosphere."""

import json
from pathlib import Path
import numpy as np
import pytest
from wd_spectra import DAZConfig, compute_daz
from wd_spectra.models import load_atmosphere_checkpoint, AtmosphereConvergenceWarning

pytestmark = pytest.mark.spectral
CONTROLS = Path(__file__).parent / "data/daz_regressions"
APPROVED = Path(__file__).parent / "data/approved_regressions/fixed"


@pytest.mark.parametrize("case", ["g149_28", "galex1931"])
def test_daz_paper_spectrum(case, monkeypatch):
    path = CONTROLS / (case + ".npz")
    with np.load(path) as saved:
        config = DAZConfig(**json.loads(str(saved["config_json"])))
        wave = saved["wavelength"]
    with np.load(APPROVED / ("daz-" + case + ".npz")) as approved:
        np.testing.assert_array_equal(wave, approved["wavelength"])
        expected = approved["surface_flux"]
    molecules = config.effective_temperature <= 12000
    atmosphere = load_atmosphere_checkpoint(
        path,
        config.effective_temperature,
        config.logg,
        "hydrogen",
        include_molecules=molecules,
        include_negative_hydrogen=molecules,
        trihydrogen_ion_partition_model="neale-tennyson-1995",
    )
    from wd_spectra.models import daz

    def forbidden(*args, **kwargs):
        raise AssertionError("Fixed-spectrum regression must not solve an atmosphere")

    monkeypatch.setattr(daz, "radiative_equilibrium_hydrogen_atmosphere", forbidden)
    with pytest.warns(AtmosphereConvergenceWarning):
        result = compute_daz(
            config,
            wave,
            initial_atmosphere=atmosphere,
            relax_atmosphere=False,
        )
    assert result.metadata["atmosphere_convergence_status"] != "converged"
    actual = result.spectrum.surface_flux_lambda
    assert np.all(np.isfinite(actual)) and np.all(actual >= 0)
    assert result.spectrum.metadata["source_converged"]
    # Same significant-flux regression bound as the other immutable controls.
    significant = wave * expected > 0.01 * np.max(wave * expected)
    np.testing.assert_allclose(
        actual[significant], expected[significant], rtol=1e-3, atol=0
    )
