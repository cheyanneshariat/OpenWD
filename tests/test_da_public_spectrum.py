"""Fast, unmarked public-default DA regressions: no atmosphere iterations.

Unlike an operator-consistency test, these compare with frozen flux arrays and
NEVER select a synthesis method in the public call. The approved cubic default
has separate corrected cold-state fixtures; historical linear controls remain
untouched in spectral_regressions. The numerical tolerances are unchanged.
These are regressions, not observational or
cold-start equilibrium certificates; separate tests cover those questions.
"""

import json
from pathlib import Path
import warnings

import numpy as np
import pytest
from wd_spectra import DAConfig, compute_da
from wd_spectra._compat import trapezoid
from wd_spectra.constants import STEFAN_BOLTZMANN
from wd_spectra.models import load_atmosphere_checkpoint, AtmosphereConvergenceWarning

CONTROLS = Path(__file__).parent / "data/spectral_regressions"
DEFAULT_CONTROLS = Path(__file__).parent / "data/da_cubic_regressions"
SPECTRUM_RTOL = 1e-3  # 0.1% in significant flux, one common numerical budget.
LINE_ATOL = 1e-3  # 0.1 percentage point of local continuum, including dark cores.
WINDOWS = ((6250, 6900), (4700, 5025), (4250, 4450), (4020, 4180))


def assert_spectrum_preserved(wave, actual, expected):
    assert np.all(np.isfinite(actual)) and np.all(actual >= 0)
    significant = wave * expected > 0.01 * np.max(wave * expected)
    np.testing.assert_allclose(
        actual[significant], expected[significant], rtol=SPECTRUM_RTOL, atol=0
    )
    np.testing.assert_allclose(
        trapezoid(actual, wave), trapezoid(expected, wave), rtol=SPECTRUM_RTOL, atol=0
    )
    for lo, hi in WINDOWS:
        take = (wave >= lo) & (wave <= hi)
        x = wave[take]
        if len(x) < 8:
            raise AssertionError("A Balmer window is inadequately sampled")

        def normalized(flux):
            f = flux[take]
            left, right = x < lo + 20, x > hi - 20
            continuum = np.interp(
                x,
                [np.mean(x[left]), np.mean(x[right])],
                [np.mean(f[left]), np.mean(f[right])],
            )
            return f / continuum

        np.testing.assert_allclose(
            normalized(actual), normalized(expected), rtol=0, atol=LINE_ATOL
        )


@pytest.mark.parametrize("case", ["da-12000-public", "da-20000"])
def test_public_da_spectrum_has_not_changed(case, monkeypatch):
    from wd_spectra.models import stellar

    def forbidden(*a, **kw):
        raise AssertionError("Fast spectral regression must not converge an atmosphere")

    monkeypatch.setattr(stellar, "radiative_equilibrium_hydrogen_atmosphere", forbidden)
    path = DEFAULT_CONTROLS / (case + ".npz")
    with np.load(path) as saved:
        cfg = DAConfig(**json.loads(str(saved["config_json"])))
        wave, expected = saved["wavelength"], saved["original_surface_flux"]
    molecular = cfg.effective_temperature <= 12000
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AtmosphereConvergenceWarning)
        state = load_atmosphere_checkpoint(
            path,
            cfg.effective_temperature,
            cfg.logg,
            "hydrogen",
            include_molecules=molecular,
            include_negative_hydrogen=molecular,
            trihydrogen_ion_partition_model="neale-tennyson-1995",
        )
        # Deliberately omit synthesis_transfer. Do not bypass a
        # failing default by opting into the reference's operator here.
        result = compute_da(cfg, wave, initial_atmosphere=state, relax_atmosphere=False)
    assert result.spectrum.metadata['transfer_discretization'] == 'formal-pchip'
    assert_spectrum_preserved(wave, result.spectrum.surface_flux_lambda, expected)
    # Independent stellar-flux guard for these two broad, warm-DA controls.
    # No fitted scale enters the calculation or this check. This does not
    # claim universal bolometric accuracy for arbitrary requested bandpasses.
    ratio = trapezoid(result.spectrum.surface_flux_lambda, wave) / (
        STEFAN_BOLTZMANN * cfg.effective_temperature**4
    )
    assert abs(ratio - 1.0) < 3e-3


@pytest.mark.parametrize("damage", ["scale", "core", "wings"])
def test_spectrum_guard_rejects_material_changes(damage):
    with np.load(CONTROLS / "da-12000-public.npz") as saved:
        wave, expected = saved["wavelength"], saved["original_surface_flux"]
    actual = expected.copy()
    if damage == "scale":
        actual *= 1.01
    elif damage == "core":
        actual[abs(wave - 6564.61) < 2] *= 1.02
    else:
        actual[(abs(wave - 6564.61) > 10) & (abs(wave - 6564.61) < 80)] *= 1.01
    with pytest.raises(AssertionError):
        assert_spectrum_preserved(wave, actual, expected)
