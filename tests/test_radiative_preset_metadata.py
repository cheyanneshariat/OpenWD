"""Public result construction after stubbed DB/DAB structure and synthesis.

These tests check API wiring and metadata, not numerical convergence.
"""

from dataclasses import asdict

import numpy as np
import pytest

from wd_spectra.atmosphere import gray_helium_atmosphere
from wd_spectra.models import stellar
from wd_spectra.spectrum import Spectrum


@pytest.mark.parametrize("family", ["DB", "DAB"])
@pytest.mark.parametrize("alpha", [None, 1.25])
def test_radiative_preset_returns_result_after_synthesis(monkeypatch, family, alpha):
    calls = []
    atmosphere = gray_helium_atmosphere(10000.0, 8.0, n_depth=3)
    spectrum = Spectrum(np.array([4000.0, 5000.0]), np.ones(2), {})

    def solve(*args, **kwargs):
        calls.append("structure")
        assert kwargs["mixing_length_alpha"] is alpha
        return atmosphere

    def synthesize(current, wave, **kwargs):
        calls.append("synthesis")
        return spectrum

    monkeypatch.setattr(stellar, "_helium_tables", lambda data: (None, None))
    monkeypatch.setattr(
        stellar, "_allard_lyman_profiles_for_effective_temperature",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        stellar, "model_request_fingerprint",
        lambda kind, config, data, **kwargs: {
            "spectral_type": kind, "config": asdict(config),
        },
    )
    monkeypatch.setattr(stellar, "solve_with_screened_boundary", solve)
    monkeypatch.setattr(stellar, "radiative_equilibrium_helium_atmosphere", solve)
    monkeypatch.setattr(
        stellar, "radiative_equilibrium_hydrogen_helium_atmosphere", solve
    )
    monkeypatch.setattr(stellar, "synthesize_helium_spectrum", synthesize)
    monkeypatch.setattr(stellar, "synthesize_hydrogen_helium_spectrum", synthesize)
    monkeypatch.setattr(
        stellar, "warn_if_atmosphere_not_converged", lambda *args: "unknown"
    )

    if family == "DB":
        config = stellar.DBConfig(
            effective_temperature=10000.0, mixing_length_alpha=alpha,
            transfer_discretization="optical-depth",
        )
        result = stellar.compute_db(config, spectrum.wavelength_angstrom)
    else:
        config = stellar.DABConfig(
            effective_temperature=10000.0, mixing_length_alpha=alpha,
        )
        result = stellar.compute_dab(config, spectrum.wavelength_angstrom)

    assert calls == ["structure", "synthesis"]
    assert result.spectrum is spectrum
    assert result.metadata["convection"] == (
        "suppressed; radiative equilibrium" if alpha is None else "ML2/alpha=1.25"
    )
