"""Frozen paper spectra and separate, genuinely cold public DAH calculations."""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from wd_spectra import DAHConfig, compute_dah
from wd_spectra.models.common import (
    AtmosphereConvergenceWarning, load_atmosphere_checkpoint, save_model_result,
)


DATA = Path(__file__).parent / "data/dah_paper"
TARGETS = json.loads((DATA / "manifest.json").read_text())["targets"]


def paper_config(key):
    """Specify only stellar parameters and resolution; exercise public defaults."""
    original = TARGETS[key]["config"]
    names = ("effective_temperature", "logg", "magnetic_field_megagauss",
             "quality", "field_geometry", "dipole_inclination_deg", "dipole_offset_radius")
    values = {name: original[name] for name in names}
    values["dipole_offset_radius"] = tuple(values["dipole_offset_radius"])
    return DAHConfig(**values)


@pytest.mark.spectral
@pytest.mark.parametrize("key", TARGETS)
def test_public_default_reproduces_frozen_paper_spectrum(key, monkeypatch):
    from wd_spectra.models import dah

    def forbidden(*args, **kwargs):
        pytest.fail("fixed-spectrum test attempted an atmosphere solve")

    monkeypatch.setattr(dah, "compute_da", forbidden)
    monkeypatch.setattr(dah, "radiative_equilibrium_hydrogen_atmosphere", forbidden)
    config = paper_config(key)
    path = DATA / TARGETS[key]["reference_file"]
    atmosphere = load_atmosphere_checkpoint(
        path, config.effective_temperature, config.logg, "hydrogen",
        include_molecules=False, include_negative_hydrogen=False,
        trihydrogen_ion_partition_model="neale-tennyson-1995",
    )
    with np.load(path) as reference:
        with pytest.warns(AtmosphereConvergenceWarning):
            result = compute_dah(config, reference["wavelength"],
                                 initial_atmosphere=atmosphere, relax_atmosphere=False)
        # No continuum fitting, velocity fitting or flux renormalization.
        np.testing.assert_allclose(result.spectrum.surface_flux_lambda,
                                   reference["flux"], rtol=2e-8, atol=0.)
    assert result.metadata["balmer_profile"] == "kurucz-griem"
    assert result.metadata["normalize_balmer_strength"] is True
    assert result.metadata["structure_field_megagauss"] == 0.
    assert result.metadata["atmosphere_convergence_status"] != "converged"


@pytest.mark.canary
@pytest.mark.parametrize("key", ["j1007+1237", "j1254+5612"])
def test_dah_public_default_cold_start(key):
    config = paper_config(key)
    # Only the final wavelength grid and frozen spectrum are read. The saved
    # atmosphere arrays never enter this calculation.
    with np.load(DATA / TARGETS[key]["reference_file"]) as reference:
        wave, expected = reference["wavelength"], reference["flux"]

    def progress(iteration, atmosphere, status):
        print(key, iteration, status.get("maximum_all_depth_total_flux_residual"),
              status.get("maximum_relative_cell_energy_balance_residual"), flush=True)

    result = compute_dah(config, wave, iteration_callback=progress)
    output = os.environ.get("OPENWD_TEST_ARTIFACTS")
    if output:
        save_model_result(result, Path(output) / key)
    certificate = result.atmosphere.metadata["equilibrium_certificate"]
    assert certificate["verified"], certificate
    assert all(certificate["checks"][name]["passed"] for name in certificate["required_checks"])
    assert result.metadata["atmosphere_convergence_status"] == "converged"
    assert result.atmosphere.metadata["maximum_convective_flux_fraction"] == 0.
    assert result.metadata["equilibrium_certificate_scope"].startswith("nonmagnetic DA")
    # Independent cold structures can stop at different certified Newton
    # iterates. Require their absolute spectra to agree to 0.2 percent.
    np.testing.assert_allclose(result.spectrum.surface_flux_lambda, expected, rtol=2e-3, atol=0.)


@pytest.mark.canary
def test_mean_field_j1018_structure_converges_from_cold():
    """The coolest strong-field mean-field structure must certify at standard quality.

    Its 40-layer attempt stalls at a ~9% flux error; the refinement ladder must
    re-solve the same physics on a finer mesh and earn a fresh certificate.
    """
    from wd_spectra.validation.magnetic_da import DAH_VALIDATION_BY_KEY

    target = DAH_VALIDATION_BY_KEY["j1018+0111"]
    config = DAHConfig(
        **target.config_kwargs(), quality="standard",
        atmosphere_structure="mean-field", balmer_profile="unified",
        normalize_balmer_strength=False, include_magnetic_eos=True,
        include_rwa_photoionization=True, include_centered_motion=True,
    )
    result = compute_dah(config, np.array([4000.0, 5000.0]))
    metadata = result.atmosphere.metadata
    assert metadata["equilibrium_certificate"]["verified"], metadata["equilibrium_certificate"]
    assert result.metadata["atmosphere_convergence_status"] == "converged"
    assert result.metadata["structure_field_megagauss"] > 1.0
    assert result.atmosphere.effective_temperature == target.effective_temperature
