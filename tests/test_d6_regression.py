"""J1637 PR regression coverage; full cold convergence remains a separate test."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from wd_spectra import D6Config
from wd_spectra.models.common import AtmosphereConvergenceWarning
from d6_regression_support import (
    ARRAYS, METRICS, STEPS, assert_trajectory_matches, atomic_input_scope,
    fixed_spectrum, trajectory,
)

DIRECTORY = Path(__file__).parent / "data/d6_regressions/warm_v1"
SEED = DIRECTORY.with_name(DIRECTORY.name + "-seed.npz")
CURRENT_DIRECTORY = DIRECTORY.with_name("warm_v2_feii")
REFERENCES = (
    pytest.param(DIRECTORY, True, id="warm_v1-legacy"),
    pytest.param(CURRENT_DIRECTORY, False, id="warm_v2_feii-current"),
)


@pytest.mark.parametrize("directory,legacy", REFERENCES)
def test_frozen_j1637_seed_certificate_and_reference_integrity(directory, legacy):
    manifest = json.loads((directory / "manifest.json").read_text())
    path = directory / "j1637.npz"
    seed_path = directory.with_name(directory.name + "-seed.npz")
    assert manifest["schema"] == 1
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["output_sha256"]
    assert hashlib.sha256(seed_path.read_bytes()).hexdigest() == manifest["seed_sha256"]
    assert manifest["config"] == json.loads(json.dumps(asdict(D6Config())))
    assert manifest["steps"] == STEPS and manifest["metrics"] == list(METRICS)
    assert manifest["cold_start_qualification"] is False
    assert manifest["source_request"]["config"] == manifest["config"]
    certificate = manifest["source_cold_certificate"]
    assert certificate["verified"] and not certificate["failures"]
    for key in certificate["required_checks"]:
        check = certificate["checks"][key]
        assert check["measured"] and check["passed"]
        assert 0 <= check["value"] < check["tolerance"]
    if not legacy:
        previous = json.loads((DIRECTORY / "manifest.json").read_text())
        for key in (
            "source_name", "source_sha256", "source_cold_certificate",
            "source_request", "config", "perturbation", "steps", "metrics",
        ):
            assert manifest[key] == previous[key]
        assert manifest["physics_revision"] != previous["physics_revision"]
        assert "default-feii-kurucz-uv-2026-10-12" in manifest["physics_revision"].split(":")
        archive = manifest["archived_reference"]
        assert archive["name"] == DIRECTORY.name
        assert archive["manifest_sha256"] == hashlib.sha256(
            (DIRECTORY / "manifest.json").read_bytes()).hexdigest()
        for key in ("output_sha256", "seed_sha256", "reference_commit"):
            assert archive[key] == previous[key]
        recovery = manifest["legacy_recovery"]
        assert recovery["passed"] is True and recovery["tolerances_changed"] is False
        assert recovery["reference_output_sha256"] == previous["output_sha256"]
        with np.load(seed_path, allow_pickle=False) as current_seed:
            with np.load(SEED, allow_pickle=False) as previous_seed:
                assert set(current_seed.files) == set(previous_seed.files)
                for key in previous_seed.files:
                    if key == "initial_temperature":
                        # The Linux capture reevaluated the prescribed exp
                        # perturbation: one layer differs from macOS by one
                        # float64 ULP. Keep both raw archives unchanged.
                        np.testing.assert_array_max_ulp(
                            current_seed[key], previous_seed[key], maxulp=1)
                    else:
                        np.testing.assert_array_equal(current_seed[key], previous_seed[key])
    with np.load(path, allow_pickle=False) as saved:
        with np.load(seed_path, allow_pickle=False) as seed:
            for key in seed.files:
                np.testing.assert_array_equal(seed[key], saved[key])
        assert saved["base_temperature"].shape == (48,)
        for key in ARRAYS:
            assert saved[key].shape == (STEPS, 48)
            assert np.all(np.isfinite(saved[key])) and np.all(saved[key] > 0)
        assert saved["metrics"].shape == (STEPS, len(METRICS))
        assert np.all(np.isfinite(saved["metrics"]))
        perturbation = 0.015 * np.exp(-((np.log10(saved["optical_depth"]) + 2) / 1)**2)
        np.testing.assert_allclose(saved["initial_temperature"],
            saved["base_temperature"] * np.exp(perturbation), rtol=1e-14)
        assert np.max(np.abs(saved["temperature"][0] / saved["initial_temperature"] - 1)) > 1e-4


@pytest.mark.canary
@pytest.mark.parametrize("directory,legacy", REFERENCES)
def test_j1637_two_step_warm_trajectory(directory, legacy):
    seed_path = directory.with_name(directory.name + "-seed.npz")
    with atomic_input_scope(legacy=legacy):
        with np.load(seed_path, allow_pickle=False) as seed:
            records = trajectory(seed["column_mass"], seed["optical_depth"],
                                 seed["initial_temperature"])
    with np.load(directory / "j1637.npz", allow_pickle=False) as saved:
        assert_trajectory_matches(records, saved)


@pytest.mark.spectral
def test_j1637_fixed_atmosphere_spectrum(monkeypatch):
    import wd_spectra.models.d6 as public

    def forbidden_solve(*args, **kwargs):
        pytest.fail("fixed-atmosphere D6 spectrum started an atmosphere solve")

    monkeypatch.setattr(public, "radiative_equilibrium_d6_atmosphere", forbidden_solve)
    with np.load(DIRECTORY / "j1637.npz", allow_pickle=False) as saved:
        with atomic_input_scope(), pytest.warns(AtmosphereConvergenceWarning):
            spectrum = fixed_spectrum(saved["column_mass"], saved["optical_depth"],
                                      saved["base_temperature"], saved["wavelength"])
        flux = spectrum.surface_flux_lambda
        assert np.all(np.isfinite(flux)) and np.all(flux > 0)
        np.testing.assert_allclose(flux, saved["surface_flux"], rtol=2e-6,
                                   atol=1e-12 * np.max(saved["surface_flux"]))
        assert spectrum.metadata["source_converged"]
        assert spectrum.metadata["independent_radiation_scaled_source_error"] < 1e-10
        assert spectrum.metadata["maximum_metal_lines"] == 25_000
        assert spectrum.metadata["transfer_discretization"] == "formal-linear"
