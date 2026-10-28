"""Fixed-state paper-spectrum checks; these never certify a cold atmosphere."""

import json
import sys
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
from wd_spectra import DAZConfig, compute_daz
from wd_spectra.models import load_atmosphere_checkpoint, AtmosphereConvergenceWarning

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))
from uv_atomic_control_legacy import legacy_uv_atomic_controls
from independent_kurucz_control_legacy import previous_kurucz_defaults
from montreal_h_radius_control_legacy import previous_hydrogen_neutral_defaults

pytestmark = pytest.mark.spectral
CONTROLS = Path(__file__).parent / "data/daz_regressions"
APPROVED = Path(__file__).parent / "data/approved_regressions/fixed"
APPROVED_KURUCZ_STARK = (
    Path(__file__).parent / "data/approved_regressions/kurucz_stark_2026_10_11/fixed"
)
APPROVED_UV_ATOMIC = (
    Path(__file__).parent / "data/approved_regressions/uv_atomic_2026_10_12/fixed"
)
APPROVED_SI_KURUCZ = (
    Path(__file__).parent / "data/approved_regressions/si_kurucz_2026_10_12/fixed"
)
APPROVED_INDEPENDENT_KURUCZ = (
    Path(__file__).parent / "data/approved_regressions/independent_kurucz_2026_10_14/fixed"
)
# Reviewed 2026-10-27 hydrogen neutral defaults (radius x0.5, Ali--Griem at
# every temperature); see that directory's manifest.
APPROVED_MONTREAL_H_RADIUS = (
    Path(__file__).parent / "data/approved_regressions/montreal_h_radius_2026_10_27/daz"
)
MONTREAL_H_RADIUS_CASES = frozenset({"g149_28"})


@pytest.mark.parametrize("case", ["g149_28", "galex1931"])
def test_daz_paper_spectrum(case, monkeypatch):
    path = CONTROLS / (case + ".npz")
    with np.load(path) as saved:
        config = DAZConfig(**json.loads(str(saved["config_json"])))
        wave = saved["wavelength"]
    montreal = case in MONTREAL_H_RADIUS_CASES
    with np.load(APPROVED_INDEPENDENT_KURUCZ / ("daz-" + case + ".npz")) as approved:
        np.testing.assert_array_equal(wave, approved["wavelength"])
        expected = independent_kurucz_flux = approved["surface_flux"]
    if montreal:
        with np.load(APPROVED_MONTREAL_H_RADIUS / ("daz-" + case + ".npz")) as approved:
            np.testing.assert_array_equal(wave, approved["wavelength"])
            expected = approved["surface_flux"]
    with np.load(APPROVED_KURUCZ_STARK / ("daz-" + case + ".npz")) as stark_control:
        np.testing.assert_array_equal(wave, stark_control["wavelength"])
        stark_flux = stark_control["surface_flux"]
    with np.load(APPROVED / ("daz-" + case + ".npz")) as previous:
        np.testing.assert_array_equal(wave, previous["wavelength"])
        previous_flux = previous["surface_flux"]
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
    assert np.all(np.isfinite(actual)) and np.all(actual > 0)
    assert result.spectrum.metadata["source_converged"]
    # Same significant-flux regression bound as the other immutable controls.
    significant = wave * expected > 0.01 * np.max(wave * expected)
    np.testing.assert_allclose(
        actual[significant], expected[significant], rtol=1e-3, atol=0
    )
    # Every earlier control predates the 2026-10-27 hydrogen neutral defaults:
    # reload the structure EOS under the previous defaults and recover the
    # immediately preceding control before the older chain.
    with previous_hydrogen_neutral_defaults() if montreal else nullcontext():
        if montreal:
            atmosphere = load_atmosphere_checkpoint(
                path,
                config.effective_temperature,
                config.logg,
                "hydrogen",
                include_molecules=molecules,
                include_negative_hydrogen=molecules,
                trihydrogen_ion_partition_model="neale-tennyson-1995",
            )
            with pytest.warns(AtmosphereConvergenceWarning):
                previous_defaults = compute_daz(
                    config, wave, initial_atmosphere=atmosphere, relax_atmosphere=False
                )
            kurucz_significant = wave * independent_kurucz_flux > 0.01 * np.max(
                wave * independent_kurucz_flux
            )
            np.testing.assert_allclose(
                previous_defaults.spectrum.surface_flux_lambda[kurucz_significant],
                independent_kurucz_flux[kurucz_significant], rtol=1e-3, atol=0,
            )
        with previous_kurucz_defaults(), pytest.warns(AtmosphereConvergenceWarning):
            predecessor = compute_daz(config, wave, initial_atmosphere=atmosphere,
                                      relax_atmosphere=False)
        with np.load(APPROVED_UV_ATOMIC / ("daz-" + case + ".npz")) as control:
            previous_default_flux = control["surface_flux"]
        previous_significant = wave * previous_default_flux > 0.01 * np.max(wave * previous_default_flux)
        np.testing.assert_allclose(predecessor.spectrum.surface_flux_lambda[previous_significant],
                                  previous_default_flux[previous_significant], rtol=1e-3, atol=0)
        with legacy_uv_atomic_controls():
            # Recover the immediate Si-Kurucz predecessor before the older chain.
            with pytest.warns(AtmosphereConvergenceWarning):
                legacy_atomic = compute_daz(
                    config, wave, initial_atmosphere=atmosphere,
                    relax_atmosphere=False,
                )
            with np.load(APPROVED_SI_KURUCZ / ("daz-" + case + ".npz")) as control:
                si_flux = control["surface_flux"]
                np.testing.assert_array_equal(wave, control["wavelength"])
            si_significant = wave * si_flux > 0.01 * np.max(wave * si_flux)
            np.testing.assert_allclose(
                legacy_atomic.spectrum.surface_flux_lambda[si_significant],
                si_flux[si_significant], rtol=1e-3, atol=0,
            )
            # Disabling only the default Kurucz Si II missing transitions must recover
            # the previous (default Kurucz electron Stark) control to the same bound.
            stark_config = replace(config, si_ii_kurucz_transitions=False)
            with pytest.warns(AtmosphereConvergenceWarning):
                stark_result = compute_daz(
                    stark_config,
                    wave,
                    initial_atmosphere=atmosphere,
                    relax_atmosphere=False,
                )
            stark_significant = wave * stark_flux > 0.01 * np.max(wave * stark_flux)
            np.testing.assert_allclose(
                stark_result.spectrum.surface_flux_lambda[stark_significant],
                stark_flux[stark_significant], rtol=1e-3, atol=0,
            )
            # Disabling the Kurucz electron Stark constants as well must recover the
            # earlier control to the same bound.
            with pytest.warns(AtmosphereConvergenceWarning):
                previous_result = compute_daz(
                    replace(stark_config, kurucz_electron_stark=False),
                    wave,
                    initial_atmosphere=atmosphere,
                    relax_atmosphere=False,
                )
            previous_actual = previous_result.spectrum.surface_flux_lambda
            previous_significant = wave * previous_flux > 0.01 * np.max(wave * previous_flux)
            np.testing.assert_allclose(
                previous_actual[previous_significant], previous_flux[previous_significant],
                rtol=1e-3, atol=0,
            )
