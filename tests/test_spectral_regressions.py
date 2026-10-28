"""Broad, fixed-atmosphere controls are not equilibrium certificates.

These protect our synthetic UV, optical lines and IR against regressions.
The atmosphere/cold-start tests separately protect equilibrium diagnostics.
"""

import json
import sys
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest
from wd_spectra._compat import trapezoid
from wd_spectra.models import (
    DAConfig,
    DBConfig,
    DABConfig,
    DZConfig,
    compute_da,
    compute_db,
    compute_dab,
    compute_dz,
    load_atmosphere_checkpoint,
    AtmosphereConvergenceWarning,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))
from uv_atomic_control_legacy import legacy_uv_atomic_controls
from independent_kurucz_control_legacy import previous_kurucz_defaults
from cool_dz_release_control_legacy import (
    previous_cool_dz_release_config,
    previous_cool_dz_release_physics,
)

from metal_population_control_legacy import previous_metal_opacity_populations
from montreal_h_radius_control_legacy import previous_hydrogen_neutral_defaults

CONTROLS = Path(__file__).parent / "data/spectral_regressions"


@pytest.fixture(autouse=True)
def _restores_every_wd_spectra_binding():
    """Legacy scopes and patches must leave no wd_spectra function binding replaced.

    Autouse fixtures are set up before ``monkeypatch`` and torn down after it,
    so this sees the bindings a later test in the same process would see.
    """
    def bindings():
        return {
            (name, attr): value
            for name, module in tuple(sys.modules.items())
            if module is not None and name.startswith("wd_spectra")
            for attr, value in tuple(vars(module).items())
            if callable(value)
        }

    before = bindings()
    yield
    after = bindings()
    changed = sorted(f"{name}.{attr}" for (name, attr), value in before.items()
                     if (name, attr) in after and after[(name, attr)] is not value)
    assert not changed, f"wd_spectra bindings left patched: {changed}"
APPROVED = Path(__file__).parent / "data/approved_regressions/fixed"
APPROVED_DENSITY_EDGE = (
    Path(__file__).parent
    / "data/approved_regressions/stark_density_floor_2026_10_05/fixed"
)
APPROVED_CRII = (
    Path(__file__).parent
    / "data/approved_regressions/kurucz_crii_2026_10_08/fixed"
)
APPROVED_KURUCZ_STARK = (
    Path(__file__).parent
    / "data/approved_regressions/kurucz_stark_2026_10_11/fixed"
)
APPROVED_SI_HE_UNIFIED = (
    Path(__file__).parent
    / "data/approved_regressions/si_he_unified_2026_10_11/fixed"
)
APPROVED_SI_KURUCZ = (
    Path(__file__).parent
    / "data/approved_regressions/si_kurucz_2026_10_12/fixed"
)
APPROVED_UV_ATOMIC = (
    Path(__file__).parent
    / "data/approved_regressions/uv_atomic_2026_10_12/fixed"
)
APPROVED_MICROFIELD_HOSTS = (
    Path(__file__).parent
    / "data/approved_regressions/microfield_hosts_2026_10_12/fixed"
)
APPROVED_INDEPENDENT_KURUCZ = (
    Path(__file__).parent
    / "data/approved_regressions/independent_kurucz_2026_10_14/fixed"
)
APPROVED_COOL_DZ_RELEASE = (
    Path(__file__).parent
    / "data/approved_regressions/cool_dz_release_2026_10_23/fixed"
)

APPROVED_METAL_POPULATIONS = (
    Path(__file__).parent
    / "data/approved_regressions/metal_populations_2026_10_24/fixed"
)

APPROVED_MONTREAL_H_RADIUS = (
    Path(__file__).parent
    / "data/approved_regressions/montreal_h_radius_2026_10_27/fixed"
)


def _legacy_metal_only_microfield_moment(metal_state):
    """The pre-2026-10-12 moment, which omitted the host ions."""
    result = np.zeros_like(metal_state.electron_density, dtype=np.float64)
    for populations in metal_state.ion_number_density.values():
        for charge in range(1, populations.shape[0]):
            result += charge**1.5 * populations[charge]
    return result
DENSITY_EDGE_CASES = frozenset({"da-4000", "da-5000", "dab-9000"})
CRII_CASES = frozenset({"dz-pg1225", "dz-j0738"})
# Reviewed 2026-10-27 hydrogen neutral defaults (radius x0.5, Ali--Griem at
# every temperature); see that directory's manifest.
MONTREAL_H_RADIUS_CASES = frozenset(
    {"da-3000", "da-4000", "da-5000", "da-20000", "dab-20000", "dz-pg1225", "dz-j0738"}
)
pytestmark = pytest.mark.spectral
CASES = [
    "da-3000",
    "da-4000",
    "da-5000",
    "da-20000",
    "db-10000",
    "db-22000",
    "dab-9000",
    "dab-20000",
    "dz-pg1225",
    "dz-j0738",
]


@pytest.mark.parametrize("case", CASES)
def test_checked_scattering_preserves_broad_spectral_controls(case, monkeypatch):
    path = CONTROLS / (case + ".npz")
    with np.load(path) as saved:
        kind = str(saved["spectral_type"])
        saved_config = json.loads(str(saved["config_json"]))
        if kind == "DB":
            # These controls were approved for the DB-production-v4 optical-depth
            # synthesis; DBConfig now defaults to the column-mass preset.
            saved_config.setdefault("transfer_discretization", "optical-depth")
        config = {"DA": DAConfig, "DB": DBConfig, "DAB": DABConfig, "DZ": DZConfig}[
            kind
        ](**saved_config)
        wave = saved["wavelength"]
    # Explicitly reviewed density-edge outputs are versioned separately.
    # Historical structures, earlier approved spectra and tolerances stay intact.
    approved_directory = APPROVED_DENSITY_EDGE if case in DENSITY_EDGE_CASES else APPROVED
    if case in CRII_CASES:
        # Versioned correction to opacity populations after the cool-DZ release;
        # all prior controls and tolerances remain independently checked below.
        approved_directory = APPROVED_METAL_POPULATIONS
    previous_directory = approved_directory
    if case in MONTREAL_H_RADIUS_CASES:
        approved_directory = APPROVED_MONTREAL_H_RADIUS
    with np.load(approved_directory / (case + ".npz")) as approved:
        np.testing.assert_array_equal(wave, approved["wavelength"])
        expected = approved["surface_flux"]
    molecular = kind == "DA" and config.effective_temperature <= 12000
    atmosphere = load_atmosphere_checkpoint(
        path,
        config.effective_temperature,
        config.logg,
        {"DA": "hydrogen", "DB": "helium", "DAB": "mixed", "DZ": "helium"}[kind],
        include_molecules=molecular,
        include_negative_hydrogen=molecular,
        log_hydrogen_to_helium=getattr(config, "log_hydrogen_to_helium", None),
        trihydrogen_ion_partition_model="neale-tennyson-1995" if kind == "DA" else None,
    )
    # These historical checkpoints are deliberately uncertified. Synthesis
    # must remain available with a warning, without pretending to re-solve.
    with pytest.warns(AtmosphereConvergenceWarning):
        result = {
            "DA": compute_da,
            "DB": compute_db,
            "DAB": compute_dab,
            "DZ": compute_dz,
        }[kind](config, wave, initial_atmosphere=atmosphere, relax_atmosphere=False)
    new = result.spectrum.surface_flux_lambda
    assert np.all(np.isfinite(new)) and np.all(new > 0)
    np.testing.assert_allclose(new, expected, rtol=2e-6, atol=1e-12 * np.max(expected))
    assert result.spectrum.metadata["source_converged"]
    assert result.spectrum.metadata["transfer_discretization"] == (
        "formal-pchip" if kind == "DA" else "formal-linear"
    )
    assert result.spectrum.metadata["independent_radiation_scaled_source_error"] < 1e-10
    assert result.metadata["atmosphere_convergence_status"] != "converged"
    # A common 0.1% bound in significant-flux regions is stricter than
    # observational agreement; no star-specific tolerance is tuned here.
    important = wave * expected > 0.01 * np.max(wave * expected)
    np.testing.assert_allclose(new[important], expected[important], rtol=1e-3, atol=0.0)
    for lo, hi in ((1150, 3000), (3500, 7000), (7000, 300000)):
        take = (wave >= lo) & (wave <= hi)
        change = trapezoid(new[take] - expected[take], wave[take]) / trapezoid(
            expected[take], wave[take]
        )
        assert abs(change) < 1e-5

    # Every earlier control predates the 2026-10-27 hydrogen neutral defaults:
    # reload the structure EOS under the previous defaults, recover the
    # immediately preceding control, then keep each older check intact.
    montreal = case in MONTREAL_H_RADIUS_CASES
    with previous_hydrogen_neutral_defaults() if montreal else nullcontext():
        if montreal:
            atmosphere = load_atmosphere_checkpoint(
                path,
                config.effective_temperature,
                config.logg,
                {"DA": "hydrogen", "DB": "helium", "DAB": "mixed", "DZ": "helium"}[kind],
                include_molecules=molecular,
                include_negative_hydrogen=molecular,
                log_hydrogen_to_helium=getattr(config, "log_hydrogen_to_helium", None),
                trihydrogen_ion_partition_model="neale-tennyson-1995" if kind == "DA" else None,
            )
            with pytest.warns(AtmosphereConvergenceWarning):
                previous_defaults = {
                    "DA": compute_da,
                    "DB": compute_db,
                    "DAB": compute_dab,
                    "DZ": compute_dz,
                }[kind](config, wave, initial_atmosphere=atmosphere, relax_atmosphere=False)
            with np.load(previous_directory / (case + ".npz")) as control:
                previous_defaults_flux = control["surface_flux"]
            np.testing.assert_allclose(
                previous_defaults.spectrum.surface_flux_lambda, previous_defaults_flux,
                rtol=2e-6, atol=1e-12 * np.max(previous_defaults_flux),
            )
        if case in CRII_CASES:
            with previous_metal_opacity_populations(), pytest.warns(AtmosphereConvergenceWarning):
                release = compute_dz(config, wave, initial_atmosphere=atmosphere,
                                     relax_atmosphere=False)
            with np.load(APPROVED_COOL_DZ_RELEASE / (case + ".npz")) as control:
                release_flux = control["surface_flux"]
            np.testing.assert_allclose(release.spectrum.surface_flux_lambda, release_flux,
                rtol=2e-6, atol=1e-12 * np.max(release_flux))
            # Every earlier approval predates the cool-DZ release: recover its
            # defaults and physics first, then keep each predecessor check intact.
            config = previous_cool_dz_release_config(config)
            with previous_metal_opacity_populations(), previous_cool_dz_release_physics():
                # Retain the exact predecessor and its tolerances as a separate check.
                with previous_kurucz_defaults(), pytest.warns(AtmosphereConvergenceWarning):
                    predecessor = compute_dz(config, wave, initial_atmosphere=atmosphere,
                                             relax_atmosphere=False)
                with np.load(APPROVED_UV_ATOMIC / (case + ".npz")) as control:
                    previous_default_flux = control["surface_flux"]
                np.testing.assert_allclose(predecessor.spectrum.surface_flux_lambda,
                    previous_default_flux, rtol=2e-6, atol=1e-12 * np.max(previous_default_flux))
                with legacy_uv_atomic_controls():
                    # First recover the immediate predecessor without altering the
                    # historical structure, EOS, metal populations or host microfields.
                    with pytest.warns(AtmosphereConvergenceWarning):
                        legacy_atomic = compute_dz(
                            config, wave, initial_atmosphere=atmosphere,
                            relax_atmosphere=False,
                        )
                    with np.load(APPROVED_MICROFIELD_HOSTS / (case + ".npz")) as control:
                        host_flux = control["surface_flux"]
                        np.testing.assert_array_equal(wave, control["wavelength"])
                    np.testing.assert_allclose(
                        legacy_atomic.spectrum.surface_flux_lambda, host_flux,
                        rtol=2e-6, atol=1e-12 * np.max(host_flux),
                    )
                    # Restoring only the previous metal-only microfield moment must recover
                    # the previous (default Kurucz Si II transitions) control; the older
                    # controls below were all captured with that moment.
                    from wd_spectra import metals

                    monkeypatch.setattr(
                        metals, "metal_ionic_microfield_perturber_density",
                        _legacy_metal_only_microfield_moment,
                    )
                    with pytest.warns(AtmosphereConvergenceWarning):
                        legacy = compute_dz(
                            config, wave, initial_atmosphere=atmosphere,
                            relax_atmosphere=False,
                        )
                    with np.load(APPROVED_SI_KURUCZ / (case + ".npz")) as control:
                        legacy_flux = control["surface_flux"]
                    np.testing.assert_allclose(
                        legacy.spectrum.surface_flux_lambda, legacy_flux,
                        rtol=2e-6, atol=1e-12 * np.max(legacy_flux),
                    )
                    # Disabling only the default Kurucz Si II missing transitions must
                    # recover the previous (default unified Si+-He profile) control.
                    unified_config = replace(config, si_ii_kurucz_transitions=False)
                    with pytest.warns(AtmosphereConvergenceWarning):
                        unified = compute_dz(
                            unified_config, wave, initial_atmosphere=atmosphere,
                            relax_atmosphere=False,
                        )
                    with np.load(APPROVED_SI_HE_UNIFIED / (case + ".npz")) as control:
                        unified_flux = control["surface_flux"]
                    np.testing.assert_allclose(
                        unified.spectrum.surface_flux_lambda, unified_flux,
                        rtol=2e-6, atol=1e-12 * np.max(unified_flux),
                    )
                    # Disabling the unified Si+-He profile as well must recover the
                    # default Kurucz electron Stark control.
                    stark_config = replace(unified_config, si_ii_he_unified_profile=False)
                    with pytest.warns(AtmosphereConvergenceWarning):
                        stark = compute_dz(
                            stark_config, wave, initial_atmosphere=atmosphere,
                            relax_atmosphere=False,
                        )
                    with np.load(APPROVED_KURUCZ_STARK / (case + ".npz")) as control:
                        stark_flux = control["surface_flux"]
                    np.testing.assert_allclose(
                        stark.spectrum.surface_flux_lambda, stark_flux,
                        rtol=2e-6, atol=1e-12 * np.max(stark_flux),
                    )
                    # Disabling the Kurucz electron Stark constants as well must recover
                    # the default Cr II control.
                    previous_config = replace(stark_config, kurucz_electron_stark=False)
                    with pytest.warns(AtmosphereConvergenceWarning):
                        previous = compute_dz(
                            previous_config, wave, initial_atmosphere=atmosphere,
                            relax_atmosphere=False,
                        )
                    with np.load(APPROVED_CRII / (case + ".npz")) as control:
                        previous_flux = control["surface_flux"]
                    np.testing.assert_allclose(
                        previous.spectrum.surface_flux_lambda, previous_flux,
                        rtol=2e-6, atol=1e-12 * np.max(previous_flux),
                    )
                    # Keep the historical control as an independent check: disabling the
                    # Cr II supplement as well recovers the pre-Cr II spectrum.
                    from wd_spectra.models import stellar
                    from wd_spectra._kurucz_supplements import SOURCE

                    assert SOURCE in result.metadata["atomic_lines"]
                    reader = stellar.read_stout_atomic_database

                    def raw_stout(*args, **kwargs):
                        kwargs["include_default_supplements"] = False
                        return reader(*args, **kwargs)

                    # Scope this patch to the call. A test-wide monkeypatch would record
                    # the reader that is current inside the legacy scopes (their
                    # replacement) and restore it at teardown, after those scopes have
                    # exited, leaving every later DZ model with the old supplements.
                    with patch.object(stellar, "read_stout_atomic_database", raw_stout), \
                            pytest.warns(AtmosphereConvergenceWarning):
                        original = compute_dz(
                            previous_config, wave, initial_atmosphere=atmosphere,
                            relax_atmosphere=False,
                        )
                    with np.load(APPROVED / (case + ".npz")) as historical:
                        old_flux = historical["surface_flux"]
                    np.testing.assert_allclose(
                        original.spectrum.surface_flux_lambda, old_flux,
                        rtol=2e-6, atol=1e-12 * np.max(old_flux),
                    )
