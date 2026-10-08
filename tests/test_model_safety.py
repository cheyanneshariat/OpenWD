from dataclasses import replace
import warnings

import numpy as np
import pytest

from wd_spectra.atmosphere import Atmosphere, gray_helium_atmosphere
from wd_spectra._convergence import equilibrium_certificate
from wd_spectra.models import (
    AtmosphereConvergenceWarning,
    DBConfig,
    DZConfig,
    ModelData,
    ModelResult,
    atmosphere_convergence_status,
    load_atmosphere_checkpoint,
    save_model_result,
)
from wd_spectra.models import stellar
from wd_spectra.models.common import (
    atmosphere_matches_model_request,
    atmosphere_with_model_request_fingerprint,
    model_request_fingerprint,
    warn_if_atmosphere_not_converged,
)
from wd_spectra.spectrum import Spectrum


def test_dz_default_uses_paper_figure_physics():
    assert DZConfig().strong_line_atomic_data == "stout"
    assert DZConfig().lyman_profile_source == "stark"


def _atmosphere(metadata=None):
    # A real EOS state is required to test a same-physics checkpoint round trip.
    return replace(gray_helium_atmosphere(10000.,8.,n_depth=3),
                   metadata={} if metadata is None else dict(metadata))


def _certified_metadata():
    """Synthetic passing diagnostics for API wiring tests, not a model run."""
    data = dict(radiative_equilibrium_converged=True,
        radiative_equilibrium_solver_converged=True,
        maximum_all_depth_total_flux_residual=1e-5,
        maximum_relative_cell_energy_balance_residual=1e-5,
        temperature_correction_measured=True,
        radiative_equilibrium_maximum_log_temperature_correction=1e-5,
        electron_scattering_source_final_maximum_relative_residual=1e-12,
        lower_boundary_absorption_escape_bound=1e-5)
    data['equilibrium_certificate']=equilibrium_certificate(data)
    return data


def _spectrum():
    return Spectrum(
        wavelength_angstrom=np.array([4_000.0, 5_000.0]),
        surface_flux_lambda=np.array([1.0, 2.0]),
        metadata={},
    )


def test_model_request_fingerprint_is_order_independent_and_request_scoped(
    tmp_path,
):
    data = ModelData(tmp_path)
    first = model_request_fingerprint(
        "DZ",
        DZConfig(abundances={"Ca": -8.0, "Mg": -7.0}),
        data,
    )
    reordered = model_request_fingerprint(
        "DZ",
        DZConfig(abundances={"Mg": -7.0, "Ca": -8.0}),
        data,
    )
    changed = model_request_fingerprint(
        "DZ",
        DZConfig(abundances={"Ca": -8.1, "Mg": -7.0}),
        data,
    )

    assert first["sha256"] == reordered["sha256"]
    assert first["sha256"] != changed["sha256"]


def test_only_an_exact_fingerprint_authorizes_direct_resume(tmp_path):
    data = ModelData(tmp_path)
    requested = model_request_fingerprint("DB", DBConfig(effective_temperature=10000.), data)
    other = model_request_fingerprint(
        "DB", DBConfig(mixing_length_alpha=0.8), data
    )
    checkpoint = atmosphere_with_model_request_fingerprint(
        _atmosphere({"checkpoint_composition_verified": True}), requested
    )

    assert atmosphere_matches_model_request(checkpoint, requested)
    assert not atmosphere_matches_model_request(checkpoint, other)
    tampered = replace(
        checkpoint,
        metadata={
            **checkpoint.metadata,
            "model_request_fingerprint": {
                **requested,
                "data_root": "/different/data",
            },
        },
    )
    assert not atmosphere_matches_model_request(tampered, requested)
    assert not atmosphere_matches_model_request(
        _atmosphere({"checkpoint_composition_verified": True}), requested
    )


def test_replacing_table_in_place_invalidates_checkpoint_even_with_same_mtime(tmp_path):
    import os

    data = ModelData(tmp_path)
    table = data.helium_i_stark
    table.parent.mkdir(parents=True)
    table.write_bytes(b"version A")
    config = DBConfig(effective_temperature=10000.)
    original = model_request_fingerprint("DB", config, data)
    checkpoint = atmosphere_with_model_request_fingerprint(
        _atmosphere(_certified_metadata()), original
    )
    timestamps = table.stat()
    table.write_bytes(b"version B")
    os.utime(table, ns=(timestamps.st_atime_ns, timestamps.st_mtime_ns))
    changed = model_request_fingerprint("DB", config, data)
    assert not atmosphere_matches_model_request(checkpoint, changed)
    fixed = stellar.fixed_synthesis_atmosphere(checkpoint, changed)
    assert atmosphere_convergence_status(fixed) == "unconverged"


def test_added_and_removed_tables_change_data_identity(tmp_path):
    from wd_spectra._provenance import model_data_identity

    data = ModelData(tmp_path)
    data.cache.mkdir()
    original = model_data_identity(data)
    table = data.cache / "new-table.dat"
    table.write_bytes(b"new")
    assert model_data_identity(data) != original
    table.unlink()
    assert model_data_identity(data) == original


def test_undoubled_microfields_invalidate_doubled_physics_checkpoints(
    monkeypatch, tmp_path
):
    from wd_spectra.models import common

    data = ModelData(tmp_path)
    config = DBConfig(effective_temperature=10000.)
    with monkeypatch.context() as historical:
        historical.setattr(
            common, "_MODEL_PHYSICS_REVISION",
            "openwd-0.1.3-cold-local-energy-and-domain-v3",
        )
        old = model_request_fingerprint("DB", config, data)
    requested = model_request_fingerprint("DB", config, data)
    checkpoint = atmosphere_with_model_request_fingerprint(
        _atmosphere({"checkpoint_composition_verified": True}), old
    )
    assert requested["physics_revision"] != old["physics_revision"]
    assert not atmosphere_matches_model_request(checkpoint, requested)


def test_unconverged_and_unknown_atmospheres_warn_without_blocking():
    unconverged = _atmosphere(
        {
            "radiative_equilibrium_converged": False,
            "maximum_all_depth_total_flux_residual": 0.12,
        }
    )
    with pytest.warns(
        AtmosphereConvergenceWarning,
        match="did not converge.*maximum all-depth flux residual=1.200e-01",
    ):
        assert (
            warn_if_atmosphere_not_converged(unconverged, "DB")
            == "unconverged"
        )

    with pytest.warns(AtmosphereConvergenceWarning, match="does not record"):
        assert warn_if_atmosphere_not_converged(_atmosphere(), "DB") == "unknown"


def test_converged_atmosphere_does_not_warn():
    atmosphere = _atmosphere(_certified_metadata())
    with warnings.catch_warnings():
        warnings.simplefilter("error", AtmosphereConvergenceWarning)
        assert warn_if_atmosphere_not_converged(atmosphere, "DB") == "converged"
    assert atmosphere_convergence_status(atmosphere) == "converged"


def test_checkpoint_round_trip_preserves_request_fingerprint(tmp_path):
    data = ModelData(tmp_path)
    fingerprint = model_request_fingerprint("DB", DBConfig(effective_temperature=10000.), data)
    atmosphere = atmosphere_with_model_request_fingerprint(
        _atmosphere({"radiative_equilibrium_converged": True}), fingerprint
    )
    result = ModelResult("DB", atmosphere, _spectrum(), DBConfig(), {})

    output = save_model_result(result, tmp_path / "saved")
    restored = load_atmosphere_checkpoint(
        output / "atmosphere.npz",
        10_000.0,
        8.0,
        "helium",
    )

    assert atmosphere_matches_model_request(restored, fingerprint)
    assert restored.metadata["radiative_equilibrium_converged"] is True


def test_experimental_partition_checkpoint_requires_new_relaxation(tmp_path):
    fingerprint = model_request_fingerprint("DB", DBConfig(), ModelData(tmp_path))
    atmosphere = atmosphere_with_model_request_fingerprint(
        _atmosphere({
            "radiative_equilibrium_converged": True,
            "experimental_h2_partition": "external research closure",
        }),
        fingerprint,
    )
    result = ModelResult("DB", atmosphere, _spectrum(), DBConfig(), {})
    output = save_model_result(result, tmp_path / "experimental")
    restored = load_atmosphere_checkpoint(
        output / "atmosphere.npz", 10_000.0, 8.0, "helium"
    )
    np.testing.assert_array_equal(restored.temperature, atmosphere.temperature)
    assert not atmosphere_matches_model_request(restored, fingerprint)
    assert restored.metadata["radiative_equilibrium_converged"] is False
    assert restored.metadata["checkpoint_chemistry_changed_requires_relaxation"]
    with pytest.warns(AtmosphereConvergenceWarning, match="did not converge"):
        assert warn_if_atmosphere_not_converged(restored, "DB") == "unconverged"


def test_compute_db_warns_and_returns_an_unconverged_exploratory_result(
    monkeypatch, tmp_path
):
    captured = {}
    partial = _atmosphere({"radiative_equilibrium_converged": False})
    monkeypatch.setattr(stellar, "_helium_tables", lambda data: (object(), object()))

    def fake_relaxation(*args, **kwargs):
        captured.update(kwargs)
        return partial

    monkeypatch.setattr(
        stellar, "radiative_equilibrium_helium_atmosphere", fake_relaxation
    )
    monkeypatch.setattr(stellar, "synthesize_helium_spectrum", lambda *a, **k: _spectrum())

    with pytest.warns(AtmosphereConvergenceWarning):
        result = stellar.compute_db(
            DBConfig(quality="quick"),
            np.array([4_000.0, 5_000.0]),
            data=ModelData(tmp_path),
        )

    assert result.atmosphere.metadata["radiative_equilibrium_converged"] is False
    assert result.metadata["atmosphere_convergence_status"] == "unconverged"
    assert captured["resume_supplied_structure_in_formal_flux_phase"] is False
    assert captured["initial_gas_pressure"] is None
    assert captured["initial_rosseland_optical_depth"] is None


def test_compute_db_exact_fingerprint_controls_direct_resume(monkeypatch, tmp_path):
    data = ModelData(tmp_path)
    config = DBConfig(effective_temperature=10000., quality="quick")
    fingerprint = model_request_fingerprint("DB", config, data)
    checkpoint = atmosphere_with_model_request_fingerprint(
        _atmosphere(_certified_metadata()), fingerprint
    )
    captured = {}
    monkeypatch.setattr(stellar, "_helium_tables", lambda data: (object(), object()))

    def fake_relaxation(*args, **kwargs):
        captured.update(kwargs)
        return replace(
            checkpoint,
            metadata={
                **checkpoint.metadata,
                "radiative_equilibrium_converged": True,
            },
        )

    monkeypatch.setattr(
        stellar, "radiative_equilibrium_helium_atmosphere", fake_relaxation
    )
    monkeypatch.setattr(stellar, "synthesize_helium_spectrum", lambda *a, **k: _spectrum())

    stellar.compute_db(
        config,
        np.array([4_000.0, 5_000.0]),
        data=data,
        initial_atmosphere=checkpoint,
    )

    assert captured["resume_supplied_structure_in_formal_flux_phase"] is True
    np.testing.assert_array_equal(
        captured["initial_gas_pressure"], checkpoint.gas_pressure
    )
    np.testing.assert_array_equal(
        captured["initial_rosseland_optical_depth"],
        checkpoint.rosseland_optical_depth,
    )
