"""DAZ API/physics wiring checks; mock states are not equilibrium evidence."""

from dataclasses import replace
import json
import numpy as np
import pytest
from wd_spectra import DAZConfig, compute_daz, select_physics, run_model
from wd_spectra.atmosphere import gray_hydrogen_atmosphere
from wd_spectra.models import daz, automatic, AtmosphereConvergenceWarning, ModelData
from wd_spectra.models.common import model_request_fingerprint
from wd_spectra.spectrum import Spectrum


@pytest.fixture
def wiring(monkeypatch):
    atmosphere = gray_hydrogen_atmosphere(9000, 8, n_depth=5)
    calls = {}
    monkeypatch.setattr(ModelData, "require", lambda *args, **kwargs: None)
    for name in (
        "read_stout_atomic_database",
        "read_verner_photoionization_database",
        "read_barklem_neutral_hydrogen_broadening",
        "read_borysow_h2_h2_cia_table",
    ):
        monkeypatch.setattr(daz, name, lambda *args, **kwargs: object())
    monkeypatch.setattr(
        daz, "_allard_lyman_profiles_for_effective_temperature", lambda *a, **k: None
    )

    def relax(*args, **kwargs):
        calls["structure"] = kwargs
        return atmosphere

    def synthesize(current, wavelength, **kwargs):
        calls["synthesis"] = kwargs
        return Spectrum(np.array([4000.0, 5000.0]), np.ones(2), {})

    monkeypatch.setattr(daz, "radiative_equilibrium_hydrogen_atmosphere", relax)
    monkeypatch.setattr(daz, "synthesize_hydrogen_spectrum", synthesize)
    return calls, atmosphere


def test_default_and_automatic_selection_do_not_use_a_helium_host(monkeypatch):
    monkeypatch.setattr(
        "wd_spectra.models.selection._helium_probe",
        lambda *a: pytest.fail("helium probe"),
    )
    assert DAZConfig().strong_line_atomic_data == "stout"
    assert DAZConfig().metal_neutral_h_broadening == "unsold"
    assert DAZConfig().abundances["Ca"] == -6.58
    assert select_physics(DAZConfig()).workflow == "daz"
    assert not select_physics(DAZConfig()).experimental


def test_metals_and_hydrogen_chemistry_are_shared_by_structure_and_spectrum(wiring):
    calls, _ = wiring
    callback = lambda *a: None
    with pytest.warns(AtmosphereConvergenceWarning):
        result = compute_daz(
            DAZConfig(effective_temperature=9000, quality="production"),
            [4000, 5000],
            iteration_callback=callback,
        )
    a, s = calls["structure"], calls["synthesis"]
    for key in (
        "metal_database",
        "metal_abundances",
        "metal_photoionization_database",
        "h2_h2_cia_table",
    ):
        assert a[key] is s[key]
    assert a["initial_temperature"] is None and a["initial_column_mass"] is None
    assert a["iteration_callback"] is callback
    assert a["n_depth"] == 100 and a["structure_solver"] == "adaptive-newton"
    assert a["include_molecules"] and a["include_negative_hydrogen"]
    assert a["trihydrogen_ion_partition_model"] == "neale-tennyson-1995"
    assert a["mixing_length_alpha"] == 0.7
    # Preserve the established atmosphere/synthesis sampling policy. Detailed
    # final synthesis need not force every weak line into the atmosphere solve.
    assert a["minimum_metal_oscillator_strength"] == 0.01
    assert s["minimum_metal_oscillator_strength"] == 1e-4
    assert a["maximum_metal_lines"] == 1000
    assert s["maximum_metal_lines"] == 20_000
    assert a["n_angle"] == 3
    assert s["n_angle"] == 4
    assert "transfer_discretization" not in s
    assert s["include_dense_helium_metal_ionization"] is False
    assert result.metadata["metal_opacity_in_structure"]
    assert result.metadata["atmosphere_convergence_status"] != "converged"


def test_fixed_state_does_not_run_an_atmosphere_or_claim_equilibrium(wiring):
    calls, atmosphere = wiring
    with pytest.warns(AtmosphereConvergenceWarning):
        result = compute_daz(
            DAZConfig(quality="quick"),
            [4000, 5000],
            initial_atmosphere=atmosphere,
            relax_atmosphere=False,
        )
    assert "structure" not in calls
    assert result.metadata["atmosphere_convergence_status"] != "converged"


@pytest.mark.parametrize("strict", [False, True])
def test_automatic_daz_is_cold_and_retains_unqualified_outputs(
    wiring, tmp_path, strict
):
    with pytest.warns(AtmosphereConvergenceWarning):
        if strict:
            with pytest.raises(RuntimeError, match="without full equilibrium"):
                run_model(
                    DAZConfig(quality="quick"),
                    tmp_path / "model",
                    require_convergence=True,
                )
        else:
            result = run_model(DAZConfig(quality="quick"), tmp_path / "model")
            assert not result.convergence_verified
    record = json.loads((tmp_path / "model/model-run.json").read_text())
    assert record["cold_start"] and record["initial_checkpoint"] is None
    assert (tmp_path / "model/spectrum.txt").is_file()
    assert wiring[0]["structure"]["initial_temperature"] is None


def test_automatic_daz_rejects_checkpoint(tmp_path):
    with pytest.raises(TypeError, match="initial_checkpoint"):
        run_model(DAZConfig(), tmp_path / "model", initial_checkpoint="old.npz")


@pytest.mark.parametrize(
    "changes",
    [
        {"abundances": {}},
        {"abundances": {"Ca": np.nan}},
        {"abundances": {"He": -1}},
        {"metal_neutral_h_broadening": "bad"},
        {"strong_line_atomic_data": "bad"},
        {"h3plus_partition_model": "bad"},
        {"ca_ii_resonance_source": "bad"},
        {"lyman_profile_source": "bad"},
        {"maximum_metal_charge": 0},
    ],
)
def test_invalid_requests_fail_explicitly(changes):
    with pytest.raises(ValueError):
        compute_daz(replace(DAZConfig(), **changes), [4000, 5000])


def test_changed_abundance_invalidates_request_identity():
    config = DAZConfig()
    original = model_request_fingerprint("DAZ", config, ModelData.default())
    changed = model_request_fingerprint(
        "DAZ", replace(config, abundances={"Ca": -7}), ModelData.default()
    )
    assert original["sha256"] != changed["sha256"]
