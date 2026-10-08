from copy import deepcopy
import numpy as np
import pytest
from wd_spectra._convergence import equilibrium_certificate, recorded_equilibrium_status


def measured():
    return dict(
        radiative_equilibrium_solver_converged=True,
        maximum_all_depth_total_flux_residual=1e-5,
        maximum_photospheric_total_flux_residual=1e-5,
        surface_flux_ratio=1.0,
        maximum_relative_cell_energy_balance_residual=1e-5,
        temperature_correction_measured=True,
        radiative_equilibrium_maximum_log_temperature_correction=1e-5,
        maximum_unrestricted_log_temperature_correction=1e-5,
        electron_scattering_source_final_maximum_relative_residual=1e-12,
        lower_boundary_absorption_escape_bound=1e-5,
    )


def test_complete_measured_evidence_is_required():
    data = measured()
    data["equilibrium_certificate"] = equilibrium_certificate(data)
    assert recorded_equilibrium_status(data) == "converged"
    assert (
        recorded_equilibrium_status({"radiative_equilibrium_converged": True})
        == "unknown"
    )
    assert not data["equilibrium_certificate"]["independent_grid_validation"]
    assert not data["equilibrium_certificate"]["full_physics_validation"]


@pytest.mark.parametrize(
    "key,value",
    [
        ("maximum_all_depth_total_flux_residual", 0.01),
        ("maximum_relative_cell_energy_balance_residual", 0.24),
        ("temperature_correction_measured", False),
        ("maximum_unrestricted_log_temperature_correction", 0.04),
        ("electron_scattering_source_final_maximum_relative_residual", 1e-3),
        ("lower_boundary_absorption_escape_bound", 0.1),
        ("radiative_equilibrium_solver_converged", False),
        ("maximum_relative_cell_energy_balance_residual", np.nan),
        ("maximum_relative_cell_energy_balance_residual", None),
    ],
)
def test_one_failed_check_cannot_be_hidden_by_old_true_flag(key, value):
    data = measured()
    data["equilibrium_certificate"] = equilibrium_certificate(data)
    data["radiative_equilibrium_converged"] = True
    data[key] = value
    assert recorded_equilibrium_status(data) == "unconverged"


def test_initial_state_zero_is_not_a_measured_correction():
    data = measured()
    data.update(
        temperature_correction_measured=False,
        maximum_unrestricted_log_temperature_correction=None,
        radiative_equilibrium_maximum_log_temperature_correction=0.0,
    )
    report = equilibrium_certificate(data)
    assert not report["verified"]
    assert report["failures"] == ["temperature_stationarity"]


def test_declared_residual_contract_records_but_does_not_require_stationarity():
    data = measured()
    data.update(
        temperature_correction_measured=False,
        maximum_unrestricted_log_temperature_correction=None,
    )
    required = (
        "all_depth_flux", "local_energy", "source_closure", "boundary_screening"
    )
    data["equilibrium_certificate"] = equilibrium_certificate(
        data, required_checks=required
    )
    report = data["equilibrium_certificate"]
    assert report["verified"]
    assert report["required_checks"] == required
    assert not report["checks"]["temperature_stationarity"]["passed"]
    assert recorded_equilibrium_status(data) == "converged"


def test_pg1159_spectrum_profile_cannot_report_equilibrium_convergence():
    data = measured()
    data.update(
        temperature_correction_measured=False,
        maximum_unrestricted_log_temperature_correction=None,
    )
    required = (
        "surface_flux",
        "photospheric_flux",
        "all_depth_flux",
        "local_energy",
        "source_closure",
        "boundary_screening",
    )
    data["equilibrium_certificate"] = equilibrium_certificate(
        data,
        required_checks=required,
        profile="pg1159-spectrum-gate-v1",
    )
    report = data["equilibrium_certificate"]
    assert report["schema"] == 2
    assert report["profile"] == "pg1159-spectrum-gate-v1"
    assert report["verified"]
    assert recorded_equilibrium_status(data) == "spectrum-qualified"

    data["maximum_photospheric_total_flux_residual"] = 0.1
    assert recorded_equilibrium_status(data) == "unconverged"


def test_local_energy_can_use_a_stricter_declared_tolerance_than_flux():
    data = measured()
    data["maximum_all_depth_total_flux_residual"] = 8e-3
    data["maximum_relative_cell_energy_balance_residual"] = 2e-3
    report = equilibrium_certificate(
        data,
        flux_tolerance=1e-2,
        local_energy_tolerance=3e-3,
        required_checks=("all_depth_flux", "local_energy"),
    )
    assert report["verified"]
    assert report["checks"]["all_depth_flux"]["tolerance"] == 1e-2
    assert report["checks"]["local_energy"]["tolerance"] == 3e-3

    data["maximum_relative_cell_energy_balance_residual"] = 4e-3
    assert not equilibrium_certificate(
        data,
        flux_tolerance=1e-2,
        local_energy_tolerance=3e-3,
        required_checks=("all_depth_flux", "local_energy"),
    )["verified"]


def test_surface_contract_can_record_failed_internal_flux_diagnostics():
    data = measured()
    data["maximum_all_depth_total_flux_residual"] = 0.16
    data["maximum_photospheric_total_flux_residual"] = 0.14
    data["surface_flux_ratio"] = 1.004
    report = equilibrium_certificate(
        data,
        flux_tolerance=1e-2,
        surface_flux_tolerance=7.5e-3,
        local_energy_tolerance=3e-3,
        required_checks=(
            "surface_flux",
            "local_energy",
            "source_closure",
            "boundary_screening",
        ),
    )
    assert report["verified"]
    assert not report["checks"]["all_depth_flux"]["passed"]
    assert not report["checks"]["photospheric_flux"]["passed"]
    assert report["checks"]["surface_flux"]["passed"]

    data["equilibrium_certificate"] = report
    assert recorded_equilibrium_status(data) == "converged"


@pytest.mark.parametrize("checks", [None, [], {"all_depth_flux": None}])
def test_malformed_checkpoint_certificate_is_unknown(checks):
    assert (
        recorded_equilibrium_status(
            dict(equilibrium_certificate=dict(schema=1, checks=checks))
        )
        == "unknown"
    )


@pytest.mark.parametrize(
    "name",
    [
        "flux_tolerance",
        "surface_flux_tolerance",
        "photospheric_flux_tolerance",
        "local_energy_tolerance",
    ],
)
@pytest.mark.parametrize("value", [0.0, -1.0, np.nan, np.inf, True])
def test_bad_certificate_tolerance_is_rejected(name, value):
    with pytest.raises(ValueError):
        equilibrium_certificate(measured(), **{name: value})
