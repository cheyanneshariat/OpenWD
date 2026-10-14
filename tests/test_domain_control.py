from dataclasses import replace
import numpy as np
import pytest
from wd_spectra import gray_helium_atmosphere
from wd_spectra._domain import (
    append_lower_domain,
    solve_with_screened_boundary,
)


def test_append_preserves_all_old_nodes_and_hydrostatics():
    a = gray_helium_atmosphere(10000.0, 8.0, n_depth=20)
    new = append_lower_domain(a)
    for key, old in [
        ("temperature", a.temperature),
        ("gas_pressure", a.gas_pressure),
        ("column_mass", a.column_mass),
        ("rosseland_optical_depth", a.rosseland_optical_depth),
    ]:
        np.testing.assert_array_equal(new["initial_" + key][: a.n_depth], old)
    p, m = new["initial_gas_pressure"], new["initial_column_mass"]
    np.testing.assert_allclose(np.diff(p), a.gravity * np.diff(m))
    assert new["n_depth"] > a.n_depth
    assert p[-1] == 2 * p[a.n_depth - 1]


def test_domain_control_never_uses_external_atmosphere_or_changes_physics():
    a = gray_helium_atmosphere(10000.0, 8.0, n_depth=20)
    requests = []

    def solve(teff, logg, **kwargs):
        requests.append(kwargs)
        assert kwargs["material_policy"] == "unchanged"
        metadata = dict(
            radiative_equilibrium_iterations=2,
            lower_boundary_absorption_escape_bound=(
                0.03 if len(requests) == 1 else 0.0001
            ),
            lower_boundary_screening_tolerance=0.003,
            equilibrium_certificate=dict(
                failures=["boundary_screening"] if len(requests) == 1 else []
            ),
        )
        return replace(a, metadata=metadata)

    result = solve_with_screened_boundary(
        solve, 10000.0, 8.0, material_policy="unchanged"
    )
    assert len(requests) == 2
    assert "initial_temperature" not in requests[0]
    assert requests[1]["initial_temperature"].size > 20
    assert result.metadata["cold_start"]
    assert result.metadata["adaptive_domain_expansions"] == 1
    assert (
        result.metadata[
            "radiative_equilibrium_iterations_including_domain_adaptation"
        ]
        == 4
    )


def test_domain_does_not_retry_numerically_unconverged_atmosphere():
    a = gray_helium_atmosphere(10000.0, 8.0, n_depth=20)
    calls = []

    def solve(**kwargs):
        calls.append(True)
        return replace(
            a,
            metadata={
                "equilibrium_certificate": {
                    "failures": ["local_energy", "boundary_screening"]
                }
            },
        )

    result = solve_with_screened_boundary(solve)
    assert len(calls) == 1
    assert result.metadata["equilibrium_certificate"]["failures"] == [
        "local_energy",
        "boundary_screening",
    ]


def test_domain_extension_policy_is_explicit_and_local_to_one_solve():
    a = gray_helium_atmosphere(10000.0, 8.0, n_depth=20)
    extensions = []
    requests = []

    def one_cell(atmosphere):
        extensions.append(atmosphere.n_depth)
        return append_lower_domain(atmosphere, maximum_new_cells=1)

    def solve(**options):
        requests.append(options)
        failed = len(requests) == 1
        return replace(a, metadata={
            "radiative_equilibrium_iterations": 1,
            "lower_boundary_absorption_escape_bound": 0.01 if failed else 0.0,
            "lower_boundary_screening_tolerance": 0.002,
            "equilibrium_certificate": {
                "failures": ["boundary_screening"] if failed else []
            },
        })

    solve_with_screened_boundary(solve, lower_domain_extension=one_cell)
    assert extensions == [20]
    assert requests[1]["n_depth"] == 21
    # A following default solve must still use the shared pressure-doubling
    # policy; no process-global extension hook was installed.
    requests.clear()
    solve_with_screened_boundary(solve)
    assert requests[1]["n_depth"] == append_lower_domain(a)["n_depth"]


@pytest.mark.parametrize("budget", [-1, True, 1.5])
def test_invalid_domain_budget_is_rejected_before_solving(budget):
    def forbidden(**kwargs):
        raise AssertionError("invalid budget reached solver")

    with pytest.raises(ValueError, match="nonnegative integer"):
        solve_with_screened_boundary(
            forbidden, maximum_domain_expansions=budget
        )


def _screen_solver(atmosphere, failures_by_call):
    requests = []

    def solve(teff, logg, **kwargs):
        requests.append(kwargs)
        failures = failures_by_call[min(len(requests), len(failures_by_call)) - 1]
        metadata = dict(
            radiative_equilibrium_iterations=3,
            lower_boundary_absorption_escape_bound=(0.5 if "boundary_screening" in failures else 1e-6),
            lower_boundary_screening_tolerance=0.003,
            equilibrium_certificate=dict(failures=list(failures)),
        )
        return replace(atmosphere, metadata=metadata)

    return solve, requests


def test_default_deepening_requires_screening_to_be_the_sole_failure():
    a = gray_helium_atmosphere(10000.0, 8.0, n_depth=20)
    solve, requests = _screen_solver(a, [["local_energy", "boundary_screening"]])
    result = solve_with_screened_boundary(solve, 10000.0, 8.0)
    assert len(requests) == 1
    assert result.metadata["adaptive_domain_expansions"] == 0


def test_opt_in_deepening_on_any_screen_failure_restarts_with_chosen_pressure_factor():
    a = gray_helium_atmosphere(10000.0, 8.0, n_depth=20)
    solve, requests = _screen_solver(
        a, [["solver_completion", "local_energy", "boundary_screening"], []]
    )
    result = solve_with_screened_boundary(
        solve, 10000.0, 8.0,
        deepen_on_any_screen_failure=True,
        restart_after_deepening=True,
        deepening_pressure_factor=4.0,
    )
    assert len(requests) == 2
    second = requests[1]
    assert second["resume_supplied_structure_in_formal_flux_phase"] is False
    p = second["initial_gas_pressure"]
    assert p[-1] == pytest.approx(4.0 * a.gas_pressure[-1])
    assert result.metadata["adaptive_domain_expansions"] == 1
    assert result.metadata["adaptive_domain_segments"][0]["failures"] == [
        "solver_completion", "local_energy", "boundary_screening"]


def test_default_deepening_still_resumes_in_formal_flux_phase():
    a = gray_helium_atmosphere(10000.0, 8.0, n_depth=20)
    solve, requests = _screen_solver(a, [["boundary_screening"], []])
    solve_with_screened_boundary(solve, 10000.0, 8.0)
    assert requests[1]["resume_supplied_structure_in_formal_flux_phase"] is True
    assert requests[1]["initial_gas_pressure"][-1] == pytest.approx(2.0 * a.gas_pressure[-1])


@pytest.mark.parametrize("factor", [1.0, 0.5, float("nan")])
def test_invalid_deepening_pressure_factor_is_rejected(factor):
    with pytest.raises(ValueError):
        solve_with_screened_boundary(lambda *a, **k: None, deepening_pressure_factor=factor)


def test_early_deepening_restarts_after_first_phase_and_skips_on_last_extension():
    from wd_spectra._domain import LowerBoundaryTooShallow

    a = gray_helium_atmosphere(10000.0, 8.0, n_depth=20)
    requests = []

    def solve(teff, logg, **kwargs):
        requests.append(kwargs)
        if len(requests) == 1:
            assert kwargs["early_lower_boundary_escape_limit"] == 0.1
            raise LowerBoundaryTooShallow(a, 0.55, iterations=7)
        metadata = dict(
            radiative_equilibrium_iterations=5,
            lower_boundary_absorption_escape_bound=1e-6,
            lower_boundary_screening_tolerance=0.003,
            equilibrium_certificate=dict(failures=[]),
        )
        return replace(a, metadata=metadata)

    result = solve_with_screened_boundary(
        solve, 10000.0, 8.0,
        deepen_on_any_screen_failure=True, restart_after_deepening=True,
        deepening_pressure_factor=4.0, early_deepening_escape_limit=0.1,
    )
    assert len(requests) == 2
    assert requests[1]["resume_supplied_structure_in_formal_flux_phase"] is False
    assert requests[1]["initial_gas_pressure"][-1] == pytest.approx(4.0 * a.gas_pressure[-1])
    segments = result.metadata["adaptive_domain_segments"]
    assert segments[0]["early_deepening"] and segments[0]["absorption_escape_bound"] == 0.55
    assert result.metadata["radiative_equilibrium_iterations_including_domain_adaptation"] == 12
    # The final allowed extension always completes a full solve.
    calls = []

    def always_shallow(teff, logg, **kwargs):
        calls.append(kwargs)
        if "early_lower_boundary_escape_limit" in kwargs:
            raise LowerBoundaryTooShallow(a, 0.55)
        return replace(a, metadata=dict(radiative_equilibrium_iterations=1,
                                        lower_boundary_absorption_escape_bound=0.5,
                                        lower_boundary_screening_tolerance=0.003,
                                        equilibrium_certificate=dict(failures=["boundary_screening"])))

    solve_with_screened_boundary(always_shallow, 10000.0, 8.0, maximum_domain_expansions=2,
                                 deepen_on_any_screen_failure=True, early_deepening_escape_limit=0.1)
    assert len(calls) == 3 and "early_lower_boundary_escape_limit" not in calls[-1]


def test_db_default_does_not_request_early_deepening(monkeypatch):
    # Early deepening stays opt-in: at 7000 K, log g 6 the deeper mesh started
    # from the first-phase state stalled, whereas deepening after a full solve
    # on the shallow mesh converges.
    from wd_spectra.models import stellar
    from wd_spectra.models.stellar import DBConfig, compute_db
    seen = {}
    monkeypatch.setattr(stellar, "_helium_tables", lambda data: (object(), object()))
    monkeypatch.setattr(stellar, "solve_with_screened_boundary",
                        lambda solver, *a, **k: seen.update(k) or gray_helium_atmosphere(9000.0, 7.0, n_depth=12))
    monkeypatch.setattr(stellar, "synthesize_helium_spectrum", lambda *a, **k: object())
    monkeypatch.setattr(stellar, "atmosphere_with_model_request_fingerprint", lambda a, f: a)
    monkeypatch.setattr(stellar, "warn_if_atmosphere_not_converged", lambda a, kind: "stub")
    compute_db(DBConfig(effective_temperature=9000.0, logg=7.0))
    assert "early_deepening_escape_limit" not in seen
    assert seen["deepen_on_any_screen_failure"] is True
    seen.clear()
    compute_db(DBConfig(effective_temperature=9000.0, logg=7.0, transfer_discretization="optical-depth"))
    assert "early_deepening_escape_limit" not in seen
