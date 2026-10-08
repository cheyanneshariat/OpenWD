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
