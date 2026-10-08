"""The nonlinear tangent must differentiate the local energy rows actually solved."""

from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
from wd_spectra import adaptive_structure as adaptive, gray_helium_atmosphere
from wd_spectra.constants import STEFAN_BOLTZMANN


@pytest.mark.parametrize("mode", ["optical-depth", "column-mass"])
@pytest.mark.parametrize("fraction", [0.0, 0.98])
@pytest.mark.parametrize(
    "convective,use_preconditioner", [(False, False), (False, True), (True, False)]
)
def test_full_local_energy_evaluator_and_tangent(
    monkeypatch, mode, fraction, convective, use_preconditioner
):
    seed = gray_helium_atmosphere(6000.0, 8.0, n_depth=8)
    wave = np.geomspace(500.0, 1e5, 19)
    shape = (len(wave), seed.n_depth)
    base = np.broadcast_to((wave[:, None] / 5000.0) ** 0.1, shape)

    def absorption(a):
        return (
            (1 - fraction) * base * (a.temperature / seed.temperature) ** -0.3
        )

    def scattering(a):
        return fraction * base * (a.temperature / seed.temperature) ** 0.2

    def thermodynamics(a):
        return SimpleNamespace(
            specific_heat_constant_pressure=np.full(8, 1e8),
            density_temperature_derivative=np.ones(8),
            adiabatic_temperature_gradient=np.full(8, 0.01),
        )

    original = adaptive.solve_trust_region_newton
    captured = []

    def checked(initial, evaluate, **options):
        ev = evaluate(initial, True)
        p = ev.payload
        captured.append(True)
        expected = (
            p["total_flux_interface"] / (STEFAN_BOLTZMANN * 6000.0**4) - 1.0
        )
        expected[:-1] -= p["cell_energy_balance_relative_residual"]
        np.testing.assert_allclose(
            ev.residual, expected, rtol=1e-13, atol=1e-13
        )
        row_scale = np.maximum(1.0, np.max(abs(ev.jacobian), axis=1))
        for column in (0, 2, 7):
            h = 2e-6
            step = np.eye(8)[column] * h
            numerical = (
                evaluate(initial + step, False).residual
                - evaluate(initial - step, False).residual
            ) / (2 * h)
            np.testing.assert_allclose(
                numerical / row_scale,
                ev.jacobian[:, column] / row_scale,
                atol=2e-6,
                rtol=2e-4,
            )
        assert options["allow_initial_convergence"] is False
        return original(initial, evaluate, **options)

    monkeypatch.setattr(adaptive, "solve_trust_region_newton", checked)
    result = adaptive.solve_adaptive_lte_structure(
        seed,
        wave,
        with_temperature=lambda t: replace(seed, temperature=t),
        true_absorption=absorption,
        scattering_opacity=scattering,
        rosseland_opacity=lambda a: np.ones(8),
        thermodynamics=thermodynamics,
        mixing_length_alpha=1.25 if convective else None,
        max_iterations=1,
        flux_tolerance=3e-3,
        temperature_tolerance=3e-4,
        n_angle=3,
        initial_temperature_was_supplied=False,
        use_initial_bolometric_rescaling=False,
        # The default conditioner must not suppress local energy equations
        # when convection is disabled. Flux alone barely constrains thin cells.
        use_convective_gradient_preconditioner=use_preconditioner,
        maximum_formal_flux_continuations=0,
        enforce_local_energy_balance=True,
        transfer_discretization=mode,
        project_initial_convective_gradient=False,
    )
    assert captured


def test_public_run_rejects_continuation_before_any_physics_work(
    monkeypatch, tmp_path
):
    from wd_spectra.models import automatic, DBConfig

    def forbidden(*a, **k):
        raise AssertionError("must reject before screening")

    monkeypatch.setattr(automatic, "select_physics", forbidden)
    with pytest.raises(TypeError, match="initial_checkpoint"):
        automatic.run_model(
            DBConfig(), tmp_path / "run", initial_checkpoint="previous.npz"
        )
    assert not (tmp_path / "run").exists()
