from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np
import pytest

from wd_spectra import _pg1159_structure as structure
from wd_spectra.atmosphere import gray_helium_atmosphere
from wd_spectra.nonlinear import NonlinearEvaluation


@dataclass(frozen=True)
class Model:
    population_nlte_fraction: float = 1.
    metal_population_relative_tolerance: float = 1e-2
    use_pg1159_response_jacobian: bool = False


def stub_equations(uphill):
    """Equations whose operator-split direction is biased at a loose closure.

    ``uphill(residual, tolerance)`` says whether the proposed correction points
    away from the root, which mimics a flux residual dominated by the
    eliminated-material closure bias.
    """
    built = []

    class Equations:
        def __init__(self, seed, model, wave, *, radiative_acceleration,
                     radiative_acceleration_scale, certification_stage,
                     material_tolerance_ceiling, resolved_material_closure=False):
            self.seed, self.model = seed, model
            self.resolved_material_closure = resolved_material_closure
            self.material_tolerance_ceiling = (
                model.metal_population_relative_tolerance
                if material_tolerance_ceiling is None else material_tolerance_ceiling)
            self.target = np.log(seed.temperature) - .07
            self.anchor, self.evaluations, self.anchors_seen = None, 0, []
            built.append(self)

        def initial_state(self):
            return np.log(self.seed.temperature)

        def evaluate(self, x, jacobian):
            self.evaluations += 1
            self.anchors_seen.append(self.anchor)
            residual = x - self.target
            tolerance = min(self.model.metal_population_relative_tolerance,
                            self.material_tolerance_ceiling)
            correction = residual if uphill(residual, tolerance) else -residual
            error = float(np.max(abs(residual)))
            diagnostics = {
                "surface_flux_ratio": 1. + error,
                "maximum_photospheric_total_flux_residual": error,
                "maximum_all_depth_total_flux_residual": error,
                "maximum_relative_cell_energy_balance_residual": .1 * error,
                "maximum_hydrostatic_log_pressure_residual": 0.,
                "electron_scattering_source_final_maximum_relative_residual": 0.,
                "lower_boundary_absorption_escape_bound": 0.,
                "_operator_split_correction": correction,
            }
            populations = SimpleNamespace(converged=True, tolerance=tolerance,
                                          evaluation=self.evaluations)
            return NonlinearEvaluation(
                residual, np.eye(len(x)) if jacobian else None,
                (self.seed, populations, diagnostics))

    return Equations, built


def solve(monkeypatch, model, uphill, **kwargs):
    equations, built = stub_equations(uphill)
    monkeypatch.setattr(structure, "PG1159Equations", equations)
    accepted = []
    seed = gray_helium_atmosphere(110000., 7., n_depth=8)
    result = structure._solve_stage(
        seed, model, np.array([100., 1000.]), include_radiative_acceleration=False,
        iteration_callback=lambda i, a, p, d: accepted.append((i, p)), **kwargs)
    return result, built, accepted


def test_rejected_full_nlte_direction_recloses_material_and_continues(monkeypatch):
    # One good step, then the 1e-2 closure proposes only uphill directions.
    result, built, accepted = solve(
        monkeypatch, Model(),
        lambda residual, tolerance: tolerance > 5e-3 and np.max(abs(residual)) < .03,
        stage_name="full-nlte-certification", maximum_iterations=10)
    assert [e.model.metal_population_relative_tolerance for e in built] == \
        pytest.approx([1e-2, 1e-3])
    assert built[1].material_tolerance_ceiling == pytest.approx(1e-3)
    # The anchor and evaluation count carry over; iteration numbering is continuous.
    assert built[1].anchors_seen[0] is accepted[0][1]
    assert built[1].evaluations > built[0].evaluations
    assert [i for i, _ in accepted] == [1, 2]
    assert accepted[1][1].tolerance == pytest.approx(1e-3)
    tightenings = result.atmosphere.metadata["material_tolerance_tightenings"]
    assert len(tightenings) == 1 and tightenings[0]["after_iteration"] == 1
    assert tightenings[0]["material_tolerance"] == pytest.approx(1e-3)
    assert result.atmosphere.metadata["residual_evaluations"] == built[1].evaluations
    assert result.nonlinear_result.converged
    assert result.converged


def test_rejection_at_minimum_tolerance_resolves_closure_once(monkeypatch):
    result, built, _ = solve(monkeypatch, Model(), lambda residual, tolerance: True,
                             stage_name="full-nlte-certification", maximum_iterations=4)
    assert [e.model.metal_population_relative_tolerance for e in built] == \
        pytest.approx([1e-2, 1e-3, 1e-4])
    assert [e.resolved_material_closure for e in built] == [False, False, True]
    tightenings = result.atmosphere.metadata["material_tolerance_tightenings"]
    assert [t["resolved_closure"] for t in tightenings] == [False, True]
    assert result.nonlinear_result.diagnostics.terminal_reason != "rejected-step-phase-handoff"
    assert not result.converged


def test_resolved_closure_continues_after_tightened_rejection(monkeypatch):
    # Uphill at 1e-2 and 1e-3 once near the root; only the resolved closure descends.
    result, built, accepted = solve(
        monkeypatch, Model(),
        lambda residual, tolerance: tolerance > 5e-4 and np.max(abs(residual)) < .03,
        stage_name="full-nlte-certification", maximum_iterations=10)
    assert [e.resolved_material_closure for e in built] == [False, False, True]
    assert built[2].model.metal_population_relative_tolerance == pytest.approx(1e-4)
    assert built[2].anchors_seen[0] is accepted[0][1]
    assert [i for i, _ in accepted] == [1, 2]
    assert accepted[1][1].tolerance == pytest.approx(1e-4)
    assert result.converged


def test_minimum_tolerance_at_initial_closure_goes_straight_to_resolved(monkeypatch):
    @dataclass(frozen=True)
    class Floored(Model):
        pg1159_minimum_material_tolerance: float = 1e-2
    result, built, _ = solve(monkeypatch, Floored(), lambda residual, tolerance: True,
                             stage_name="full-nlte-certification", maximum_iterations=4)
    assert [e.resolved_material_closure for e in built] == [False, True]
    assert built[1].model.metal_population_relative_tolerance == pytest.approx(1e-4)


def test_partial_nlte_stage_never_tightens_material(monkeypatch):
    result, built, _ = solve(
        monkeypatch, Model(population_nlte_fraction=.5), lambda residual, tolerance: True,
        certification_stage=False, material_tolerance_ceiling=1e-2,
        stage_name="half-nlte-population-bridge", maximum_iterations=4)
    assert len(built) == 1
    assert result.atmosphere.metadata["material_tolerance_tightenings"] == ()


def test_run_without_rejection_builds_one_closure(monkeypatch):
    result, built, accepted = solve(monkeypatch, Model(), lambda residual, tolerance: False,
                                    stage_name="full-nlte-certification", maximum_iterations=10)
    assert len(built) == 1
    assert result.atmosphere.metadata["material_tolerance_tightenings"] == ()
    assert [i for i, _ in accepted] == [1, 2]
    assert result.converged


@pytest.mark.parametrize("field,value", [
    ("pg1159_rejected_step_material_tightening", 1.),
    ("pg1159_rejected_step_material_tightening", 0.),
    ("pg1159_minimum_material_tolerance", 0.),
    ("pg1159_resolved_material_tolerance", 0.),
])
def test_invalid_tightening_settings_are_rejected(monkeypatch, field, value):
    model = SimpleNamespace(**vars(Model()), **{field: value})
    monkeypatch.setattr(structure, "PG1159Equations", stub_equations(lambda *a: False)[0])
    with pytest.raises(ValueError, match="material tightening"):
        structure._solve_stage(gray_helium_atmosphere(110000., 7., n_depth=8), model,
                               np.array([100., 1000.]), include_radiative_acceleration=False)


def test_rejected_direction_secant_fits_derivative_through_curvature():
    rng = np.random.default_rng(1)
    n = 6
    response, x0, direction, other = rng.normal(size=(n, n)), *rng.normal(size=(3, n))
    residual = lambda x: response @ (x - x0) + .3 * ((x - x0) @ direction) ** 2 + 1.
    trace = [(x0 - 1, residual(x0 - 1)), (x0.copy(), residual(x0))]
    trace += [(x0 + f * other, residual(x0 + f * other)) for f in (1, .5)]
    trace += [(x0 + f * direction, residual(x0 + f * direction)) for f in (1, .5, .25, .125)]
    unit, action = structure.rejected_direction_secant(trace, x0)
    np.testing.assert_allclose(unit, direction / np.linalg.norm(direction))
    np.testing.assert_allclose(action, response @ unit, atol=1e-12)
    assert structure.rejected_direction_secant(trace[:3], x0) is None
    assert structure.rejected_direction_secant(trace, x0 + 5) is None


def test_secant_correction_reproduces_measured_actions_only():
    rng = np.random.default_rng(2)
    matrix = rng.normal(size=(5, 5))
    q = np.linalg.qr(rng.normal(size=(5, 2)))[0]
    measured = rng.normal(size=(5, 2))
    corrected = structure.secant_corrected(matrix, [(q[:, 0], measured[:, 0]), (q[:, 1], measured[:, 1])])
    np.testing.assert_allclose(corrected @ q, measured, atol=1e-12)
    complement = np.eye(5) - q @ q.T
    np.testing.assert_allclose(corrected @ complement, matrix @ complement, atol=1e-12)


def test_tightened_closure_carries_into_the_next_stage(monkeypatch):
    result, built, _ = solve(
        monkeypatch, Model(),
        lambda residual, tolerance: tolerance > 5e-3 and np.max(abs(residual)) < .03,
        certification_stage=False, stage_name="full-nlte-relaxation", maximum_iterations=10)
    closure = result.closure_state
    assert closure["tolerance"] == pytest.approx(1e-3) and not closure["resolved"]
    carried, carried_built, _ = solve(
        monkeypatch, Model(), lambda residual, tolerance: False,
        stage_name="full-nlte-certification", maximum_iterations=10, closure_state=closure)
    assert carried_built[0].model.metal_population_relative_tolerance == pytest.approx(1e-3)
    assert carried_built[0].material_tolerance_ceiling == pytest.approx(1e-3)
    assert carried.converged


def test_stage_without_rejection_carries_no_closure(monkeypatch):
    result, _, _ = solve(monkeypatch, Model(), lambda residual, tolerance: False,
                         stage_name="full-nlte-certification", maximum_iterations=10)
    assert result.closure_state is None


def test_public_continuation_passes_closure_only_to_full_nlte_stages(monkeypatch):
    @dataclass(frozen=True)
    class Continuation:
        population_nlte_fraction: float = 1.
        metal_population_damping: float = .25
        helium_population_damping: float = .4
        metal_population_acceleration_depth: int = 4
        coupled_population_acceleration_depth: int = 80
        metal_population_relative_tolerance: float = 1e-2
        metal_population_iterations: int = 120
        use_pg1159_response_jacobian: bool = False
    received = []
    closure = {"tolerance": 1e-4, "resolved": True, "secant_corrections": ()}

    def stage(a, m, w, **kwargs):
        received.append((kwargs["stage_name"], kwargs.get("closure_state")))
        return structure.PG1159AtmosphereResult(
            replace(a, metadata={"elapsed_seconds": 1.}), object(),
            SimpleNamespace(converged=True, iterations=1),
            closure_state=closure if kwargs["stage_name"] == "half-nlte-population-bridge"
            or kwargs["stage_name"] == "full-nlte-relaxation" else None)
    monkeypatch.setattr(structure, "_solve_stage", stage)
    structure.solve_pg1159_atmosphere(gray_helium_atmosphere(110000., 7., n_depth=8), Continuation(),
                                      np.array([100., 1000.]), cold_start=True)
    assert received == [("planck-initializer", None), ("half-nlte-population-bridge", None),
                        ("full-nlte-relaxation", closure), ("full-nlte-certification", closure)]
