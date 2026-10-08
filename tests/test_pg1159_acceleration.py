import numpy as np
import pytest
from wd_spectra._pg1159_acceleration import population_update


@pytest.mark.parametrize("weights", [None, np.array([1.0, 1.0, 0.01])])
def test_population_svd_resolves_weak_coupling_without_changing_fixed_point(weights):
    target = np.array([0.12, -0.08, 0.2])
    contraction = np.array([0.1, 0.8, 0.99999])
    x = np.zeros(3)
    history = []
    for _ in range(12):
        g = target + contraction * (x - target)
        proposed, reason = population_update(
            x,
            g,
            history,
            depth=8,
            mixing=0.65,
            maximum_step=0.75,
            residual_weights=weights,
        )
        x = 0.35 * x + 0.65 * g if proposed is None else proposed
    np.testing.assert_allclose(x, target, rtol=0.0, atol=1e-8)


def test_population_svd_retains_step_safeguard():
    history = []
    x = np.zeros(2)
    for _ in range(4):
        g = 0.999 * x + 1.0
        proposed, reason = population_update(
            x, g, history, depth=5, mixing=0.65, maximum_step=0.75
        )
        if proposed is not None:
            assert np.max(abs(proposed - x)) <= 0.75
        x = x + 0.001
    assert proposed is None and reason == "step"


@pytest.mark.parametrize(
    "weight", [np.ones(3), np.array([1.0, np.nan]), np.array([1.0, -1.0])]
)
def test_population_svd_validates_weights(weight):
    history = [(np.zeros(2), np.ones(2)), (np.ones(2), np.zeros(2))]
    with pytest.raises(ValueError, match="weights"):
        population_update(
            np.ones(2),
            np.ones(2) * 2,
            history,
            depth=5,
            mixing=0.65,
            maximum_step=0.75,
            residual_weights=weight,
        )


@pytest.mark.parametrize("tail_weight", [1e-24, 1e-9])
def test_empty_population_history_cannot_veto_physical_acceleration(tail_weight):
    history = []
    for i in range(3):
        x = np.array([i * 0.01, i * 100.0])
        g = x + np.array([1e-4 * (0.1 - x[0]), 0.0])
        proposed, reason = population_update(
            x,
            g,
            history,
            depth=8,
            mixing=0.65,
            maximum_step=0.75,
            residual_weights=np.array([1.0, tail_weight]),
        )
    assert reason == "accepted"
    np.testing.assert_allclose(proposed[0], 0.1, atol=1e-10)
    np.testing.assert_allclose(proposed[1], x[1], atol=1e-10)
    assert np.max(abs(proposed - x)) < 0.75


def test_plain_empty_tail_update_is_not_limited_as_an_extrapolation():
    history = []
    for i in range(3):
        x = np.array([i*.01, -100.+i])
        g = x + np.array([1e-4*(.1-x[0]), -10.])
        proposed, reason = population_update(x, g, history, depth=8, mixing=.65,
            maximum_step=.75, residual_weights=np.array([1., 1e-24]))
    assert reason == 'accepted'
    np.testing.assert_allclose(proposed[0], .1, atol=1e-10)
    np.testing.assert_allclose(proposed[1], .35*x[1]+.65*g[1], atol=1e-10)






def test_small_resolved_tail_is_bounded_without_vetoing_the_atom():
    history = []
    for i in range(3):
        x = np.array([i * .01, i * 100.])
        g = x + np.array([1e-4 * (.1 - x[0]), 0.])
        proposed, reason = population_update(x, g, history, depth=8,
            mixing=.65, maximum_step=.75, residual_weights=np.array([1., 1e-5]))
    assert reason == "accepted"
    np.testing.assert_allclose(proposed[0], .1, atol=1e-10)
    assert abs(proposed[1] - x[1]) <= .75

def test_weak_curved_mode_can_cross_a_temporary_residual_increase():
    # The second variable must follow the square of the weak first mode.
    # Requiring a monotone inner residual traps this regular fixed point;
    # the independent final rate check, not monotonicity, certifies closure.
    x = np.zeros(2)
    history = []
    for _ in range(12):
        residual = np.array([1e-4 * (.1 - x[0]), x[0] ** 2 - x[1]])
        proposed, _ = population_update(x, x + residual, history, depth=80,
            mixing=.65, maximum_step=.75)
        x = x + .65 * residual if proposed is None else proposed
    np.testing.assert_allclose(x, [.1, .01], rtol=0., atol=1e-9)


def test_recycled_secants_do_not_include_the_change_in_map_offset():
    from wd_spectra._pg1159_acceleration import PopulationHistory
    history = PopulationHistory()
    x = np.zeros(2)
    target = np.array([.1, -.2])
    contraction = np.array([.2, .9999])
    for _ in range(8):
        g = target + contraction * (x - target)
        proposal, _ = population_update(x, g, history, depth=20,
            mixing=.65, maximum_step=.75)
        x = .35*x + .65*g if proposal is None else proposal
    np.testing.assert_allclose(x, target, atol=1e-9)
    history.restart(20, retain=True)
    new_target = target + np.array([.01, -.03])
    g = new_target + contraction * (x - new_target)
    proposal, reason = population_update(x, g, history, depth=20,
        mixing=.65, maximum_step=.75)
    assert reason == "accepted"
    np.testing.assert_allclose(proposal, new_target, atol=1e-8)
    # A substantial change in the EOS discards this approximation explicitly.
    history.restart(20, retain=False)
    proposal, reason = population_update(x, g, history, depth=20,
        mixing=.65, maximum_step=.75)
    assert proposal is None and reason == "history"


def test_zero_depth_restart_discards_acceleration_history():
    from wd_spectra._pg1159_acceleration import PopulationHistory

    history = PopulationHistory()
    history.extend(
        [(np.array([value]), np.array([value + 1.0])) for value in range(4)]
    )
    history.restart(0, retain=True)

    assert history == []
    assert history.retained == []


def test_disabled_acceleration_does_not_collect_history():
    history = []

    proposal, reason = population_update(
        np.zeros(2),
        np.ones(2),
        history,
        depth=0,
        mixing=0.65,
        maximum_step=0.75,
    )

    assert proposal is None
    assert reason == "history"
    assert history == []
