"""Native GTH elimination preserves positive weak links and number conservation."""
import numpy as np
import pytest
from wd_spectra import light_metal_nlte as metal


def native():
    if metal._rt is None or not hasattr(metal._rt, "positive_rate_equilibrium"):
        pytest.skip("compiled positive rate solver unavailable")
    return metal._rt


@pytest.mark.parametrize("count", [1, 2, 17, 152, 290])
def test_positive_solver_preserves_known_detailed_balance(count):
    native()
    random = np.random.default_rng(187)
    expected = np.geomspace(1e-60, 1.0, count)
    conductance = 10.0 ** random.uniform(-20.0, 12.0, (count, count))
    conductance = conductance + conductance.T
    matrix = np.ascontiguousarray(conductance / expected[None, :])
    # The diagonal is immaterial to the positive method, even if cancellation
    # has destroyed its weak links in the usual generator representation.
    np.fill_diagonal(matrix, -np.sum(matrix, axis=0))
    original = matrix.copy()
    weight = random.uniform(1.0, 3.0, count)
    total = 1e20
    reference = metal._positive_rate_equilibrium_python(matrix, weight, total)
    actual = metal._positive_rate_equilibrium(matrix, weight, total)
    expected *= total / np.dot(weight, expected)
    np.testing.assert_allclose(actual, reference, rtol=3e-13, atol=0.0)
    np.testing.assert_allclose(actual, expected, rtol=3e-13, atol=0.0)
    np.testing.assert_allclose(np.dot(weight, actual), total, rtol=3e-15)
    np.testing.assert_array_equal(matrix, original)
    assert np.all(actual >= 0.0)


@pytest.mark.parametrize(
    "problem", ["negative", "nan", "disconnected", "negative_total"]
)
def test_native_rate_failure_retains_recoverable_exception(problem):
    native()
    matrix = np.ones((3, 3))
    total = 1.0
    if problem == "negative":
        matrix[0, 1] = -1.0
    elif problem == "nan":
        matrix[0, 1] = np.nan
    elif problem == "disconnected":
        matrix[:2, 2] = 0.0
    else:
        total = -1.0
    for solver in (
        metal._positive_rate_equilibrium_python,
        metal._positive_rate_equilibrium,
    ):
        with pytest.raises(metal.NonphysicalPopulationError):
            solver(matrix, np.ones(3), total)


@pytest.mark.parametrize(
    "case", ["nonsquare", "weights", "output", "dtype", "readonly"]
)
def test_native_rate_solver_rejects_invalid_buffers(case):
    compiled = native()
    matrix = np.ones((3, 3))
    weight = np.ones(3)
    output = np.empty(3)
    if case == "nonsquare":
        matrix = np.ones((3, 4))
    elif case == "weights":
        weight = np.ones(2)
    elif case == "output":
        output = np.empty(2)
    elif case == "dtype":
        matrix = matrix.astype(np.float32)
    elif case == "readonly":
        output.flags.writeable = False
    with pytest.raises((TypeError, ValueError, BufferError)):
        compiled.positive_rate_equilibrium(matrix, weight, 1.0, output)


def test_positive_solver_falls_back_without_extension(monkeypatch):
    matrix = np.array([[0.0, 2.0], [3.0, 0.0]])
    monkeypatch.setattr(metal, "_rt", None)
    np.testing.assert_allclose(
        metal._positive_rate_equilibrium(matrix, np.ones(2), 10.0), [4.0, 6.0]
    )
