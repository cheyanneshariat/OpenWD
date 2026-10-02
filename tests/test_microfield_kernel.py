"""The shared microfield optimization preserves the original logistic exactly."""
import numpy as np
import pytest

from wd_spectra.eos import hooper_microfield_cumulative_probability


def reference(beta, correlation, charge):
    factor = (1.0 + correlation)**3.15
    c1 = .1402*(factor + 4.0*(charge-1.0)*correlation**3)
    c2 = .1285*factor
    log_beta = np.log(np.maximum(beta, np.finfo(np.float64).tiny))
    log_ratio = np.log(c1) + 3.0*log_beta - np.logaddexp(0.0, np.log(c2)+1.5*log_beta)
    with np.errstate(over='ignore', invalid='ignore'):
        return np.where(log_ratio >= 0.,
            1./(1.+np.exp(-np.minimum(log_ratio,745.))),
            np.exp(np.maximum(log_ratio,-745.))/(1.+np.exp(np.maximum(log_ratio,-745.))))


@pytest.mark.parametrize('shape', [(), (48,), (577,), (37,48)])
@pytest.mark.parametrize('charge', [1.,2.,8.])
def test_hooper_reuse_is_bitwise_identical(shape, charge):
    rng = np.random.default_rng(572)
    beta = 10.**rng.uniform(-300.,300.,shape)
    correlation = rng.uniform(0.,.8,shape[-1:] if shape else ())
    expected = np.asarray(reference(beta, correlation, charge))
    actual = hooper_microfield_cumulative_probability(beta, correlation, charge)
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))


def test_hooper_extremes_do_not_evaluate_an_overflowing_unused_branch():
    beta = np.r_[0., np.geomspace(1.e-300,1.e300,1000)]
    with np.errstate(over='raise'):
        actual = hooper_microfield_cumulative_probability(beta, .2, 2.)
    expected = reference(beta, np.asarray(.2), 2.)
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
    assert np.all(np.isfinite(actual)) and np.all(np.diff(actual) >= 0.)
