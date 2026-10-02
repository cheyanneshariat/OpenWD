"""Native profile kernels preserve the independently evaluated NumPy profiles."""
import sys

import numpy as np
import pytest

from wd_spectra import metals
from wd_spectra.models.common import ModelData
from wd_spectra.models.d6 import _atomic_inputs


@pytest.fixture(scope="module")
def native():
    backend = metals._rt
    if not hasattr(backend, "stark_manifold_bins"):
        pytest.skip("optional Stark C kernels not built")
    return backend


@pytest.fixture(scope="module")
def patterns():
    db = _atomic_inputs(ModelData.default(), ("Mg",))[0]
    ion = db.ions["Mg", 1]
    def pattern(label):
        level = next(level for level in ion.levels if level.label == label)
        return metals.rydberg_stark_manifold(db, ion, level)
    return [pattern("2p6.7g.(2G<9/2>)"), pattern("2p6.10g.(2G<9/2>)"),
            pattern("2p6.4f.(2Fo<7/2>)")]


def test_native_pseudo_voigt_matches_numpy(native, monkeypatch):
    rng = np.random.default_rng(914)
    for _ in range(80):
        center = rng.uniform(100., 100000.)
        sigma, gamma = 10. ** rng.uniform(-7, 2, 2)
        wave = center + np.r_[0., np.geomspace(1.e-8, 1000., 120)]
        reference = metals._pseudo_voigt_profile_per_angstrom(wave, center, sigma, gamma)
        # The manifold path uses the exact Voigt core; check the C kernel itself.
        actual = np.empty_like(wave)
        native.pseudo_voigt_profile(wave, center, sigma, gamma, actual)
        np.testing.assert_allclose(actual, reference, rtol=2.e-13, atol=1.e-290)


@pytest.mark.parametrize("lower_index", [None, 2, 0])
@pytest.mark.parametrize("maximum_beta", [0.013, 1.3, 7.13, np.inf])
def test_native_manifold_profile_matches_numpy(native, patterns, monkeypatch,
                                               lower_index, maximum_beta):
    rng = np.random.default_rng(318)
    center = 4333.17
    wave = np.unique(np.r_[np.linspace(center - 20., center + 20., 1301),
                            center - np.geomspace(20., 3000., 300),
                            center + np.geomspace(20., 3000., 300)])
    lower = None if lower_index is None else patterns[lower_index]
    for _ in range(4):
        args = (wave, center, float(10.**rng.uniform(-2., -.5)),
                float(10.**rng.uniform(-3., 1.)), float(10.**rng.uniform(6., 12.)),
                patterns[1], lower, maximum_beta)
        kwargs = dict(correlation=.2, radiator_core_charge=2., support_half_width=3000.)
        monkeypatch.setattr(metals, "_rt", None)
        reference = metals.manifold_quasistatic_line_profile(*args, **kwargs)
        monkeypatch.setattr(metals, "_rt", native)
        actual = metals.manifold_quasistatic_line_profile(*args, **kwargs)
        # Compare the complete profile, including its very faint tail.  The
        # tolerance allows the changed summation order in C bin deposition.
        np.testing.assert_allclose(actual, reference, rtol=2.e-7,
                                   atol=2.e-10*np.max(reference))


def test_native_voigt_rejects_invalid_buffers(native):
    out = np.empty(5)
    with pytest.raises(TypeError):
        native.pseudo_voigt_profile(np.ones(5, dtype=np.float32), 5., .1, .2, out)
    with pytest.raises(ValueError):
        native.pseudo_voigt_profile(np.ones(10)[::2], 5., .1, .2, out)
    with pytest.raises(ValueError):
        native.pseudo_voigt_profile(np.ones(4), 5., .1, .2, out)
    out.flags.writeable = False
    with pytest.raises((ValueError, BufferError)):
        native.pseudo_voigt_profile(np.ones(5), 5., .1, .2, out)


def test_native_bins_reject_invalid_shapes_and_nonfinite_tracks(native):
    shift = np.array([[0., 1.], [0., 2.], [0., 3.]])
    weight = np.full_like(shift, .5)
    probability = np.array([.4, .6])
    fraction = np.ones(2)
    args = [shift, weight, None, None, probability, fraction,
            -1., 5., .1, 1., 0., np.empty(101), np.empty(0)]
    native.stark_manifold_bins(*args)
    for index, replacement in [(1, np.ones((2, 2))), (4, np.ones(3)),
                                (11, np.empty(100)), (2, shift)]:
        invalid = list(args)
        invalid[index] = replacement
        with pytest.raises(ValueError):
            native.stark_manifold_bins(*invalid)
    shift[1, 1] = np.nan
    with pytest.raises(ValueError):
        native.stark_manifold_bins(*args)


@pytest.mark.parametrize("coarse_size", [0, 7])
def test_native_finish_matches_interpolation_at_edges(native, coarse_size):
    center, fine_step, coarse_step, fine_half = 5000., .125, 2., .43
    wave = center + np.array([-100., -6., -3., -.43, -.2, 0., .2, .43,
                               .5, 3., 6., 100., np.inf, -np.inf, np.nan])
    fine = np.array([.2, -.00001, .3, .8, .3, -.00001, .2])
    coarse = np.linspace(.001, .01, coarse_size)
    impact = metals._pseudo_voigt_profile_per_angstrom(wave, center, .03, .04)
    reference = .2 * impact
    inside = np.abs(wave-center) <= fine_half
    reference[inside] += np.interp(
        wave[inside]-center, fine_step*(np.arange(7)-3),
        np.maximum(fine, 0.)/fine_step)
    if coarse_size:
        reference[~inside] += np.interp(
            wave[~inside]-center, coarse_step*(np.arange(coarse_size)-coarse_size//2),
            coarse/coarse_step) + .7*impact[~inside]
    output = np.empty_like(wave)
    native.stark_profile_finish(wave, fine, coarse, center, .03, .04,
                               fine_step, coarse_step, fine_half, .2, .7, .9, output)
    np.testing.assert_allclose(output, reference/.9, rtol=2.e-13, atol=1.e-16)


def test_native_finish_rejects_invalid_buffers(native):
    args = [np.ones(5), np.ones(7), np.ones(3), 5., .1, .2,
            .1, 1., .3, .2, .7, .9, np.empty(5)]
    for index, replacement in [(0, np.ones(5, dtype=np.float32)),
                                (1, np.ones(6)), (2, np.ones(4)),
                                (6, 0.), (11, 0.), (12, np.empty(4))]:
        invalid = list(args)
        invalid[index] = replacement
        with pytest.raises((TypeError, ValueError)):
            native.stark_profile_finish(*invalid)
    args[-1].flags.writeable = False
    with pytest.raises((ValueError, BufferError)):
        native.stark_profile_finish(*args)


@pytest.mark.parametrize("kernel", ["pseudo_voigt_profile", "stark_manifold_bins",
                                    "stark_profile_finish"])
def test_native_kernels_release_buffers_on_success_and_failure(native, kernel):
    if kernel == "pseudo_voigt_profile":
        original = [np.ones(5), 5., .1, .2, np.empty(5)]
    elif kernel == "stark_manifold_bins":
        original = [np.arange(6., dtype=float).reshape(3, 2), np.full((3, 2), .5),
                    np.zeros((3, 2)), np.full((3, 2), .5), np.array([.4, .6]),
                    np.ones(2), -1., .3, .1, 1., 0., np.empty(7), np.empty(7)]
    else:
        original = [np.ones(5), np.ones(7), np.ones(3), 5., .1, .2,
                    .1, 1., .3, .2, .7, .9, np.empty(5)]
    indices = [i for i, value in enumerate(original) if isinstance(value, np.ndarray)]
    for failure in [None, *indices]:
        args = list(original)
        if failure is not None:
            args[failure] = args[failure].astype(np.float32)
        before = [sys.getrefcount(args[i]) for i in indices]
        for _ in range(3):
            if failure is None:
                getattr(native, kernel)(*args)
            else:
                with pytest.raises(TypeError):
                    getattr(native, kernel)(*args)
        assert [sys.getrefcount(args[i]) for i in indices] == before


def test_field_cache_uses_complete_physical_state():
    from wd_spectra.eos import hooper_microfield_cumulative_probability
    function = metals._stark_field_probabilities
    function.cache_clear()
    for coupling, cutoff, correlation, charge, stride in [
        (2.e10, 7.13, .2, 2., 1), (3.e10, 7.13, .2, 2., 1),
        (2.e10, 8.13, .2, 2., 1), (2.e10, 7.13, .3, 2., 1),
        (2.e10, 7.13, .2, 1., 1), (2.e10, 7.13, .2, 2., 3),
    ]:
        beta, cdf, probability, fraction = function(coupling, cutoff, correlation, charge, stride)
        expected = hooper_microfield_cumulative_probability(
            np.minimum(metals._STARK_MANIFOLD_FIELD_HZ[::stride]/coupling, cutoff),
            correlation, charge)
        np.testing.assert_array_equal(cdf, expected)
        assert not cdf.flags.writeable and not probability.flags.writeable
        assert not beta.flags.writeable and not fraction.flags.writeable
    assert function.cache_info().misses == 6
    function(2.e10, 7.13, .2, 2., 1)
    assert function.cache_info().hits == 1
