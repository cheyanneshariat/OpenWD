"""The optional fused kernels retain the released frequency-space algorithm."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from wd_spectra import metals
from wd_spectra.constants import LIGHT_SPEED


@pytest.fixture(scope="module")
def native_frequency():
    backend = metals._rt
    names = ("frequency_voigt_profile", "stark_frequency_profile_finish",
             "stark_manifold_bins")
    if not all(hasattr(backend, name) for name in names):
        pytest.skip("optional frequency-space Stark C kernels not built")
    return backend


def frequency_reference(wave, center, sigma, gamma):
    # This retained NumPy implementation is independent of the new C entry
    # points and uses the released frequency coordinate, including its wings.
    with np.errstate(all="ignore"):
        return metals._voigt_profile_per_angstrom(wave, center, sigma, gamma)


def finish_reference(wave, fine, coarse, center, sigma, gamma, fine_step,
                     coarse_step, fine_half, core_weight, inner_mass, bound):
    impact = frequency_reference(wave, center, sigma, gamma)
    offset = wave - center
    inside = np.abs(offset) <= fine_half
    result = core_weight * impact
    result[inside] += np.interp(
        offset[inside], fine_step * (np.arange(len(fine)) - len(fine)//2),
        np.maximum(fine, 0.) / fine_step)
    if len(coarse):
        result[~inside] += np.interp(
            offset[~inside], coarse_step * (np.arange(len(coarse)) - len(coarse)//2),
            coarse / coarse_step) + inner_mass * impact[~inside]
    return result / bound


@pytest.mark.parametrize("y", [.001, .05, .5, 1., 4., 14., 16.])
def test_frequency_profile_w4_region_boundaries(native_frequency, y):
    center, sigma = 4333.17, .03
    # Probe both sides of every applicable W4 boundary.  The gap exceeds the
    # wavelength-to-frequency roundoff, so both implementations see the same
    # region rather than an ambiguous point exactly on a piecewise boundary.
    boundaries = [value for value in (15.-y, 5.5-y, (y+.176)/.195)
                  if value > 0.]
    x = np.array([0., -100., 100.] + [sign*(edge+side*1.e-5)
        for edge in boundaries for sign in (-1., 1.) for side in (-1., 1.)])
    sigma_nu = sigma * LIGHT_SPEED / (center * 1.e-8)**2 * 1.e-8
    wave = LIGHT_SPEED / (LIGHT_SPEED/(center*1.e-8)
                          + x*sigma_nu*np.sqrt(2.)) * 1.e8
    gamma = y * sigma * np.sqrt(2.)
    reference = frequency_reference(wave, center, sigma, gamma)
    actual = np.empty_like(wave)
    assert native_frequency.frequency_voigt_profile(
        wave, center, sigma, gamma, actual) is None
    np.testing.assert_allclose(actual, reference, rtol=1.e-12,
                               atol=1.e-12*np.max(reference))


@pytest.mark.parametrize("center,sigma,gamma", [
    (100., 1.e-4, .005), (5000., .03, 0.), (100000., .03, 2.),
    (5000., 0., .05), (5000., -.03, -.05), (5000., 2., .005),
])
def test_frequency_profile_width_limits_and_nonfinite_wave(
        native_frequency, center, sigma, gamma):
    wave = np.array([center, center*1.1, center*.9, np.inf, -np.inf, np.nan])
    reference = frequency_reference(wave, center, sigma, gamma)
    output = np.empty_like(wave)
    native_frequency.frequency_voigt_profile(wave, center, sigma, gamma, output)
    np.testing.assert_allclose(output, reference, rtol=1.e-12,
                               atol=1.e-12*np.nanmax(reference), equal_nan=True)


def test_frequency_profile_retains_faint_asymmetric_wings(native_frequency):
    center = 4333.17
    wave = center * np.array([.5, 1.5])
    output = np.empty_like(wave)
    native_frequency.frequency_voigt_profile(wave, center, .03, .005, output)
    # A wavelength-centred pseudo-Voigt has equal wings at these symmetric
    # offsets.  Frequency detuning instead gives the far-wing ratio nine.
    assert output[0] > 0.
    assert output[1] / output[0] == pytest.approx(9., rel=1.e-7)
    np.testing.assert_allclose(output, frequency_reference(wave, center, .03, .005),
                               rtol=1.e-12, atol=0.)


@pytest.mark.parametrize("coarse_size", [0, 1, 7])
@pytest.mark.parametrize("fine_step", [.1, .125])
def test_frequency_finish_clips_fine_interpolates_edges_and_normalizes(
        native_frequency, coarse_size, fine_step):
    center, coarse_step, fine_half = 5000., 2., .43
    fine = np.array([.2, -.1, .3, .8, .3, -.1, .2])
    # Coarse masses deliberately retain a signed roundoff tail: only the
    # convolved fine distribution is clipped by the released algorithm.
    coarse = np.linspace(-.001, .01, coarse_size)
    grid_points = center + fine_step * (np.arange(len(fine)) - len(fine)//2)
    wave = np.r_[center + np.array([-100., -6., -3., -.43, -.15, 0.,
                                   .15, .43, .5, 3., 6., 100.]),
                 grid_points, np.nextafter(grid_points, -np.inf),
                 np.nextafter(grid_points, np.inf), np.inf, -np.inf, np.nan]
    original = [value.copy() for value in (wave, fine, coarse)]
    args = (wave, fine, coarse, center, .03, .04, fine_step, coarse_step,
            fine_half, .2, .7)
    output = np.empty_like(wave)
    assert native_frequency.stark_frequency_profile_finish(*args, .9, output) is None
    expected = finish_reference(*args, .9)
    np.testing.assert_allclose(output, expected, rtol=1.e-12, atol=1.e-12,
                               equal_nan=True)
    unnormalized = np.empty_like(wave)
    native_frequency.stark_frequency_profile_finish(*args, 1., unnormalized)
    np.testing.assert_allclose(output*.9, unnormalized, rtol=1.e-14, atol=1.e-15,
                               equal_nan=True)
    for current, before in zip((wave, fine, coarse), original):
        np.testing.assert_array_equal(current, before)


@pytest.fixture(scope="module")
def simple_manifold():
    field = metals._STARK_MANIFOLD_FIELD_HZ
    shift = field[:, None] * np.array([-10., 12.])
    weight = np.broadcast_to(np.array([.4, .6]), shift.shape).copy()
    return metals.RydbergStarkManifold(
        10, 4, (3, 4), shift, weight, np.sum(weight*shift**2, axis=1),
        np.max(np.abs(shift), axis=1))


@pytest.mark.parametrize("use_impact,use_finish", [
    (False, False), (True, False), (False, True), (True, True),
])
def test_partial_native_apis_preserve_post_fft_assembly(
        native_frequency, simple_manifold, monkeypatch, use_impact, use_finish):
    center = 4333.17
    wave = np.unique(np.r_[np.linspace(center-20., center+20., 401),
                           center + np.array([-2000., -1000., 1000., 2000.])])
    args = (wave, center, .03, .005, 2.e10, simple_manifold, None, 7.13)
    kwargs = dict(correlation=.2, radiator_core_charge=2., support_half_width=3000.)
    monkeypatch.setattr(metals, "_rt", None)
    python_reference = metals.manifold_quasistatic_line_profile(*args, **kwargs)
    calls = {"impact": 0, "finish": []}

    def impact(*values):
        calls["impact"] += 1
        return native_frequency.frequency_voigt_profile(*values)

    def finish(*values):
        calls["finish"].append(tuple(value.copy() if isinstance(value, np.ndarray)
                                      else value for value in values[:-1]))
        return native_frequency.stark_frequency_profile_finish(*values)

    def obsolete_finish(*unused):
        raise AssertionError("the old pseudo-Voigt finish must not be selected")

    available = dict(stark_manifold_bins=native_frequency.stark_manifold_bins,
                     stark_profile_finish=obsolete_finish)
    if use_impact:
        available["frequency_voigt_profile"] = impact
    if use_finish:
        available["stark_frequency_profile_finish"] = finish
    monkeypatch.setattr(metals, "_rt", SimpleNamespace(**available))
    actual = metals.manifold_quasistatic_line_profile(*args, **kwargs)
    np.testing.assert_allclose(actual, python_reference, rtol=2.e-7,
                               atol=2.e-10*np.max(python_reference))
    assert calls["impact"] == int(use_impact)
    assert len(calls["finish"]) == int(use_finish)
    if use_finish:
        captured = calls["finish"][0]
        # These are the actual arrays after native deposition and normalized
        # FFT convolution; this checks the new finish without reproducing bins.
        assert len(captured[1]) > 1 and len(captured[2]) > 1
        assert np.all(np.isfinite(captured[1]))
        assert np.all(np.isfinite(captured[2]))
        expected = finish_reference(*captured)
        np.testing.assert_allclose(actual, expected, rtol=1.e-12,
                                   atol=1.e-12*np.max(expected))


@pytest.fixture(scope="module")
def physical_manifolds():
    from wd_spectra.models.common import ModelData
    from wd_spectra.models.d6 import _atomic_inputs
    database = _atomic_inputs(ModelData.default(), ("Mg",))[0]
    ion = database.ions["Mg", 1]
    result = []
    for label in ("2p6.7g.(2G<9/2>)", "2p6.10g.(2G<9/2>)"):
        level = next(level for level in ion.levels if level.label == label)
        result.append(metals.rydberg_stark_manifold(database, ion, level))
    return result


@pytest.mark.parametrize("lower_index,maximum_beta", [(None, .013), (0, 7.13)])
def test_frequency_native_real_manifold_preserves_released_tolerance(
        native_frequency, physical_manifolds, monkeypatch, lower_index, maximum_beta):
    center = 4333.17
    wave = np.unique(np.r_[np.linspace(center-20., center+20., 501),
                           center-np.geomspace(20., 3000., 100),
                           center+np.geomspace(20., 3000., 100)])
    lower = None if lower_index is None else physical_manifolds[lower_index]
    args = (wave, center, .03, .005, 2.e10, physical_manifolds[1], lower, maximum_beta)
    kwargs = dict(correlation=.2, radiator_core_charge=2., support_half_width=3000.)
    monkeypatch.setattr(metals, "_rt", None)
    reference = metals.manifold_quasistatic_line_profile(*args, **kwargs)
    monkeypatch.setattr(metals, "_rt", native_frequency)
    actual = metals.manifold_quasistatic_line_profile(*args, **kwargs)
    np.testing.assert_allclose(actual, reference, rtol=2.e-7,
                               atol=2.e-10*np.max(reference))


def kernel_arguments(name):
    if name == "frequency_voigt_profile":
        return [np.linspace(4999., 5001., 7), 5000., .03, .005, np.empty(7)]
    return [np.linspace(4999., 5001., 7), np.ones(7), np.ones(3),
            5000., .03, .005, .1, 1., .3, .2, .7, .9, np.empty(7)]


@pytest.mark.parametrize("name", ["frequency_voigt_profile", "stark_frequency_profile_finish"])
def test_frequency_kernels_validate_buffers_before_writing(native_frequency, name):
    original = kernel_arguments(name)
    indices = [i for i, value in enumerate(original) if isinstance(value, np.ndarray)]
    function = getattr(native_frequency, name)
    for index in indices:
        for replacement in (original[index].astype(np.float32),
                            original[index].astype(np.dtype(float).newbyteorder("S")),
                            np.ones((len(original[index]), 1)),
                            np.ones(2*len(original[index]))[::2]):
            args = list(original)
            args[index] = replacement
            args[-1][...] = 193.
            with pytest.raises((TypeError, ValueError)):
                function(*args)
            np.testing.assert_array_equal(args[-1], 193.)
    args = list(original)
    args[-1] = np.full(7, 193.)
    args[-1].flags.writeable = False
    with pytest.raises((ValueError, BufferError)):
        function(*args)
    np.testing.assert_array_equal(args[-1], 193.)
    args = list(original)
    args[-1] = np.full(6, 193.)
    with pytest.raises(ValueError):
        function(*args)
    np.testing.assert_array_equal(args[-1], 193.)


@pytest.mark.parametrize("name", ["frequency_voigt_profile", "stark_frequency_profile_finish"])
def test_frequency_kernels_validate_scalars_and_grid_sizes(native_frequency, name):
    original = kernel_arguments(name)
    bad = ([(1, 0.), (1, np.inf), (2, np.nan), (3, np.inf)]
           if name == "frequency_voigt_profile" else
           [(1, np.empty(0)), (1, np.ones(6)), (2, np.ones(4)),
            (3, 0.), (3, np.inf), (4, np.nan), (5, np.inf), (6, 0.),
            (7, np.inf), (8, -.1), (9, np.nan), (10, np.inf), (11, 0.)])
    for index, replacement in bad:
        args = list(original)
        args[index] = replacement
        args[-1][...] = 193.
        with pytest.raises(ValueError):
            getattr(native_frequency, name)(*args)
        np.testing.assert_array_equal(args[-1], 193.)


@pytest.mark.parametrize("name", ["frequency_voigt_profile", "stark_frequency_profile_finish"])
def test_frequency_kernels_release_buffers_on_success_and_failure(native_frequency, name):
    original = kernel_arguments(name)
    indices = [i for i, value in enumerate(original) if isinstance(value, np.ndarray)]
    for failure in [None, *indices]:
        args = list(original)
        if failure is not None:
            args[failure] = args[failure].astype(np.float32)
        before = [sys.getrefcount(args[i]) for i in indices]
        for _ in range(3):
            if failure is None:
                assert getattr(native_frequency, name)(*args) is None
            else:
                with pytest.raises(TypeError):
                    getattr(native_frequency, name)(*args)
        assert [sys.getrefcount(args[i]) for i in indices] == before
    # Acquiring a read-only input is valid; only the output needs write access.
    for index in indices[:-1]:
        original[index].flags.writeable = False
    getattr(native_frequency, name)(*original)


@pytest.mark.parametrize("name", ["frequency_voigt_profile", "stark_frequency_profile_finish"])
def test_frequency_kernels_reject_overlapping_outputs(native_frequency, name):
    inputs = [0] if name == "frequency_voigt_profile" else [0, 1, 2]
    for index in inputs:
        for shift in (0, 1):
            args = kernel_arguments(name)
            backing = np.linspace(4999., 5001., 8)
            args[index] = backing[:7]
            args[-1] = backing[shift:shift+7]
            before = backing.copy()
            with pytest.raises(ValueError, match="overlap"):
                getattr(native_frequency, name)(*args)
            np.testing.assert_array_equal(backing, before)
    # Disjoint views of one allocation are safe and must not be rejected
    # simply because their .base object is the same.
    args = kernel_arguments(name)
    backing = np.empty(14)
    backing[:7] = args[0]
    args[0], args[-1] = backing[:7], backing[7:]
    getattr(native_frequency, name)(*args)
    assert np.all(np.isfinite(args[-1]))


def compare_legacy_dispatch(native, manifold, monkeypatch, wave, scalars,
                            expected_impact, expected_finish):
    """Compare to the deposition-only extension without changing input types."""
    outcomes = []
    counts = []
    for expose_new in (False, True):
        count = dict(bins=0, impact=0, finish=0)

        def bins(*args):
            count["bins"] += 1
            return native.stark_manifold_bins(*args)

        def impact(*args):
            count["impact"] += 1
            return native.frequency_voigt_profile(*args)

        def finish(*args):
            count["finish"] += 1
            return native.stark_frequency_profile_finish(*args)

        def obsolete_finish(*args):
            raise AssertionError("legacy pseudo-Voigt finish must remain unused")

        available = dict(stark_manifold_bins=bins, stark_profile_finish=obsolete_finish)
        if expose_new:
            available.update(frequency_voigt_profile=impact,
                             stark_frequency_profile_finish=finish)
        monkeypatch.setattr(metals, "_rt", SimpleNamespace(**available))
        try:
            value = metals.manifold_quasistatic_line_profile(
                wave, *scalars, 2.e10, manifold, None, 7.13,
                correlation=.2, radiator_core_charge=2., support_half_width=3000.)
        except (TypeError, ValueError) as error:
            # In particular, the released scalar assembly or extended scalar
            # precision can raise.  Compatibility does not create a new API.
            outcomes.append((type(error), str(error)))
        else:
            outcomes.append(value)
        counts.append(count)
    assert counts[0] == dict(bins=1, impact=0, finish=0)
    assert counts[1] == dict(bins=1, impact=expected_impact, finish=expected_finish)
    reference, actual = outcomes
    if isinstance(reference, tuple):
        assert actual == reference
    else:
        assert type(actual) is type(reference)
        assert np.shape(actual) == np.shape(reference)
        assert np.asarray(actual).dtype == np.asarray(reference).dtype
        np.testing.assert_allclose(actual, reference, rtol=1.e-12,
                                   atol=1.e-12*np.max(np.abs(reference)))


@pytest.mark.parametrize("kind,expected_finish", [
    ("2d", 0), ("float32", 0), ("noncontiguous", 1), ("0d", 0),
])
def test_public_manifold_wave_shape_and_precision_compatibility(
        native_frequency, simple_manifold, monkeypatch, kind, expected_finish):
    wave = np.linspace(4300., 4366., 42)
    if kind == "2d":
        wave = wave.reshape(6, 7)
    elif kind == "float32":
        wave = wave.astype(np.float32)
    elif kind == "noncontiguous":
        wave = wave[::2]
        assert not wave.flags.c_contiguous
    else:
        wave = np.array(4333.17)
    compare_legacy_dispatch(native_frequency, simple_manifold, monkeypatch,
                            wave, (4333.17, .03, .005), 1, expected_finish)


@pytest.mark.parametrize("scalar_type", [np.float32, np.longdouble])
@pytest.mark.parametrize("index", [0, 1, 2], ids=["center", "sigma", "gamma"])
def test_public_manifold_preserves_other_scalar_precision(
        native_frequency, simple_manifold, monkeypatch, scalar_type, index):
    scalars = [4333.17, .03, .005]
    scalars[index] = scalar_type(scalars[index])
    compare_legacy_dispatch(native_frequency, simple_manifold, monkeypatch,
                            np.linspace(4300., 4366., 43), scalars, 0, 0)


@pytest.mark.parametrize("scalar_type", [float, int, np.float64])
def test_public_manifold_admitted_scalars_keep_both_native_methods(
        native_frequency, simple_manifold, monkeypatch, scalar_type):
    scalars = ((4333, 1, 1) if scalar_type is int else (4333.17, .03, .005))
    compare_legacy_dispatch(native_frequency, simple_manifold, monkeypatch,
                            np.linspace(4300., 4366., 43),
                            tuple(scalar_type(value) for value in scalars), 1, 1)


@pytest.mark.parametrize("fine_size,coarse_size", [(7, 7), (1, 0)])
def test_frequency_finish_inclusive_binary_boundary_and_single_bin(
        native_frequency, fine_size, coarse_size):
    center, half = 5000., .5
    edges = center + np.array([-half, half])
    wave = np.r_[edges, np.nextafter(edges, -np.inf), np.nextafter(edges, np.inf)]
    fine = np.full(fine_size, .4)
    coarse = np.full(coarse_size, .002)
    args = (wave, fine, coarse, center, .03, .005, .125, 2., half, .2, .7, .9)
    output = np.empty_like(wave)
    native_frequency.stark_frequency_profile_finish(*args, output)
    reference = finish_reference(*args)
    np.testing.assert_allclose(output, reference, rtol=1.e-12, atol=1.e-12)
    # At both exactly representable edges, use the inner fine distribution;
    # the neighbouring exterior sample has a deliberately different weight.
    assert output[0] > 1.
    assert output[1] > 1.
    assert output[2] < .01  # nextafter(lower edge, -inf) is exterior
    assert output[5] < .01  # nextafter(upper edge, +inf) is exterior
