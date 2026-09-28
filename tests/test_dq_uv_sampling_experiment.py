import numpy as np
import pytest

from wd_spectra._dq.dq_uv_sampling_experiment import augment_uv, refined_material, thinned_grid


def test_original_nodes_and_non_uv_are_preserved():
    original = np.array([1000., 1250., 2000., 3799., 4000., 100000.])
    refined = augment_uv(original, 4000)
    assert np.all(np.isin(original, refined))
    np.testing.assert_array_equal(refined[refined >= 3800.], original[original >= 3800.])
    assert np.all(np.diff(refined) > 0)
    np.testing.assert_array_equal(augment_uv(refined, 4000), refined)


def test_no_extension_outside_input_domain():
    refined = augment_uv([2000., 3000.], 4000)
    assert refined[0] == 2000. and refined[-1] == 3000.


def test_full_continuum_preserves_nodes_and_refines_infrared():
    original=np.array([1000.,1250.,3800.,5100.,6800.,100000.])
    refined=augment_uv(original,4000,full_continuum=True)
    assert np.all(np.isin(original,refined))
    assert np.all(np.isin(np.geomspace(1000.,100000.,4000),refined))
    np.testing.assert_array_equal(augment_uv(refined,4000,full_continuum=True),refined)
    limited=augment_uv([2000.,9000.],4000,full_continuum=True)
    assert limited[0]==2000. and limited[-1]==9000.


@pytest.mark.parametrize('wave', [[1000., 999.], [1000., np.nan], [0., 1.]])
def test_invalid_grid_rejected(wave):
    with pytest.raises(ValueError):
        augment_uv(wave, 4000)


def test_material_records_the_actual_augmented_grid():
    class Base:
        def __init__(self):
            self.experiment_metadata = {'fixed_wavelength_grid': None}
        def structure_grid(self, seed, count):
            return np.array([1000., 2000., 5000., 100000.])
    material = refined_material(Base, 4000)()
    wave = material.structure_grid(None, 300)
    np.testing.assert_array_equal(wave, material.saved_structure_grid)
    assert material.experiment_metadata['uv_structure_sampling_requires_saved_spectrum'] is False
    assert len(wave) > 4


@pytest.mark.parametrize('stride', [1, 2, 4, 100])
def test_thinning_matches_experiment_exactly(stride):
    original = np.geomspace(1000., 100000., 20)
    expected = np.unique(np.r_[original[::stride], original[-1]])
    np.testing.assert_array_equal(thinned_grid(original, stride), expected)
    if stride == 1:
        np.testing.assert_array_equal(thinned_grid(original, stride), original)


@pytest.mark.parametrize('stride', [0, -1, 1.5, True])
def test_invalid_structure_stride(stride):
    with pytest.raises(ValueError, match='positive integer'):
        thinned_grid([1., 2.], stride)
    with pytest.raises(ValueError, match='positive integer'):
        refined_material(object, 4000, structure_stride=stride)


@pytest.mark.parametrize('wave', [[1.], [1., 1.], [2., 1.], [0., 1.],
                                 [1., np.nan], [1., np.inf], [[1., 2.]]])
def test_invalid_thinning_grid(wave):
    with pytest.raises(ValueError, match='wavelength grid'):
        thinned_grid(wave, 4)


def test_public_material_thins_once_after_refinement_on_each_domain(tmp_path, monkeypatch):
    """Actual runtime class wiring, without substituting for a cold canary."""
    from wd_spectra._dq import runtime
    from wd_spectra._dq.validation import independent_grid

    class Base:
        def __init__(self):
            self.experiment_metadata = {}
        def structure_grid(self, seed, count):
            return seed
        def spectrum(self, atmosphere, wave, n_angle):
            return wave

    monkeypatch.setattr(runtime.dq_explicit_gradient, 'gradient_material', lambda unused, **kwargs: Base)
    material = runtime.material_class(tmp_path)()
    for original in (np.array([1000., 2000., 5000., 100000.]),
                     np.array([1000., 2222., 4444., 5555., 100000.])):
        full = augment_uv(original, 4000, full_continuum=True)
        expected = np.unique(np.r_[full[::4], full[-1]])
        wave = material.structure_grid(original, 300)
        np.testing.assert_array_equal(wave, expected)
        np.testing.assert_array_equal(material.saved_structure_grid, expected)
        assert material.experiment_metadata['structure_sampling_stride'] == 4
        assert material.experiment_metadata['structure_nodes_before_thinning'] == len(full)
        assert material.experiment_metadata['structure_nodes_after_thinning'] == len(expected)
    fine = independent_grid()
    assert len(fine) == 218520
    assert material.spectrum(None, fine, 4) is fine
    assert material.experiment_metadata['final_synthesis_thinned'] is False
