"""Deterministic continuum refinement around the current-energy DQ driver.

Retain the original UV-only diagnostic option. The released DQ driver builds
the full continuum-refined mesh, then deterministically thins structure nodes.
Opacity physics and convergence gates are unchanged. No past spectrum is needed.
"""

import numpy as np

from .provenance import digest


def augment_uv(wavelength, points, *, full_continuum=False):
    """Retain every input node and add a parameter-independent UV mesh."""
    if points < 2:
        raise ValueError('UV reference mesh requires at least two points')
    wave = np.asarray(wavelength, float)
    if (wave.ndim != 1 or len(wave) < 2 or np.any(~np.isfinite(wave))
            or np.any(wave <= 0) or np.any(np.diff(wave) <= 0)):
        raise ValueError('Expected finite ordered positive wavelength grid')
    reference = np.geomspace(1000., 100000., points)
    extra = reference[(reference >= wave[0]) & ((reference < 3800.) | full_continuum)
                      & (reference <= wave[-1])]
    return np.unique(np.r_[wave, extra])


def thinned_grid(wavelength, stride):
    """Keep every nth node and both endpoints; never alter final synthesis."""
    if type(stride) is not int or stride < 1:
        raise ValueError('positive integer structure stride required')
    wave = np.asarray(wavelength, float)
    if (wave.ndim != 1 or len(wave) < 2 or np.any(~np.isfinite(wave))
            or np.any(wave <= 0) or np.any(np.diff(wave) <= 0)):
        raise ValueError('Expected finite ordered positive wavelength grid')
    return np.unique(np.r_[wave[::stride], wave[-1]])


def refined_material(base, points, *, full_continuum=False, structure_stride=1):
    if type(structure_stride) is not int or structure_stride < 1:
        raise ValueError('positive integer structure stride required')
    class UVRefined(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.experiment_metadata.update(
                uv_structure_sampling_experiment=True,
                uv_structure_sampling_reference_points=points,
                uv_structure_sampling_rule='union existing grid with geomspace(1000,100000,N) below 3800 A',
                uv_structure_sampling_source_sha256=digest(__file__),
                uv_structure_sampling_requires_saved_spectrum=False)
            self.experiment_metadata.update(
                structure_sampling_stride=structure_stride,
                structure_sampling_rule='every nth refined node plus both endpoints',
                final_synthesis_thinned=False,
                full_continuum_sampling=full_continuum,
                continuum_sampling_rule=('union existing grid with geomspace(1000,100000,N)'
                                         if full_continuum else 'UV only'))

        def structure_grid(self, seed, count):
            path = self.experiment_metadata.get('fixed_wavelength_grid')
            if path is not None:
                with np.load(path, allow_pickle=False) as saved:
                    if 'weights' in saved.files:
                        raise ValueError('UV refinement does not support externally weighted grids')
            original = super().structure_grid(seed, count)
            full = augment_uv(original, points, full_continuum=full_continuum)
            wave = thinned_grid(full, structure_stride)
            self.saved_structure_grid = wave
            self.experiment_metadata.update(structure_nodes_before_thinning=len(full),
                                            structure_nodes_after_thinning=len(wave))
            print(f'Continuum structure refinement: {len(original)} -> {len(full)}; '
                  f'stride {structure_stride}: {len(wave)} wavelengths', flush=True)
            return wave
    return UVRefined

