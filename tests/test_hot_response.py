"""Compare cached hot responses with the original transfer solves."""
import numpy as np
import pytest
from wd_spectra._mass_feautrier import MassResponseOperator, mass_emissivity_field
from wd_spectra.opacity import optical_depth_from_mass_opacity
from wd_spectra._hot_response import HotResponseOperator


@pytest.mark.parametrize('angles', [1, 3])
@pytest.mark.parametrize('scattering', [0., .9, .999999, 1.01])
def test_basis_matches_distinct_full_material_responses(angles, scattering):
    mass = np.geomspace(1e-10, 30., 24)
    wave = np.array([500., 1000., 4000., 10000., 20000.])
    ext = np.array([.3, .5, 1., 2., 3.])[:, None]*(.5+np.exp(np.sin(np.arange(24))))
    tau = optical_depth_from_mass_opacity(mass, ext)
    fraction = np.full_like(ext, scattering)
    fraction[:, -1] = min(scattering, .99)
    # For gain, keep it in a thin surface cell and use positive emission.
    if scattering > 1:
        fraction[:, 1:] = .5
    absorption = ext*(1-fraction)
    emission = ext*(1+mass*.2)[None, :]
    source, _ = mass_emissivity_field(tau, emission, absorption, ext*fraction,
        column_mass=mass, bottom_source=emission[:, -1]/absorption[:, -1], n_angle=angles)
    options = dict(extinction=ext, n_angle=angles, wavelength_chunk_size=2,
                   allow_stimulated_gain=scattering > 1, reconstruct_intensity=True)
    expected_operator = MassResponseOperator(tau, wave, source, fraction, mass, **options)
    candidate = HotResponseOperator(tau, wave, source, fraction, mass, **options)
    rng = np.random.default_rng(327)
    for _ in range(3):
        direct, db, relative_ext = rng.normal(size=(3,)+ext.shape)
        direct *= source
        db *= source
        dk = relative_ext*ext
        expected = expected_operator.apply(direct, db, dk)
        actual = candidate.apply(direct, db, dk)
        for a, b in zip(actual, expected):
            np.testing.assert_allclose(a, b, rtol=2e-11, atol=2e-11*np.max(abs(b)))
        streamed = np.empty_like(actual[1])
        def consume(start, stop, value):
            assert not value.flags.writeable
            streamed[start:stop] = value
        flux, mean, source_response = candidate.apply(direct, db, dk,
            return_auxiliary_response=False, mean_response_consumer=consume)
        np.testing.assert_array_equal(flux, actual[0])
        np.testing.assert_array_equal(streamed, actual[1])
        assert mean is None and source_response is None
    assert len(candidate.bases) == 3
    assert all(not a.flags.writeable for basis in candidate.bases.values() for a in basis)


@pytest.mark.parametrize('invalid', ['wave', 'mass', 'source', 'fraction', 'extinction', 'shape', 'chunk', 'direction'])
@pytest.mark.parametrize('operator', [MassResponseOperator, HotResponseOperator])
def test_response_reuse_preserves_invalid_input_rejection(invalid, operator):
    values = dict(tau=np.ones((3, 4)), wave=np.array([500., 1000., 2000.]),
                  source=np.ones((3, 4)), fraction=np.full((3, 4), .5),
                  mass=np.geomspace(.01, 1., 4), extinction=np.ones((3, 4)),
                  wavelength_chunk_size=2)
    if invalid == 'wave':
        values['wave'][1] = values['wave'][0]
    elif invalid == 'mass':
        values['mass'][0] = 0.
    elif invalid == 'source':
        values['source'][0, 0] = np.nan
    elif invalid == 'fraction':
        values['fraction'][0, 0] = 1.1
    elif invalid == 'extinction':
        values['extinction'][0, 0] = 0.
    elif invalid == 'shape':
        values['tau'] = np.ones((3, 3))
    elif invalid == 'chunk':
        values['wavelength_chunk_size'] = 0
    direct = np.ones((3, 4))
    if invalid == 'direction':
        direct[0, 0] = np.inf
    with pytest.raises(ValueError):
        instance = operator(**values)
        instance.apply(direct, np.ones_like(direct), np.ones_like(direct))
