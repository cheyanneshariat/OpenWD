"""Physical polarity of an isolated, longitudinal H-beta normal triplet."""
from dataclasses import replace

import numpy as np
import pytest

from wd_spectra.atmosphere import gray_hydrogen_atmosphere
from wd_spectra.constants import ELEMENTARY_CHARGE_ESU, ELECTRON_MASS, LIGHT_SPEED, PI
from wd_spectra.magnetic import (
    BalmerLineTemplate, _observer_polarization, weak_field_balmer_manifolds,
)
from wd_spectra.radiative_transfer import emergent_stokes_specific_intensity


@pytest.mark.parametrize("polarity", [1., -1.], ids=["toward", "away"])
def test_hbeta_normal_triplet_optical_v_polarity(polarity):
    # Milne-Eddington benchmark: a depth-independent Gaussian absorption
    # profile and a source S=a+b*m increasing inward. This is deliberately
    # not a self-consistent atmosphere or a claim about every Stark sublobe.
    atmosphere = gray_hydrogen_atmosphere(15000., 8., n_depth=8)
    atmosphere = replace(atmosphere, temperature=np.full(8, 15000.))
    center = 4862.694  # vacuum H-beta
    nu0 = LIGHT_SPEED / (center * 1e-8)
    shift = ELEMENTARY_CHARGE_ESU * .5e6 / (4 * PI * ELECTRON_MASS * LIGHT_SPEED)
    wave = LIGHT_SPEED / np.array([nu0 + shift, nu0, nu0 - shift]) / 1e-8
    mesh = center + np.linspace(-30., 30., 30001)
    profile = 3. * np.exp(-.5 * ((mesh - center) / .8)**2)
    template = BalmerLineTemplate(mesh, np.broadcast_to(profile[:, None], (mesh.size, 8)))
    manifolds = weak_field_balmer_manifolds(atmosphere, wave, .5, templates={4: template})
    # Pin the actual physical frequency shift, not just the conversion of
    # arbitrary signed coefficients: Delta m=+1 is the blue manifold.
    assert manifolds.plus[0, 0] > 1000 * manifolds.minus[0, 0]
    assert manifolds.minus[-1, 0] > 1000 * manifolds.plus[-1, 0]
    line_i, eta_q, eta_v = manifolds.stokes(polarity)
    eta_i = .2 + line_i
    zero = np.zeros_like(eta_i)
    mass = np.linspace(0., 20., 8)
    mu, a, b = .63, 1., .2
    source = np.broadcast_to(a + b * mass, eta_i.shape)
    emission = np.stack([eta_i * source, zero, zero, eta_v * source], axis=-1)
    native = emergent_stokes_specific_intensity(
        mass, eta_i, eta_q, eta_v, zero, zero, source, mu,
        emission_stokes=emission, formal_solver="delo-linear")
    _, _, optical_v = _observer_polarization(native.q, native.u, native.v, 0.)
    # Independent scalar circular-mode solution with a thermal bottom.
    def mode(kappa):
        return a + mu * b / kappa * (-np.expm1(-kappa * mass[-1] / mu))
    plus, minus = eta_i[:, 0] + eta_v[:, 0], eta_i[:, 0] - eta_v[:, 0]
    expected_i = .5 * (mode(plus) + mode(minus))
    expected_v = .5 * (mode(minus) - mode(plus))
    np.testing.assert_allclose(native.i, expected_i, rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(optical_v, expected_v, rtol=1e-11, atol=1e-12)
    assert polarity * optical_v[0] > 0.  # blue sigma: optical V positive toward us
    assert polarity * optical_v[-1] < 0. # red sigma: opposite sign
    np.testing.assert_array_equal(optical_v, -native.v)
