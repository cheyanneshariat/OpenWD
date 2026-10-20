#!/usr/bin/env python3
"""He- free-free absorption from e-He continuum dipole matrix elements.

Physics (atomic units, hbar = m_e = e = 1):
  * e-He interaction: l-dependent pseudopotential fitted to Nesbet (1979)
    phase shifts (l = 0, 1, 2; k = 0.1-1.1 a.u.) and scattering length 1.1835:
      V1(r) = B1 exp(-b1 r^2) - alpha_He / (2 (r^2 + d^2)^2)    (l >= 1)
      V0(r) = V1(r) + C exp(-g r^2)                              (l = 0)
  * radial continuum waves u_l(k r) ~ sin(k r - l pi/2 + delta_l), Numerov.
  * emission rate per unit omega: (4 omega^3 / 3 c^3) |<a|r|b>|^2; with
    momentum-normalized partial waves, the direction-averaged squared dipole is
      int dOmega_a <|<k_a|r|k_b>|^2> = S / (pi^3 k_a^2 k_b^2),
      S = sum over (l_a, l_b = l_a +- 1) of max(l_a, l_b) |I_{l_a l_b}|^2,
      I = int_0^inf u_a u_b r dr.
    s<->p integrals use the length form with an Abel-regularized asymptotic
    tail; pairs with both l >= 1 (local potential) use the acceleration form
    I = int u_a V1'(r) u_b dr / omega^2, which converges absolutely.
  * Kirchhoff with the Planck function gives the net (stimulated-corrected)
    absorption coefficient
      alpha = n_e n_He (32 pi^2/3)(omega/c)(1 - e^{-omega/T}) (2 pi T)^{-3/2} 4 pi
              int_0^inf e^{-E_a/T} S(k_a, k_b) / (k_a k_b) dE_a,
    and k_ff = alpha / (n_He P_e) in cm^4 dyn^-1 (John's convention).
No stellar data enter; validation is against John (1994) at lambda >= 1 um.
"""
import numpy as np

ALPHA_HE = 1.383191
PARAMS = np.array([-0.9656058551198032, 0.8639588873764198, 1.074464834297183,
                   2.6209083478972657, 0.49850740772119595])
C_AU = 137.035999084
A0_CM = 5.29177210903e-9
EH_ERG = 4.3597447222071e-11
EH_EV = 27.211386245988
K_B_AU = 1.0 / 315775.02480407  # Hartree per kelvin


def v1(r, p=None):
    p = PARAMS if p is None else p
    B1, b1, d, C, g = p
    return B1 * np.exp(-b1 * r**2) - ALPHA_HE / (2.0 * (r**2 + d**2) ** 2)


def dv1(r, p=None):
    p = PARAMS if p is None else p
    B1, b1, d, C, g = p
    return -2.0 * b1 * r * B1 * np.exp(-b1 * r**2) + 2.0 * ALPHA_HE * r / (r**2 + d**2) ** 3


def v0(r, p=None):
    p = PARAMS if p is None else p
    return v1(r, p) + p[3] * np.exp(-p[4] * r**2)


def riccati(l, x):
    from scipy.special import spherical_jn, spherical_yn
    return x * spherical_jn(l, x), x * spherical_yn(l, x)


def solve_waves(k, lmax, h=0.01, rmax=400.0):
    """Normalized radial waves u[l, ik, r] ~ sin(kr - l pi/2 + delta) and delta[l, ik]."""
    k = np.atleast_1d(np.asarray(k, float))
    r = np.arange(1, int(rmax / h) + 1) * h
    nk = k.size
    U = np.empty((lmax + 1, nk, r.size))
    D = np.empty((lmax + 1, nk))
    for l in range(lmax + 1):
        V = v0(r) if l == 0 else v1(r)
        # Numerov for u'' = F u, F = l(l+1)/r^2 + 2V - k^2
        F = (l * (l + 1) / r**2 + 2.0 * V)[None, :] - (k**2)[:, None]
        u = np.zeros((nk, r.size))
        u[:, 0] = h ** (l + 1)
        u[:, 1] = (2 * h) ** (l + 1)
        w = 1.0 - h * h * F / 12.0
        for i in range(1, r.size - 1):
            u[:, i + 1] = ((12.0 - 10.0 * w[:, i]) * u[:, i] - w[:, i - 1] * u[:, i - 1]) / w[:, i + 1]
            big = np.abs(u[:, i + 1]) > 1e200
            if big.any():
                u[big, : i + 2] *= 1e-200
        # match on two outer points to A (jhat cos d - nhat sin d)
        i1, i2 = r.size - 1, r.size - 51
        j1, n1 = riccati(l, k * r[i1])
        j2, n2 = riccati(l, k * r[i2])
        # u = A cos d jhat - A sin d nhat  -> linear solve for (A cos d, A sin d)
        det = -j1 * n2 + j2 * n1
        a_c = (-u[:, i1] * n2 + u[:, i2] * n1) / det
        a_s = (j1 * u[:, i2] - j2 * u[:, i1]) / det
        A = np.hypot(a_c, a_s)
        D[l] = np.arctan2(a_s, a_c)
        U[l] = u / A[:, None]
    return r, U, D


def _tail_length(l_a, l_b, ka, kb, da, db, R):
    """Abel-regularized int_R^inf r u_a u_b dr from leading asymptotics
    u ~ sin(th) + (l(l+1)/2x) cos(th), th = x - l pi/2 + delta (first order)."""
    ta = ka * R - l_a * np.pi / 2 + da
    tb = kb * R - l_b * np.pi / 2 + db
    qa, qb = l_a * (l_a + 1) / 2.0 / ka, l_b * (l_b + 1) / 2.0 / kb

    def int_r_cos(q, ph):  # int_R^inf r cos(q r' + ph) dr', r' = r - R, Abel
        return -R * np.sin(ph) / q - np.cos(ph) / q**2

    def int_cos(q, ph):
        return -np.sin(ph) / q

    def int_r_sin(q, ph):
        return R * np.cos(ph) / q - np.sin(ph) / q**2

    def int_sin(q, ph):
        return np.cos(ph) / q

    m, p = kb - ka, kb + ka
    # r sin ta sin tb = r/2 [cos(tb - ta) - cos(tb + ta)]
    out = 0.5 * (int_r_cos(m, tb - ta) - int_r_cos(p, tb + ta))
    # qb sin ta cos tb = qb/2 [sin(ta + tb) + sin(ta - tb)]  -> sin(tb+ta) - sin(tb - ta)
    out += qb * 0.5 * (int_sin(p, tb + ta) - int_sin(m, tb - ta))
    # qa cos ta sin tb = qa/2 [sin(tb + ta) + sin(tb - ta)]
    out += qa * 0.5 * (int_sin(p, tb + ta) + int_sin(m, tb - ta))
    return out


def _length_average(ua, ub, r, h, la, lb, ka, kb, da, db):
    """Length form averaged over cut radii R in [r_max/2, r_max] (diagnostic only)."""
    cum = np.cumsum(ua * ub * r * h, axis=1)
    idx = np.arange(r.size // 2, r.size, 25)
    vals = [cum[:, i] + _tail_length(la, lb, ka, kb, da, db, r[i]) for i in idx]
    return np.mean(vals, axis=0)


def dipole_sum(ka, kb, lmax=24, h=0.01, rmax=400.0, check=False):
    """S(k_a, k_b) for arrays of equal length (pairs)."""
    ka = np.asarray(ka, float)
    kb = np.asarray(kb, float)
    r, Ua, Da = solve_waves(ka, lmax, h, rmax)
    _, Ub, Db = solve_waves(kb, lmax, h, rmax)
    omega = 0.5 * (kb**2 - ka**2)
    w = np.full(r.size, h)
    w[-1] = 0.5 * h
    dv = dv1(r)
    S = np.zeros(ka.size)
    diag = {}
    for la in range(lmax + 1):
        for lb in (la - 1, la + 1):
            if lb < 0 or lb > lmax:
                continue
            if la == 0 or lb == 0:
                # s-p pair with the s-only short-range term W: absolutely
                # convergent form (derived by applying the commutator twice):
                # w'^2 I = <p|V1'|s> - <p|D(W s)> - w' <p|r W|s>,
                # D = 1/r - d/dr, w' = E_p - E_s.
                if la == 0:
                    us, up, ks, kp = Ua[0], Ub[1], ka, kb
                else:
                    us, up, ks, kp = Ub[0], Ua[1], kb, ka
                wp = 0.5 * (kp**2 - ks**2)
                W = PARAMS[3] * np.exp(-PARAMS[4] * r**2)
                dW = -2.0 * PARAMS[4] * r * W
                dus = np.gradient(us, h, axis=1)
                DWu = W * us / r - dW * us - W * dus
                I = (np.sum(up * dv * us * w, axis=1) - np.sum(up * DWu * w, axis=1)) / wp**2 \
                    - np.sum(up * r * W * us * w, axis=1) / wp
                if check:
                    diag[(la, lb)] = (I, _length_average(Ua[la], Ub[lb], r, h, la, lb, ka, kb, Da[la], Db[lb]))
            else:
                I = np.sum(Ua[la] * Ub[lb] * (dv * w), axis=1) / omega**2
                if check and (la, lb) in ((1, 2), (2, 1)):
                    diag[(la, lb)] = (I, _length_average(Ua[la], Ub[lb], r, h, la, lb, ka, kb, Da[la], Db[lb]))
            S += max(la, lb) * I**2
    return (S, diag) if check else S


def k_ff(wavelength_angstrom, temperature, n_energy=48, lmax=24, h=0.01, rmax=400.0):
    """He- free-free coefficient (cm^4 dyn^-1, stimulated emission included)."""
    from numpy.polynomial.laguerre import laggauss
    T = temperature * K_B_AU
    omega = 6.62607015e-27 * 2.99792458e10 / (wavelength_angstrom * 1e-8) / EH_ERG
    x, wts = laggauss(n_energy)
    Ea = x * T
    ka = np.sqrt(2.0 * Ea)
    kb = np.sqrt(2.0 * (Ea + omega))
    S = dipole_sum(ka, kb, lmax, h, rmax)
    integral = T * np.sum(wts * S / (ka * kb))
    alpha_per = (32.0 * np.pi**2 / 3.0) * (omega / C_AU) * (-np.expm1(-omega / T)) \
        * (2.0 * np.pi * T) ** -1.5 * 4.0 * np.pi * integral       # alpha / (n_e n_He), a.u.
    k_au = alpha_per / T                                           # / (n_He n_e T)
    return k_au * A0_CM**5 / EH_ERG


def _solve_l(k, l, r, h):
    """Normalized wave u (len(k), len(r)) and phase shift for one l (Numerov)."""
    V = v0(r) if l == 0 else v1(r)
    F = (l * (l + 1) / r**2 + 2.0 * V)[None, :] - (k**2)[:, None]
    w = 1.0 - h * h * F / 12.0
    u = np.zeros((k.size, r.size))
    u[:, 0] = h ** (l + 1)
    u[:, 1] = (2 * h) ** (l + 1)
    for i in range(1, r.size - 1):
        u[:, i + 1] = ((12.0 - 10.0 * w[:, i]) * u[:, i] - w[:, i - 1] * u[:, i - 1]) / w[:, i + 1]
        if i % 200 == 0:
            big = np.abs(u[:, i + 1]) > 1e150
            if big.any():
                u[big, : i + 2] *= 1e-150
    i1, i2 = r.size - 1, r.size - 51
    j1, n1 = riccati(l, k * r[i1])
    j2, n2 = riccati(l, k * r[i2])
    det = -j1 * n2 + j2 * n1
    a_c = (-u[:, i1] * n2 + u[:, i2] * n1) / det
    a_s = (j1 * u[:, i2] - j2 * u[:, i1]) / det
    return u / np.hypot(a_c, a_s)[:, None], np.arctan2(a_s, a_c)


def k_ff_grid(temperature, wavelengths_angstrom, n_energy=48, lmax=24, h=0.02, rmax=200.0):
    """k_ff (cm^4/dyn) at one temperature for many wavelengths, l-by-l (low memory)."""
    from numpy.polynomial.laguerre import laggauss
    T = temperature * K_B_AU
    lam = np.asarray(wavelengths_angstrom, float)
    omega = 6.62607015e-27 * 2.99792458e10 / (lam * 1e-8) / EH_ERG
    x, wts = laggauss(n_energy)
    ka1 = np.sqrt(2.0 * x * T)
    ka = np.tile(ka1, lam.size)
    kb = np.sqrt(2.0 * (np.tile(x * T, lam.size) + np.repeat(omega, n_energy)))
    om = np.repeat(omega, n_energy)
    r = np.arange(1, int(rmax / h) + 1) * h
    wq = np.full(r.size, h)
    wq[-1] = 0.5 * h
    dv = dv1(r)
    W = PARAMS[3] * np.exp(-PARAMS[4] * r**2)
    dW = -2.0 * PARAMS[4] * r * W
    S = np.zeros(ka.size)
    prev = None
    for l in range(lmax + 1):
        ua1, _ = _solve_l(ka1, l, r, h)
        ua = np.tile(ua1, (lam.size, 1))
        ub, _ = _solve_l(kb, l, r, h)
        if prev is not None:
            pa, pb = prev  # waves at l-1
            for (us_or_a, ub_or_b, la, lb) in ((pa, ub, l - 1, l), (ua, pb, l, l - 1)):
                if la == 0 or lb == 0:
                    if la == 0:
                        us, up, ks, kp = us_or_a, ub_or_b, ka, kb
                    else:
                        us, up, ks, kp = ub_or_b, us_or_a, kb, ka
                    wp = 0.5 * (kp**2 - ks**2)
                    dus = np.gradient(us, h, axis=1)
                    DWu = W * us / r - dW * us - W * dus
                    I = (np.sum(up * dv * us * wq, axis=1) - np.sum(up * DWu * wq, axis=1)) / wp**2 \
                        - np.sum(up * r * W * us * wq, axis=1) / wp
                else:
                    I = np.sum(us_or_a * ub_or_b * dv * wq, axis=1) / om**2
                S += max(la, lb) * I**2
        prev = (ua, ub)
    integral = T * np.sum((wts[None, :] * (S / (ka * kb)).reshape(lam.size, n_energy)), axis=1)
    alpha_per = (32.0 * np.pi**2 / 3.0) * (omega / C_AU) * (-np.expm1(-omega / T)) \
        * (2.0 * np.pi * T) ** -1.5 * 4.0 * np.pi * integral
    return alpha_per / T * A0_CM**5 / EH_ERG
