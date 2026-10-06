"""Total (radiative + dielectronic) recombination coefficients from CHIANTI.

CHIANTI ``<ion>.rrparams`` / ``.drparams`` give alpha(T) for recombination of
that ion into the next lower stage, in cm^3 s^-1. For Fe/Ni IV-VII these are
the Shull & van Steenberg (1982) radiative and Mazzotta et al. (1998)
dielectronic fits. They are used only to supply recombination missing from a
truncated explicit atom; they are not fitted to any spectrum.
"""
import hashlib
from pathlib import Path
import numpy as np

ROMAN = {1:'I',2:'II',3:'III',4:'IV',5:'V',6:'VI',7:'VII',8:'VIII'}


def _rows(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        if line.strip().startswith('-1'):
            break
        rows.append(line.split())
    return rows


def radiative(path):
    rows = _rows(path); kind = int(rows[0][0])
    # Types 1/2 (Verner & Ferland / Badnell form) carry an extra column after Z, ion.
    v = [float(x) for x in rows[1][2:]] if kind == 3 else [float(x) for x in rows[1][3:]]
    if kind == 3:
        a, eta = v[:2]
        return lambda t: a * (np.asarray(t) / 1e4) ** -eta
    if kind in (1, 2):
        a, b, t0, t1 = v[:4]
        def alpha(t):
            t = np.asarray(t, dtype=float)
            bb = b + (v[4] * np.exp(-v[5] / t) if kind == 2 else 0.)
            x0, x1 = np.sqrt(t / t0), np.sqrt(t / t1)
            return a / (x0 * (1 + x0) ** (1 - bb) * (1 + x1) ** (1 + bb))
        return alpha
    raise ValueError(f'unsupported CHIANTI radiative-recombination type {kind}: {path}')


def dielectronic(path):
    rows = _rows(path); kind = int(rows[0][0])
    if kind != 1:
        raise ValueError(f'unsupported CHIANTI dielectronic-recombination type {kind}: {path}')
    e = np.array([float(x) for x in rows[1][2:]]); c = np.array([float(x) for x in rows[2][2:]])
    return lambda t: np.asarray(t, dtype=float) ** -1.5 * np.sum(
        c[:, None] * np.exp(-e[:, None] / np.asarray(t, dtype=float)[None, :]), axis=0)


def chianti_total_recombination(directory, element, parent_charges):
    """{parent charge: alpha_RR(T)+alpha_DR(T)} after verifying SHA256SUMS."""
    directory = Path(directory)
    sums = dict(line.split()[::-1] for line in (directory / 'SHA256SUMS').read_text().splitlines())
    result, audit = {}, {}
    for charge in parent_charges:
        stem = f'{element.lower()}_{charge + 1}'
        funcs = []
        for kind, reader in (('rrparams', radiative), ('drparams', dielectronic)):
            path = directory / f'{stem}.{kind}'
            if hashlib.sha256(path.read_bytes()).hexdigest() != sums[path.name]:
                raise ValueError(f'CHIANTI recombination checksum mismatch: {path}')
            funcs.append(reader(path))
            audit[path.name] = sums[path.name]
        result[charge] = (lambda rr, dr: (lambda t: rr(t) + dr(t)))(*funcs)
    return result, audit
