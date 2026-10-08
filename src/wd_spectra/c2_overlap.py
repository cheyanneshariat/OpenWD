"""Experimental continuous-rotational-branch Swan opacity, not physical widths.

Koester (2010) describes a just-overlapping/smeared-line approximation. This
is an independent line-list-based limiting experiment, NOT a reproduction of
his unpublished implementation: distribute each line's two half strengths
over the adjacent half-J cells within its own (v',v'',Delta J,F',F'',e/f)
branch. Midpoint frequency edges follow the actual raw state energies, even
at a turning point. Every line strength, partition sum and non-Swan system
is retained. No global smoothing width, Teff switch or astronomical fit.

This assumes rotational overlap, which is not established at every density.
It is an approximate band model, not measured C2-He broadening.
"""

import bz2
import hashlib
from itertools import islice
import json
from pathlib import Path
import numpy as np
from .carbon_molecular import C2_EXOMOL_SHA256


def interval_deposit(edges, low, high, strength):
    """Integrate uniform intervals into arbitrary bins; do not renormalize loss.

    Zero-width intervals are delta functions. Complexity is O(N_lines+N_bins),
    including complete intervening bins by a difference array rather than a loop.
    """
    edges, low, high, strength = map(
        lambda a: np.asarray(a, dtype=float), (edges, low, high, strength)
    )
    if (
        edges.ndim != 1
        or len(edges) < 2
        or np.any(np.diff(edges) <= 0)
        or low.ndim != 1
        or high.shape != low.shape
        or strength.shape != low.shape
        or any(np.any(~np.isfinite(a)) for a in (edges, low, high, strength))
        or np.any(high < low)
        or np.any(strength < 0)
    ):
        raise ValueError("Invalid finite-volume interval geometry")
    n = len(edges) - 1
    result = np.zeros(n)
    delta = high == low
    i = np.searchsorted(edges, low[delta], side="right") - 1
    inside = (i >= 0) & (i < n)
    result += np.bincount(i[inside], weights=strength[delta][inside], minlength=n)
    use = (~delta) & (high > edges[0]) & (low < edges[-1])
    density = strength[use] / (high[use] - low[use])
    lo, hi = np.maximum(low[use], edges[0]), np.minimum(high[use], edges[-1])
    first = np.clip(np.searchsorted(edges, lo, side="right") - 1, 0, n - 1)
    last = np.clip(np.searchsorted(edges, hi, side="left") - 1, 0, n - 1)
    same = first == last
    result += np.bincount(
        first[same], weights=density[same] * (hi[same] - lo[same]), minlength=n
    )
    first, last, density, lo, hi = (a[~same] for a in (first, last, density, lo, hi))
    result += np.bincount(first, weights=density * (edges[first + 1] - lo), minlength=n)
    result += np.bincount(last, weights=density * (hi - edges[last]), minlength=n)
    difference = np.bincount(first + 1, weights=density, minlength=n + 1) - np.bincount(
        last, weights=density, minlength=n + 1
    )
    result += np.cumsum(difference[:-1]) * np.diff(edges)
    if result.min() < -1e-10 * max(result.max(), 1e-300):
        raise FloatingPointError("Negative interval deposit beyond roundoff")
    return np.maximum(result, 0)


def branch_half_cells(frequency, angular):
    """Two half-J cells per line; handles reversed bands without sign clipping."""
    nu, j = np.asarray(frequency, dtype=float), np.asarray(angular, dtype=float)
    if (
        len(nu) < 2
        or nu.shape != j.shape
        or np.any(np.diff(j) <= 0)
        or np.any(np.diff(j) > 2)
    ):
        raise ValueError("A unique contiguous rotational branch is required")
    mid = (nu[1:] + nu[:-1]) / 2
    left = np.r_[nu[0] - (nu[1] - nu[0]) / 2, mid]
    right = np.r_[mid, nu[-1] + (nu[-1] - nu[-2]) / 2]
    lo = np.column_stack((np.minimum(left, nu), np.minimum(nu, right)))
    hi = np.column_stack((np.maximum(left, nu), np.maximum(nu, right)))
    return lo, hi


def unique_branch_segments(angular):
    """Isolate ambiguous assignments locally without discarding a whole ladder.

    High-lying perturbed states can have duplicated approximate v/F assignments.
    Do not connect through those ambiguities or invent lines across a J gap.
    """
    j = np.asarray(angular, dtype=float)
    if j.ndim != 1 or np.any(~np.isfinite(j)) or np.any(np.diff(j) < 0):
        raise ValueError("Sorted finite J required")
    same = np.diff(j) == 0
    bad = (
        (np.r_[False, same] | np.r_[same, False])
        if len(j)
        else np.array([], dtype=bool)
    )
    valid = np.flatnonzero(~bad)
    cuts = np.flatnonzero((np.diff(valid) > 1) | (np.diff(j[valid]) > 2)) + 1
    segments = [(int(a[0]), int(a[-1] + 1)) for a in np.split(valid, cuts) if len(a)]
    return segments, int(np.count_nonzero(bad))


def load_branches(directory, *, resolve_spin_projection=False):
    for name, digest in C2_EXOMOL_SHA256.items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Checksum mismatch: {name}")
    with bz2.open(directory / "12C2__8states.states.bz2", "rt") as f:
        raw = np.loadtxt(f, dtype=str, usecols=(0, 1, 2, 3, 8, 9, 10, 14, 12))
    ids = raw[:, 0].astype(int)
    size = ids.max() + 1
    energy, weight, j, v, spin, parity = (np.zeros(size) for _ in range(6))
    label = np.full(size, "", dtype="U12")
    energy[ids], weight[ids], j[ids] = raw[:, 1:4].astype(float).T
    parity[ids] = (raw[:, 4] == "f").astype(int)
    label[ids] = raw[:, 5]
    v[ids] = raw[:, 6].astype(float)
    spin[ids] = np.char.replace(raw[:, 7], "F", "").astype(float)
    sigma = np.zeros(size)
    sigma[ids] = raw[:, 8].astype(float)
    rows = []
    processed = 0
    with bz2.open(directory / "12C2__8states.trans.bz2", "rt") as f:
        while True:
            text = list(islice(f, 150000))
            if not text:
                break
            a = np.fromstring("".join(text), sep=" ").reshape(-1, 4)
            u, l = a[:, :2].astype(int).T
            use = (label[u] == "d(3PIg)") & (label[l] == "a(3PIu)") & (a[:, 2] > 0)
            u, l, A = u[use], l[use], a[use, 2]
            quantum_rows = [
                energy[u] - energy[l],
                A * weight[u],
                energy[l],
                j[l],
                v[u],
                v[l],
                j[u] - j[l],
                spin[u],
                spin[l],
                parity[u],
                parity[l],
            ]
            if resolve_spin_projection:
                # The supplied Fi labels do not uniquely distinguish some
                # perturbed levels. Retain Sigma as well, rather than
                # declaring all their rotational transitions ambiguous.
                quantum_rows.extend((sigma[u], sigma[l]))
            rows.append(np.array(quantum_rows))
            processed += len(a)
            if processed % 1500000 == 0:
                print(f"Read {processed:,} transitions", flush=True)
    a = np.concatenate(rows, axis=1)
    # Lexicographic ordering by branch labels, then increasing lower J.
    order = np.lexsort(tuple(a[i] for i in (3, *range(a.shape[0] - 1, 3, -1))))
    a = a[:, order]
    boundaries = np.r_[
        0, np.flatnonzero(np.any(np.diff(a[4:], axis=1) != 0, axis=0)) + 1, a.shape[1]
    ]
    low = np.repeat(a[0, :, None], 2, axis=1)
    high = low.copy()
    singleton = ambiguous = branches = 0
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        segments, bad = unique_branch_segments(a[3, start:end])
        ambiguous += bad
        for s, e in segments:
            s, e = s + start, e + start
            if e - s == 1:
                singleton += 1
                continue
            low[s:e], high[s:e] = branch_half_cells(a[0, s:e], a[3, s:e])
            branches += 1
    # Unassignable lines stay explicit delta contributions, never disappear.
    print(
        f"{branches} continuous branch segments; {singleton} isolated and {ambiguous} ambiguous lines retained unsmeared",
        flush=True,
    )
    return (
        a,
        low,
        high,
        dict(
            branches=int(branches),
            singleton_lines=int(singleton),
            ambiguous_lines=int(ambiguous),
        ),
    )


def add_rotational_overlap(base, directory, *, resolve_spin_projection=False):
    """Return an artifact containing BOTH the original and overlap Swan data."""
    if "swan_cross_section" not in base:
        raise ValueError("Rotational overlap requires identified Swan transitions")
    provenance = json.loads(str(np.asarray(base["provenance_json"]).item()))
    if provenance["sha256"] != C2_EXOMOL_SHA256:
        raise ValueError("Base line data differ")
    lines, low, high, counts = load_branches(
        directory, resolve_spin_projection=resolve_spin_projection
    )
    nu, ag, el = lines[:3]
    center = 1e8 / base["wavelength_angstrom"][::-1]
    dlog = np.diff(np.log(center))
    if not np.allclose(dlog, dlog[0], rtol=1e-7):
        raise ValueError("Logarithmic bins required")
    edges = np.r_[center * np.exp(-dlog[0] / 2), center[-1] * np.exp(dlog[0] / 2)]
    old = base["swan_cross_section"]
    new = np.empty_like(old)
    checks = []
    for it, t in enumerate(base["temperature_K"]):
        q = np.exp(
            np.interp(
                np.log(t),
                np.log(base["partition_temperature_K"]),
                np.log(base["partition_function"]),
            )
        )
        strength = (
            ag
            * np.exp(-1.438776877 * el / t)
            * -np.expm1(-1.438776877 * nu / t)
            / (8 * np.pi * 2.99792458e10 * nu**2 * q)
        )
        delta = interval_deposit(edges, nu, nu, strength)
        expected = old[::-1, it] * np.diff(edges)
        if not np.allclose(
            delta, expected, rtol=3e-7, atol=max(expected.max(), 1e-300) * 1e-11
        ):
            raise ValueError("Raw-line deposition does not reproduce base Swan table")
        integrated = interval_deposit(
            edges, low.ravel(), high.ravel(), np.repeat(strength / 2, 2)
        )
        new[:, it] = (integrated / np.diff(edges))[::-1]
        unsmeared = np.all(low == high, axis=1)
        optical = (nu >= 1e8 / 6500) & (nu <= 1e8 / 4000)
        checks.append(
            dict(
                temperature=float(t),
                integrated_strength_ratio=float(integrated.sum() / delta.sum()),
                optical_unsmeared_strength_fraction=float(
                    strength[unsmeared & optical].sum() / strength[optical].sum()
                ),
            )
        )
        if it % 10 == 0:
            print(
                f"Built {it+1}/{len(base['temperature_K'])} temperatures; area ratio {checks[-1]['integrated_strength_ratio']:.9f}",
                flush=True,
            )
    result = dict(base)
    result["swan_rotational_overlap_cross_section"] = new
    provenance.update(
        schema=3,
        available_profiles=["line_bin", "rotational_overlap"],
        rotational_overlap=dict(
            algorithm=(
                "continuous half-J cells with explicit Sigma, experimental version 2"
                if resolve_spin_projection
                else "continuous half-J cells, version 1"
            ),
            **counts,
            total_swan_lines=int(lines.shape[1]),
            discarded_lines=0,
            strength_checks=checks,
            motivation="https://articles.adsabs.harvard.edu/pdf/2010MmSAI..81..921K",
            builder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        ),
    )
    result["provenance_json"] = json.dumps(provenance)
    return result
