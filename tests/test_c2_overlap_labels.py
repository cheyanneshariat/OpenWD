"""Raw-state branch identities must not confuse spin-distinct levels."""

import bz2
import hashlib

import numpy as np

from wd_spectra import c2_overlap


def write_spin_fixture(tmp_path, monkeypatch):
    # Same approximate Fi/v/parity, but two distinct supplied Sigma values.
    # Each is a continuous P branch with three lower-J values.
    states, transitions = [], []
    for sigma in (0, 1):
        for k in range(3):
            lower = len(states) + 1
            j = 10 + k
            el = 1000 + 20 * k + 100 * sigma
            nu = 20000 + 4 * k + 50 * sigma
            for sid, energy, jj, label in (
                (lower, el, j, "a(3PIu)"),
                (lower + 1, el + nu, j - 1, "d(3PIg)"),
            ):
                states.append(
                    f"{sid} {energy} {2*jj+1} {jj} 0.1 1 0 + e {label} 0 1 {sigma} 2 F1 {jj-1} d {energy}\n"
                )
            transitions.append(f"{lower+1} {lower} 1 {nu}\n")
    names = {}
    for name, content in (
        ("12C2__8states.states.bz2", "".join(states)),
        ("12C2__8states.trans.bz2", "".join(transitions)),
    ):
        payload = bz2.compress(content.encode())
        (tmp_path / name).write_bytes(payload)
        names[name] = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(c2_overlap, "C2_EXOMOL_SHA256", names)


def test_spin_projection_resolves_distinct_ladders_without_moving_lines(
    tmp_path, monkeypatch
):
    write_spin_fixture(tmp_path, monkeypatch)
    old, old_lo, old_hi, old_counts = c2_overlap.load_branches(tmp_path)
    new, lo, hi, counts = c2_overlap.load_branches(
        tmp_path, resolve_spin_projection=True
    )
    assert old_counts["ambiguous_lines"] == 6
    assert np.all(old_lo == old_hi)
    assert counts == dict(branches=2, singleton_lines=0, ambiguous_lines=0)
    assert new.shape == (13, 6)
    assert np.all(hi > lo)
    # Input frequencies, line strengths, energies and the old labels unchanged.
    np.testing.assert_array_equal(
        old[:, np.argsort(old[0])], new[:11, np.argsort(new[0])]
    )
    np.testing.assert_allclose(hi - lo, 2.0)
    strengths = new[1]
    edges = np.arange(19900.0, 20201)
    bins = c2_overlap.interval_deposit(
        edges, lo.ravel(), hi.ravel(), np.repeat(strengths / 2, 2)
    )
    assert np.isclose(bins.sum(), strengths.sum(), rtol=1e-14)


def test_existing_builder_still_uses_legacy_grouping_by_default(tmp_path, monkeypatch):
    write_spin_fixture(tmp_path, monkeypatch)
    implicit = c2_overlap.load_branches(tmp_path)
    explicit = c2_overlap.load_branches(tmp_path, resolve_spin_projection=False)
    for a, b in zip(implicit[:3], explicit[:3]):
        np.testing.assert_array_equal(a, b)
    assert implicit[3] == explicit[3]
