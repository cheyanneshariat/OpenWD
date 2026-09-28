#!/usr/bin/env python3
"""Build the bundled H2db subset used by the DAH model.

Input is the extracted public H2db archive of Schimeczek & Wunner (DaRUS,
doi:10.18419/DARUS-2118, CC BY 4.0): ``transitions.7z`` and ``energies.7z``
unpacked into ``ROOT/transitions`` and ``ROOT/energies``.  The output keeps

* every Balmer transition (lower level n=2, upper n=3..22 identified from the
  near-zero-field row), and
* every stationary-state energy track,

for beta = B/B0 <= ``--maximum-beta`` (B0 = 4.70103e9 G), plus the first row
beyond it so linear interpolation reaches the limit.  Values are copied
without modification; labels are the archive's own directory labels.  The
reader in ``wd_spectra.magnetic_atomic`` applies the physical conventions
(sign of m, mirrored Delta m = 0 partners) and documents them.

Usage::

    python tools/build_h2db_subset.py /path/to/h2db \
        src/wd_spectra/data/runtime/cache/h2db/h2db_balmer_subset.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import numpy as np

_TRANSITION = re.compile(
    r"m_([+-]?\d+)_to_([+-]?\d+)/pi_z_([+-]1)_to_([+-]1)/nu_(\d+)_to_(\d+)\.tsv$"
)
_ENERGY = re.compile(r"m_(-?\d+)/pi_z_([+-]1)/nu_(\d+)\.tsv$")


def _table(path: Path) -> np.ndarray:
    table = np.genfromtxt(
        path, comments="#", dtype=np.float64, missing_values="n/a",
        filling_values=np.nan,
    )
    return table[np.newaxis, :] if table.ndim == 1 else table


def _truncate(table: np.ndarray, maximum_beta: float) -> np.ndarray:
    beyond = np.flatnonzero(table[:, 0] > maximum_beta)
    return table if beyond.size == 0 else table[: beyond[0] + 1]


def _balmer_upper_level(transition_energy: float, initial_energy: float) -> int | None:
    if abs(initial_energy + 0.25) > 3.0e-3:
        return None
    levels = np.arange(3, 23)
    expected = 0.25 - 1.0 / levels.astype(float) ** 2
    index = int(np.argmin(abs(expected - transition_energy)))
    if abs(expected[index] - transition_energy) >= 2.5e-3:
        return None
    return int(levels[index])


def build(root: Path, output: Path, maximum_beta: float) -> None:
    labels, offsets, columns = [], [0], []
    for path in sorted((root / "transitions").rglob("*.tsv")):
        match = _TRANSITION.search(path.as_posix())
        if match is None:
            continue
        table = _table(path)
        finite = np.isfinite(table[:, 0]) & np.isfinite(table[:, 1]) & np.isfinite(table[:, 3])
        table = table[finite]
        if table.shape[0] < 2:
            continue
        upper = _balmer_upper_level(table[0, 1], table[0, 3])
        if upper is None:
            continue
        table = _truncate(table, maximum_beta)
        labels.append([int(value) for value in match.groups()] + [upper])
        columns.append(table[:, :4])
        offsets.append(offsets[-1] + table.shape[0])
    transitions = np.concatenate(columns)

    energy_labels, energy_offsets, energy_columns = [], [0], []
    for path in sorted((root / "energies").rglob("*.tsv")):
        match = _ENERGY.search(path.as_posix())
        if match is None:
            continue
        table = _table(path)
        table = table[np.isfinite(table[:, 0]) & np.isfinite(table[:, 1])]
        table = _truncate(table, maximum_beta)
        energy_labels.append([int(value) for value in match.groups()])
        energy_columns.append(table[:, :2])
        energy_offsets.append(energy_offsets[-1] + table.shape[0])
    energies = np.concatenate(energy_columns)

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        transition_labels=np.asarray(labels, dtype=np.int16),
        transition_offsets=np.asarray(offsets, dtype=np.int64),
        transition_beta=transitions[:, 0],
        transition_energy_rydberg=transitions[:, 1].astype(np.float32),
        transition_dipole_strength=transitions[:, 2].astype(np.float32),
        transition_initial_energy_rydberg=transitions[:, 3].astype(np.float32),
        energy_labels=np.asarray(energy_labels, dtype=np.int16),
        energy_offsets=np.asarray(energy_offsets, dtype=np.int64),
        energy_beta=energies[:, 0],
        energy_rydberg=energies[:, 1],
        maximum_beta=np.float64(maximum_beta),
        reference_field_gauss=np.float64(4.70103e9),
        source=np.str_(
            "Schimeczek & Wunner H2db, DaRUS doi:10.18419/DARUS-2118, CC BY 4.0"
        ),
    )
    print(
        f"{len(labels)} Balmer transitions ({transitions.shape[0]} rows), "
        f"{len(energy_labels)} energy tracks ({energies.shape[0]} rows) -> {output} "
        f"({output.stat().st_size / 1e6:.1f} MB)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--maximum-beta", type=float, default=3.0)
    args = parser.parse_args()
    build(args.root, args.output, args.maximum_beta)


if __name__ == "__main__":
    main()
