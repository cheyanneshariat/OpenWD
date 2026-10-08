"""He I model atoms for the coupled He I + He II + He III statistical equilibrium.

The coupled solver historically hard-wired TLUSTY's public 14-term atom
(``he1_14lev.dat``: individual n <= 2 terms, singlet/triplet n = 3-5
superlevels, spin-combined n = 6-8).  :class:`HeliumIAtom` carries the term
data so that the same solver can use TLUSTY's 24-term atom (``he1.dat``,
OSTAR2002/BSTAR2006: every LS term through n = 4, then the same n >= 5
superlevels).  The 14-term instance is built in :mod:`helium_nlte` from its
existing constants and remains the default everywhere.

The 24-term data are read at run time from the TLUSTY atom file, as the
collision fits are read from the TLUSTY source.  Its first 19 terms are, in
order and with the same statistical weights, the 19 LS states of TLUSTY's
COLLHE Storey--Hummer/Berrington--Kingston fit, so those collision rates apply
term by term.  Photoionization uses the same TLUSTY/Opacity Project fits as
the 14-term atom, resolved by term: the Seaton--Fernley fits for n <= 2, the
l-resolved HEPHOT fits for n = 3, 4 and l <= 2, and the hydrogenic Kramers
cross section 2.815e29 / (nu^3 n^5) for 4F and every superlevel.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path
import re
from typing import Callable, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .helium import (
    _HE_I_RYDBERG_OP_LOOKUP,
    neutral_helium_photoionization_cross_section,
)

FloatArray = NDArray[np.float64]

_ORBITAL = {"S": 0, "P": 1, "D": 2, "F": 3, "G": 4}
_MULTIPLICITY = {"sing": 1, "trip": 3}
_FINE_STATE_COUNT = 19


@dataclass(frozen=True, eq=False)
class HeliumIAtom:
    """Term data of a He I model atom (indices are 0-based; ``oscillator_strength`` keys 1-based)."""

    name: str
    threshold_frequency_hz: FloatArray
    statistical_weight: FloatArray
    principal_quantum_number: FloatArray
    multiplicity: NDArray[np.int64]           # 1, 3, or 0 for a spin-combined superlevel
    orbital_angular_momentum: NDArray[np.int64]  # L, or -1 for an l-combined superlevel
    label: tuple[str, ...]
    oscillator_strength: Mapping[tuple[int, int], float]
    collision_fine_groups: tuple[tuple[int, ...], ...]
    photoionization: Callable[[int, ArrayLike], FloatArray]
    source: str

    @property
    def n_terms(self) -> int:
        return int(self.threshold_frequency_hz.size)

    @property
    def n_fitted_collision_terms(self) -> int:
        """Leading terms whose excitation rates come from the 19-state COLLHE fit."""
        return len(self.collision_fine_groups)

    def photoionization_cross_section(self, term_index: int, frequency_hz: ArrayLike) -> FloatArray:
        return self.photoionization(term_index, frequency_hz)

    def term_index(self, principal: int, multiplicity: int, orbital: int) -> int:
        """Term holding (n, 2S+1, L), falling back to its superlevel and finally the top shell."""
        n_max = int(np.max(self.principal_quantum_number))
        for n, s, l in ((principal, multiplicity, orbital), (principal, multiplicity, -1),
                        (principal, 0, -1), (min(principal, n_max), 0, -1)):
            match = np.flatnonzero((self.principal_quantum_number == n) & (self.multiplicity == s)
                                   & (self.orbital_angular_momentum == l))
            if match.size:
                return int(match[0])
        raise KeyError(f"no He I term for n={principal}, 2S+1={multiplicity}, L={orbital}")


def _resolved_term_cross_section(term_index, frequency_hz, *, threshold, principal, multiplicity, orbital):
    frequency = np.asarray(frequency_hz, dtype=np.float64)
    if np.any(~np.isfinite(frequency)) or np.any(frequency <= 0.0):
        raise ValueError("frequency must be finite and positive")
    n, s, l = int(principal[term_index]), int(multiplicity[term_index]), int(orbital[term_index])
    if n <= 2:
        # Terms 0-4 are the Seaton--Fernley fits in the EOS order 1 1S, 2 3S, 2 1S, 2 3P, 2 1P.
        return neutral_helium_photoionization_cross_section(frequency, term_index)
    fit = _HE_I_RYDBERG_OP_LOOKUP.get((n, s, l)) if l >= 0 else None
    if fit is not None:
        x = np.log10(frequency / 3.28805e15) - fit[0]
        polynomial = fit[4] + x * (fit[5] + x * (fit[6] + x * fit[7]))
        log_cross_section_mb = np.where(x < fit[1], polynomial, fit[2] + fit[3] * x)
        cross_section = np.power(10.0, log_cross_section_mb) * 1.0e-18
    else:
        cross_section = 2.815e29 / (frequency**3 * n**5)
    return np.where(frequency >= threshold[term_index], cross_section, 0.0)


def _parse_level_label(label: str) -> tuple[int, int]:
    text = label.strip().strip("<>").strip()
    words = text.split()
    multiplicity = next((_MULTIPLICITY[w] for w in words if w in _MULTIPLICITY), 0)
    orbital = _ORBITAL.get(words[-1], -1) if label.strip()[0] != "<" else -1
    return multiplicity, orbital


def _numbers(line: str) -> list[float]:
    return [float(v.replace("D", "E").replace("d", "e")) for v in
            re.findall(r"[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[dDeE][+-]?\d+)?", line)]


def read_tlusty_helium_i_atom(path: str | Path) -> HeliumIAtom:
    """Read a TLUSTY He I atom file (levels and line oscillator strengths)."""

    path = Path(path)
    text = path.read_text(errors="replace")
    sections = re.split(r"^\*+\s*", text, flags=re.MULTILINE)
    levels = next(s for s in sections if s.lower().startswith("levels"))
    lines = next(s for s in sections if s.lower().startswith("line transitions"))
    frequency, weight, principal, label, multiplicity, orbital = [], [], [], [], [], []
    for row in levels.splitlines()[1:]:
        match = re.match(r"\s*(\S+)\s+(\S+)\s+(\d+)\s+'([^']*)'", row)
        if match is None:
            continue
        frequency.append(float(match.group(1).replace("D", "E")))
        weight.append(float(match.group(2)))
        principal.append(int(match.group(3)))
        label.append(match.group(4).strip())
        s, l = _parse_level_label(match.group(4))
        multiplicity.append(s)
        orbital.append(l)
    n_terms = len(frequency)
    if n_terms < _FINE_STATE_COUNT:
        raise ValueError(f"{path} has {n_terms} levels; the resolved atom needs at least {_FINE_STATE_COUNT}")
    oscillator_strength = {}
    for row in lines.splitlines()[1:]:
        tokens = row.split("!")[0].split()
        if len(tokens) < 9 or not (tokens[0].isdigit() and tokens[1].isdigit()):
            continue
        value = _numbers(" ".join(tokens[7:8]))
        f = value[0] if value else 0.0
        lower, upper = int(tokens[0]), int(tokens[1])
        if not (1 <= lower < upper <= n_terms):
            raise ValueError(f"invalid He I transition {lower}-{upper} in {path}")
        if f > 0.0:
            oscillator_strength[(lower, upper)] = f
    arrays = dict(threshold=np.asarray(frequency, dtype=np.float64),
                  principal=np.asarray(principal, dtype=np.float64),
                  multiplicity=np.asarray(multiplicity, dtype=np.int64),
                  orbital=np.asarray(orbital, dtype=np.int64))
    # The fine COLLHE states are the first 19 LS terms, n <= 4.
    if not (np.all(arrays["principal"][:_FINE_STATE_COUNT] <= 4)
            and np.all(arrays["orbital"][:_FINE_STATE_COUNT] >= 0)):
        raise ValueError("the first 19 He I terms must be the individual n <= 4 LS terms")
    return HeliumIAtom(
        name=f"TLUSTY {n_terms}-term He I ({path.name})",
        threshold_frequency_hz=arrays["threshold"],
        statistical_weight=np.asarray(weight, dtype=np.float64),
        principal_quantum_number=arrays["principal"],
        multiplicity=arrays["multiplicity"],
        orbital_angular_momentum=arrays["orbital"],
        label=tuple(label),
        oscillator_strength=oscillator_strength,
        collision_fine_groups=tuple((index,) for index in range(_FINE_STATE_COUNT)),
        photoionization=partial(_resolved_term_cross_section, **arrays),
        source=str(path.resolve()),
    )
