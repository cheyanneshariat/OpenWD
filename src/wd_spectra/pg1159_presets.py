"""Validated parameter and model-atom presets for PG 1159 benchmarks.

These constants describe accepted calculations, rather than exposing the
many one-off sensitivity settings used while developing the PG 1159 solver.
Keeping the target, composition, and rate-atom sizes together prevents a
saved population checkpoint from being restored with a subtly different
atom.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Final, Mapping


PG1424_EFFECTIVE_TEMPERATURE_K: Final = 110_000.0
PG1424_LOGG: Final = 7.0

# Opacity-sampled velocity stencil used only by the expensive He/C/O
# structural transfer.  The final displayed spectrum uses its independent
# fine wavelength grid.  This retains the thermal core and representative
# inner/far-wing samples while avoiding the old 17-point stencil on every
# one of several thousand formal metal components.
PG1424_STRUCTURE_LINE_VELOCITY_SAMPLES_KMS: Final[tuple[float, ...]] = (
    -600.0,
    -150.0,
    -40.0,
    -10.0,
    0.0,
    10.0,
    40.0,
    150.0,
    600.0,
)

PG1424_HECO_MASS_FRACTIONS: Final[Mapping[str, float]] = MappingProxyType(
    {"He": 0.52, "C": 0.45, "O": 0.03}
)

# Werner, Rauch & Kruk (2015), Table 1.  The rounded values are normalized
# once by the model builder because the listed He/C/O fractions already sum
# to unity before the trace species are added.
PG1424_PUBLISHED_MASS_FRACTIONS: Final[Mapping[str, float]] = MappingProxyType(
    {
        "He": 0.52,
        "C": 0.45,
        "O": 0.03,
        "F": 5.0e-5,
        "Ne": 1.0e-2,
        "Si": 2.0e-4,
        "P": 3.2e-5,
        "S": 1.0e-4,
        "Ar": 6.0e-5,
        "Fe": 1.3e-3,
    }
)

# Charge is zero based: charge 2 is C III/O III, charge 5 is O VI, etc.
PG1424_COMPOSITION_LADDER_ATOM_COUNTS: Final[
    Mapping[str, Mapping[int, int]]
] = (
    MappingProxyType(
        {
            "C": MappingProxyType({2: 44, 3: 54, 4: 1}),
            "O": MappingProxyType({2: 1, 3: 1, 4: 28, 5: 4, 6: 1}),
        }
    )
)

# Atom used by the accepted V3 He/C/O structural continuation.  This now
# matches the final C/O line atom: the former 105-entry C III interpretation
# incorrectly included eight RDI auto states, and the former nine-level O VI
# structure did not close the public n=5 cascade.
PG1424_RELAXED_STRUCTURE_ATOM_COUNTS: Final[
    Mapping[str, Mapping[int, int]]
] = MappingProxyType(
    {
        "C": MappingProxyType({2: 97, 3: 54, 4: 1}),
        "O": MappingProxyType({2: 47, 3: 83, 4: 105, 5: 14, 6: 1}),
    }
)

# Backward-compatible name used by the composition-ladder validation script.
PG1424_STRUCTURE_ATOM_COUNTS = PG1424_COMPOSITION_LADDER_ATOM_COUNTS

PG1424_BEST_LINE_ATOM_COUNTS: Final[Mapping[str, Mapping[int, int]]] = (
    MappingProxyType(
        {
            # The public file has 105 apparent C III entries only if its eight
            # RDI autoionizing states with tabulated level energies are
            # incorrectly promoted as ordinary
            # bound levels.  TMAP eliminates those LTE states from the rate
            # equations (Werner et al. 2003, Sect. 3.2), leaving 97 usable
            # explicit public terms versus the historical 133.  All 745 RBB
            # records remain available to the formal line representation.
            "C": MappingProxyType({2: 97, 3: 54, 4: 1}),
            # Fourteen explicit O VI terms are intentional.  The historical
            # nine-term count leaves the public n=5 formal levels attached to
            # an O VII LTE reservoir and creates false FUV emission.
            "O": MappingProxyType({2: 47, 3: 83, 4: 105, 5: 14, 6: 1}),
        }
    )
)

WERNER2015_LINE_ATOM_COUNTS: Final[Mapping[str, Mapping[int, int]]] = (
    MappingProxyType(
        {
            "C": MappingProxyType({2: 133, 3: 54}),
            "O": MappingProxyType({2: 47, 3: 83, 4: 105, 5: 9}),
        }
    )
)

# These species have one NLTE population per ion stage in the accepted local
# model.  Excitation within each trace ion is still LTE.  Phosphorus is
# deliberately absent: the threshold-only P IV--VI and isolated 18-level P V
# experiments did not pass the FUSE guardrails.
PG1424_ION_STAGE_NLTE_TRACE_ELEMENTS: Final[tuple[str, ...]] = (
    "F",
    "Ne",
    "Si",
    "S",
    "Ar",
    "Fe",
)


def mutable_atom_counts(
    counts: Mapping[str, Mapping[int, int]],
) -> dict[str, dict[int, int]]:
    """Return a defensive mutable copy for model-atom readers."""

    return {
        element: {int(charge): int(number) for charge, number in stages.items()}
        for element, stages in counts.items()
    }
