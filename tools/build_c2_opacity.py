"""Build reproducible C2 band opacities from checksum-pinned ExoMol 8states.

Line strengths include stimulated emission and the full statistical weight.
The resulting cross sections are line-bin averages, not a resolved pressure-
broadened line list. This approximation must be tested at increasing R.
"""

import argparse
import bz2
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen
import numpy as np

from wd_spectra.carbon_molecular import (
    C2_EXOMOL_ROOT_URL,
    C2_EXOMOL_SHA256,
    C2_EXOMOL_REFERENCE,
)


def line_strength(a, upper_weight, lower_energy, wavenumber, temperature, partition):
    return (
        a
        * upper_weight
        * np.exp(-1.438776877 * lower_energy / temperature)
        * -np.expm1(-1.438776877 * wavenumber / temperature)
        / (8 * np.pi * 2.99792458e10 * wavenumber**2 * partition)
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-directory", type=Path, default=Path(".cache/c2-exomol"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--fetch", action="store_true")
    p.add_argument(
        "--line-bin-only",
        action="store_true",
        help="legacy diagnostic artifact without the default rotational-overlap profile",
    )
    p.add_argument("--resolving-power", type=float, default=15000)
    p.add_argument(
        "--split-swan",
        action="store_true",
        help="retain a separate d(3Pi_g)-a(3Pi_u) opacity component for density-shift diagnostics",
    )
    args = p.parse_args()
    args.split_swan = args.split_swan or not args.line_bin_only
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite an opacity artifact")
    if not np.isfinite(args.resolving_power) or args.resolving_power < 100:
        raise ValueError("resolving power must be finite and >=100")
    for name, digest in C2_EXOMOL_SHA256.items():
        path = args.input_directory / name
        if not path.exists() and args.fetch:
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = urlopen(f"{C2_EXOMOL_ROOT_URL}/{name}", timeout=60).read()
            if hashlib.sha256(payload).hexdigest() != digest:
                raise ValueError(f"Checksum mismatch: {name}")
            path.write_bytes(payload)
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Checksum mismatch: {name}")
    with bz2.open(args.input_directory / "12C2__8states.states.bz2", "rt") as f:
        states = np.loadtxt(f, usecols=(0, 1, 2))
    ids = states[:, 0].astype(int)
    energy, weight = (np.zeros(ids.max() + 1) for _ in range(2))
    energy[ids], weight[ids] = states[:, 1], states[:, 2]
    electronic = np.empty(ids.max() + 1, dtype="U12")
    if args.split_swan:
        with bz2.open(args.input_directory / "12C2__8states.states.bz2", "rt") as f:
            electronic[ids] = np.loadtxt(f, usecols=(9,), dtype=str)
    temperatures = np.unique(
        np.r_[np.geomspace(1000, 10000, 40), np.geomspace(10000, 300000, 32)]
    )
    # Explicit finite-level sum continues above the published 10000-K PF
    # limit, not a frozen endpoint or extrapolated polynomial. Completeness
    # at high T is limited to 8states and is reported in the table provenance.
    partition = np.array(
        [
            np.sum(weight[ids] * np.exp(-1.438776877 * energy[ids] / t))
            for t in temperatures
        ]
    )
    published = np.loadtxt(args.input_directory / "12C2__8states.pf")
    at10k = float(np.sum(weight[ids] * np.exp(-1.438776877 * energy[ids] / 10000)))
    if abs(at10k / np.interp(10000, published[:, 0], published[:, 1]) - 1) > 0.01:
        raise ValueError("State sum disagrees with published partition function")
    nu_lo, nu_hi = 1000.0, 40000.0  # 0.25--10 microns
    edges = np.geomspace(
        nu_lo, nu_hi, int(np.ceil(args.resolving_power * np.log(nu_hi / nu_lo))) + 1
    )
    integrated = np.zeros((len(temperatures), len(edges) - 1))
    swan_integrated = np.zeros_like(integrated) if args.split_swan else None
    processed = accepted = 0
    from itertools import islice

    with bz2.open(args.input_directory / "12C2__8states.trans.bz2", "rt") as f:
        while True:
            lines = list(islice(f, 150000))
            if not lines:
                break
            values = np.fromstring("".join(lines), sep=" ").reshape(-1, 4)
            processed += len(values)
            upper, lower = values[:, :2].astype(int).T
            nu = (
                energy[upper] - energy[lower]
            )  # updated MARVEL energies, not old transition-file frequencies
            use = (nu >= nu_lo) & (nu <= nu_hi) & (values[:, 2] > 0)
            upper, lower, a, nu = upper[use], lower[use], values[use, 2], nu[use]
            index = np.clip(np.searchsorted(edges, nu) - 1, 0, len(edges) - 2)
            accepted += len(nu)
            swan = (
                (electronic[upper] == "d(3PIg)") & (electronic[lower] == "a(3PIu)")
                if args.split_swan
                else None
            )
            for j, (t, q) in enumerate(zip(temperatures, partition)):
                s = line_strength(a, weight[upper], energy[lower], nu, t, q)
                integrated[j] += np.bincount(index, weights=s, minlength=len(edges) - 1)
                if args.split_swan:
                    swan_integrated[j] += np.bincount(
                        index[swan], weights=s[swan], minlength=len(edges) - 1
                    )
            if processed % 1500000 == 0:
                print(
                    f"{processed:,} lines processed; {accepted:,} retained", flush=True
                )
    centers = np.sqrt(edges[:-1] * edges[1:])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    artifact = dict(
        wavelength_angstrom=(1e8 / centers)[::-1],
        temperature_K=temperatures,
        cross_section=(integrated / np.diff(edges))[:, ::-1].T,
        partition_temperature_K=temperatures,
        partition_function=partition,
        source=C2_EXOMOL_REFERENCE,
        resolving_power=args.resolving_power,
        **(
            {"swan_cross_section": (swan_integrated / np.diff(edges))[:, ::-1].T}
            if args.split_swan
            else {}
        ),
        provenance_json=json.dumps(
            dict(
                schema=2 if args.split_swan else 1,
                swan_component="d(3PIg)-a(3PIu)" if args.split_swan else None,
                sha256=C2_EXOMOL_SHA256,
                accepted_lines=accepted,
                frequency="updated state-energy differences",
                line_profile="constant-log-wavenumber bin average",
                temperature_policy="explicit 8states partition sum; completeness above 10000 K not established",
                missing_bands=["Deslandres-d'Azambuja", "Mulliken"],
                pressure_shift=False,
                data_license="CC-BY-SA-4.0",
                license_url="https://www.exomol.com/data/licence/",
                data_url="https://www.exomol.com/data/molecules/C2/12C2/8states/",
                modifications="Binned cross sections from updated state-energy differences; explicit finite-state partition sums",
            )
        ),
    )
    if not args.line_bin_only:
        from wd_spectra.c2_overlap import add_rotational_overlap

        artifact = add_rotational_overlap(
            artifact, args.input_directory, resolve_spin_projection=True
        )
    np.savez_compressed(args.output, **artifact)
    print(f"Wrote {args.output}: {accepted:,} lines, {len(centers):,} bins", flush=True)


if __name__ == "__main__":
    main()
