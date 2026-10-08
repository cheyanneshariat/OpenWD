"""Rebuild the approximate historical C2 Deslandres-d'Azambuja LTE opacity.

Cooper (NASA-TM-78574, 1979) Table 5.2 and Appendix A set the absolute
electronic moment and absorption-oscillator convention. Nicholls (1965) Table
6 supplies Franck-Condon factors. Cooper Table 2.1 constants reproduce the
vacuum band origins in Table B-4; the sign of C-state omega_e*y_e is negative.

Actual ExoMol lower-A-state populations avoid an extra singlet/triplet or
homonuclear statistical-weight guess. Each historical band is distributed
using the finite-bin rigid-rotor integrator in dq_ca_band_envelope.py.
The full 8states partition function and existing molecular chemistry remain
unchanged. This is constitutive input, NOT an equilibrated model,
modern C-A line list, or calibrated C2-He profile. High-v perturbations,
rotational line factors beyond asymptotic P/R weights, and transition-moment
variation with internuclear separation are not resolved by this estimate.
"""

import argparse
import bz2
import hashlib
import json
from pathlib import Path

import numpy as np

from dq_ca_band_envelope import geometry_for_band, exact_band_fractions
from wd_spectra.carbon_molecular import C2_EXOMOL_SHA256
from wd_spectra.constants import ELEMENTARY_CHARGE_ESU, ELECTRON_MASS, LIGHT_SPEED

HC_K = 1.438776877
HARTREE_CM = 219474.6313632
MU2 = 0.93  # summed |R_e/e a0|^2; Cooper -1 sequence, uncertainty +/-0.18
FC = np.array(
    [
        [
            5.5083e-1,
            3.0217e-1,
            1.0619e-1,
            3.0532e-2,
            7.8455e-3,
            1.8797e-3,
            4.2980e-4,
            9.5066e-5,
            2.0502e-5,
        ],
        [
            3.5613e-1,
            9.4332e-2,
            2.6413e-1,
            1.7509e-1,
            7.4478e-2,
            2.5412e-2,
            7.6113e-3,
            2.0945e-3,
            5.4340e-4,
        ],
        [
            8.4095e-2,
            3.9457e-1,
            1.1158e-4,
            1.5034e-1,
            1.8469e-1,
            1.1138e-1,
            4.8806e-2,
            1.7821e-2,
            5.7893e-3,
        ],
        [
            8.6154e-3,
            1.8062e-1,
            3.1377e-1,
            3.3201e-2,
            5.8565e-2,
            1.5371e-1,
            1.3025e-1,
            7.1957e-2,
            3.1497e-2,
        ],
        [
            3.4244e-4,
            2.6977e-2,
            2.5951e-1,
            2.0869e-1,
            8.8989e-2,
            1.0906e-2,
            1.0660e-1,
            1.2967e-1,
            8.9610e-2,
        ],
        [
            2.6811e-6,
            1.3440e-3,
            5.3123e-2,
            3.1208e-1,
            1.1889e-1,
            1.2873e-1,
            1.8475e-4,
            6.1353e-2,
            1.1405e-1,
        ],
        [
            1.6219e-8,
            1.0619e-5,
            3.1685e-3,
            8.4248e-2,
            3.3965e-1,
            5.6088e-2,
            1.4428e-1,
            1.1745e-2,
            2.7417e-2,
        ],
    ]
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def band_constants(vu, vl):
    xu, xl = vu + 0.5, vl + 0.5
    gu = 1809.1 * xu - 15.81 * xu**2 - 4.02 * xu**3
    gl = 1608.35 * xl - 12.078 * xl**2 - 0.010 * xl**3
    origin = 34261.9 - 8391.00 + gu - gl
    return origin, 1.783 - 0.018 * xu, 1.61634 - 0.01686 * xl - 0.00005 * xl**2


def oscillator_strength(origin, franck_condon, moment=MU2, electronic_degeneracy=2):
    """Cooper App. A expressed in atomic units; f is dimensionless."""
    return (
        (2 / 3) * origin / HARTREE_CM * moment * franck_condon / electronic_degeneracy
    )


def lower_states(path):
    if digest(path) != C2_EXOMOL_SHA256[path.name]:
        raise ValueError("8states lower-population checksum mismatch")
    rows = []
    with bz2.open(path, "rt") as stream:
        for line in stream:
            fields = line.split()
            if fields[9] == "A(1PIu)":
                rows.append((float(fields[1]), float(fields[2]), int(fields[10])))
    result = np.asarray(rows)
    if len(result) == 0 or not np.isfinite(result).all():
        raise ValueError("missing/nonfinite singlet-A lower states")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--table",
        type=Path,
        default=Path(__file__).resolve().parents[1]/"src/wd_spectra/data/dq/c2-8states-r15000.npz",
    )
    parser.add_argument(
        "--states",
        type=Path,
        required=True,
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    for (vu, vl), wavelength in {
        (0, 0): 3850.62,
        (0, 2): 4381.01,
        (0, 3): 4697.00,
        (1, 3): 4337.53,
        (2, 4): 4302.94,
    }.items():
        if abs(1e8 / band_constants(vu, vl)[0] - wavelength) > 0.01:
            raise AssertionError("Historical vacuum origins not reproduced")
    # Independent scale sanity check: Swan f00 is ~0.025 for Cooper's moment.
    swan_f00 = oscillator_strength(19378.44, 0.73521, 3.52, 6)
    if not 0.024 < swan_f00 < 0.026:
        raise AssertionError("electronic degeneracy/atomic-unit conversion error")
    with np.load(args.table, allow_pickle=False) as archive:
        base = {name: archive[name] for name in archive.files}
    states = lower_states(args.states)
    center = (1e8 / base["wavelength_angstrom"])[::-1]
    dl = np.diff(np.log(center))
    if not np.allclose(dl, dl[0], rtol=1e-7):
        raise ValueError("constant-log-frequency source table required")
    edges = np.r_[center * np.exp(-dl[0] / 2), center[-1] * np.exp(dl[0] / 2)]
    widths = np.diff(edges)
    bands = []
    for vu in range(7):
        for vl in range(9):
            origin, bu, bl = band_constants(vu, vl)
            bsum, bdiff = bu + bl, bu - bl
            mhead = -bsum / (2 * bdiff)
            head = origin - bsum**2 / (4 * bdiff)
            bands.append(
                dict(
                    vu=vu,
                    vl=vl,
                    origin=origin,
                    bu=bu,
                    bl=bl,
                    f=oscillator_strength(origin, FC[vu, vl]),
                    geometry=geometry_for_band(edges, origin, bu, bl),
                    head_A=float(1e8 / head),
                    head_physical=bool(mhead <= -1 or mhead >= 1),
                )
            )
    addition = np.zeros_like(base["cross_section"])
    low_upper = np.zeros_like(addition)
    records = []
    for it, t in enumerate(base["temperature_K"]):
        q = np.exp(
            np.interp(
                np.log(t),
                np.log(base["partition_temperature_K"]),
                np.log(base["partition_function"]),
            )
        )
        population = (
            np.bincount(
                states[:, 2].astype(int),
                weights=states[:, 1] * np.exp(-HC_K * states[:, 0] / t),
                minlength=9,
            )
            / q
        )
        cross = np.zeros(len(center))
        low = np.zeros_like(cross)
        band_records = []
        for band in bands:
            fractions, missing = exact_band_fractions(
                band["geometry"], band["bl"], float(t), len(center)
            )
            # Integrated cross section over wavenumber: pi e^2/(me c^2) f N_l/N.
            strength = (
                np.pi
                * ELEMENTARY_CHARGE_ESU**2
                / (ELECTRON_MASS * LIGHT_SPEED**2)
                * band["f"]
                * population[band["vl"]]
            )
            sigma = (
                strength
                * fractions
                / widths
                * center
                / band["origin"]
                * -np.expm1(-HC_K * center / t)
            )
            cross += sigma
            if band["vu"] <= 2:
                low += sigma
            band_records.append(
                dict(
                    vu=band["vu"],
                    vl=band["vl"],
                    population=float(population[band["vl"]]),
                    nominal_area_cm=float(strength),
                    finite_grid_fraction_lost=missing,
                )
            )
        addition[:, it] = cross[::-1]
        low_upper[:, it] = low[::-1]
        if it in {
            int(np.argmin(abs(base["temperature_K"] - target)))
            for target in (5050, 7182, 8476)
        }:
            wave = base["wavelength_angstrom"]
            original = (
                base["cross_section"][:, it]
                - base["swan_cross_section"][:, it]
                + base["swan_rotational_overlap_cross_section"][:, it]
            )
            pts = {}
            for w in (4339.0, 4365.0, 4381.0, 4700.0, 5165.0):
                a, o = np.interp(w, wave, addition[:, it]), np.interp(w, wave, original)
                pts[str(w)] = dict(
                    addition=float(a), original_C2=float(o), ratio=float(a / o)
                )
            use = (wave >= 4280) & (wave <= 4405)
            records.append(
                dict(
                    temperature=float(t),
                    points=pts,
                    added_peak_A=float(wave[use][np.argmax(addition[use, it])]),
                    bands=band_records,
                )
            )
        if it % 10 == 0:
            print(
                f"C-A diagnostic {it+1}/{len(base['temperature_K'])}, T={t:.0f}",
                flush=True,
            )
    if not np.isfinite(addition).all() or np.any(addition < 0):
        raise ValueError("invalid historical C-A opacity estimate")
    provenance = json.loads(str(base["provenance_json"].item()))
    provenance["historical_C_A_diagnostic"] = {
        "description": __doc__,
        "moment_squared_au": MU2,
        "moment_uncertainty_au": 0.18,
        "lower_electronic_degeneracy": 2,
        "v_upper_range": [0, 6],
        "v_lower_range": [0, 8],
        "states_sha256": digest(args.states),
        "parent_sha256": digest(args.table),
        "implementation_sha256": digest(Path(__file__)),
        "sources": [
            "https://ntrs.nasa.gov/citations/19790013711",
            "https://nvlpubs.nist.gov/nistpubs/jres/69A/jresv69An5p397_A1b.pdf",
        ],
    }
    output = dict(base)
    output["cross_section"] = base["cross_section"] + addition
    output["dazambuja_diagnostic_cross_section"] = addition
    output["dazambuja_low_upper_diagnostic_cross_section"] = low_upper
    output["cross_section_original"] = base["cross_section"]
    output["source"] = (
        "8states + approximate historical C-A opacity estimate"
    )
    output["provenance_json"] = json.dumps(provenance)
    table_path = args.output / "c2-with-historical-dazambuja.npz"
    np.savez_compressed(table_path, **output)
    for b in bands:
        del b["geometry"]
    report = dict(
        scope=__doc__,
        bands=bands,
        selected_temperatures=records,
        swan_f00_scale_check=float(swan_f00),
        output=str(table_path),
        output_sha256=digest(table_path),
        notes=[
            "No stellar parameters or line wavelengths fitted.",
            "Opacity estimate only; atmosphere and molecular chemistry are not re-equilibrated.",
            "Historical low-order constants become unreliable for perturbed high-v C levels; retain this limitation when using the approximate envelope.",
        ],
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(table_path),
                "selected": [
                    {k: v for k, v in row.items() if k != "bands"} for row in records
                ],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
