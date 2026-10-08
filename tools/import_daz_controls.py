"""Import synthetic DAZ paper controls; never copy observations or refit flux.

Only needed when initially curating these fixtures, not by package users/CI.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import numpy as np
from wd_spectra import DAZConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", type=Path)
    args = parser.parse_args()
    output = Path(__file__).resolve().parents[1] / "tests/data/daz_regressions"
    output.mkdir(exist_ok=True)
    for case, atmosphere_folder in (
        ("g149_28", "daz-converged-g149-28-v1"),
        ("galex1931", "daz-converged-galex1931-v1"),
    ):
        destination = output / (case + ".npz")
        if destination.exists():
            raise FileExistsError(destination)
        original = (
            args.workspace
            / "results/dz-paper-seven-resolution-matched-v1"
            / (case + ".npz")
        )
        structure = (
            args.workspace / "results" / atmosphere_folder / (case + "-atmosphere.npz")
        )
        info = json.loads(original.with_suffix(".json").read_text())
        with np.load(structure) as a, np.load(original) as spectrum:
            config = DAZConfig(
                effective_temperature=info["effective_temperature"],
                logg=info["logg"],
                abundances=info["published_abundances"],
                quality="production",
                lyman_profile_source="stark",
                metal_neutral_h_broadening="unsold",
            )
            # Exact original samples over the paper's plotted interval.
            select = (spectrum["model_wavelength_vacuum"] >= 3700) & (
                spectrum["model_wavelength_vacuum"] <= 4500
            )
            np.savez_compressed(
                destination,
                **{
                    key: a[key]
                    for key in (
                        "effective_temperature",
                        "logg",
                        "temperature",
                        "gas_pressure",
                        "column_mass",
                        "rosseland_optical_depth",
                    )
                },
                wavelength=spectrum["model_wavelength_vacuum"][select],
                original_surface_flux=spectrum["model_surface_flux_lambda"][select],
                config_json=json.dumps(asdict(config)),
                provenance_json=json.dumps(
                    {
                        "synthetic_source": str(original.relative_to(args.workspace)),
                        "synthetic_sha256": hashlib.sha256(
                            original.read_bytes()
                        ).hexdigest(),
                        "atmosphere_source": str(structure.relative_to(args.workspace)),
                        "atmosphere_sha256": hashlib.sha256(
                            structure.read_bytes()
                        ).hexdigest(),
                        "convergence_certified": False,
                    }
                ),
            )
        print(destination)


if __name__ == "__main__":
    main()
