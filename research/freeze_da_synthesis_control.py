"""Explicit one-time capture of the established DA synthesis, not a test updater.

Uses the already saved 12,000 K atmosphere, forbids atmosphere iteration, and
uses the established formal operator. Never run by pytest. The reference was
captured before restoration; this utility does not refresh it after code edits.
"""

import json
from pathlib import Path
import numpy as np
from wd_spectra import models
from wd_spectra.models import stellar

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "examples/results/example/da-12000K-20260908T070221Z-3280a330"
OUTPUT = ROOT / "tests/data/spectral_regressions/da-12000-public.npz"


def main():
    if OUTPUT.exists():
        raise FileExistsError(
            "The regression reference is immutable; review any replacement"
        )
    metadata = json.loads((SOURCE / "metadata.json").read_text())
    cfg = models.DAConfig(**metadata["config"])
    state = models.load_atmosphere_checkpoint(
        SOURCE / "atmosphere.npz",
        cfg.effective_temperature,
        cfg.logg,
        "hydrogen",
        include_molecules=True,
        include_negative_hydrogen=True,
        trihydrogen_ion_partition_model="neale-tennyson-1995",
    )
    with np.load(SOURCE / "atmosphere.npz") as saved:
        for name in (
            "temperature",
            "column_mass",
            "gas_pressure",
            "density",
            "electron_density",
        ):
            np.testing.assert_allclose(getattr(state, name), saved[name], rtol=1e-8)
    wave = np.unique(
        np.r_[
            np.geomspace(900.0, 30000.0, 700),
            np.arange(6250.0, 6900.01, 0.5),
            np.arange(4700.0, 5025.01, 0.5),
            np.arange(4250.0, 4450.01, 0.5),
            np.arange(4020.0, 4180.01, 0.5),
        ]
    )

    def forbidden(*a, **kw):
        raise AssertionError("No atmosphere iteration in a fixed-synthesis reference")

    original = stellar.radiative_equilibrium_hydrogen_atmosphere
    stellar.radiative_equilibrium_hydrogen_atmosphere = forbidden
    try:
        result = models.compute_da(
            cfg,
            wave,
            initial_atmosphere=state,
            relax_atmosphere=False,
        )
    finally:
        stellar.radiative_equilibrium_hydrogen_atmosphere = original
    with np.load(SOURCE / "atmosphere.npz") as saved:
        arrays = {k: saved[k] for k in saved.files}
    arrays.update(
        config_json=json.dumps(metadata["config"]),
        spectral_type="DA",
        wavelength=wave,
        original_surface_flux=result.spectrum.surface_flux_lambda,
        provenance_json=json.dumps(
            dict(
                source=str(SOURCE.relative_to(ROOT)),
                reference_operator="established formal-linear default",
                reference_is_observed_spectrum=False,
                atmosphere_depths=state.n_depth,
                atmosphere_iterations=0,
                purpose="Public-default regression, including Balmer cores and wings",
            )
        ),
    )
    np.savez_compressed(OUTPUT, **arrays)
    print(OUTPUT, len(wave), "wavelengths", flush=True)


if __name__ == "__main__":
    main()
