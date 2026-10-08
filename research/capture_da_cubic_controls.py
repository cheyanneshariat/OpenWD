"""Archive already checked cold/cubic outputs, without recalculating a spectrum.

The old spectral_regressions fixtures are never replaced. These new controls
record the reviewed interpolation change and corrected EOS separately.
"""
import json
from pathlib import Path
import numpy as np
from rebuild_paper_figure2 import RELEASE, identity

BASE = RELEASE / "results/da-flux-normalization-20260910"
CASES = {
    "da-12000-public": (BASE / "public-cold-default/cubic", None),
    "da-20000": (RELEASE / "results/figure1-microphysics-20260910/models/t20000_g8.00",
                 BASE / "qualified-comparisons/da20"),
}


def main():
    output = RELEASE / "tests/data/da_cubic_regressions"
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for case, (cold, comparison) in CASES.items():
        meta = json.loads((cold / "metadata.json").read_text())
        assert meta["atmosphere_metadata"]["equilibrium_certificate"]["verified"]
        assert meta["atmosphere_metadata"]["initial_temperature_was_supplied"] is False
        if comparison is None:
            wave, flux = np.loadtxt(cold / "spectrum.txt", unpack=True)
            source = identity(cold / "spectrum.txt")
            assert meta["spectrum_metadata"]["transfer_discretization"] == "formal-pchip"
            assert meta["spectrum_metadata"]["source_converged"]
        else:
            qualification = json.loads((comparison / "summary.json").read_text())
            assert qualification["source_closure"]["source_converged"]
            assert qualification["cold_atmosphere"] == identity(cold / "atmosphere.npz")
            with np.load(comparison / "spectra.npz") as data:
                wave, flux = data["wavelength"], data["new_flux"]
            source = identity(comparison / "spectra.npz")
        # Retain actual evaluated wavelengths, not an interpolated reference.
        take = np.arange(len(wave)) % 4 == 0
        take &= (wave >= 900.) & (wave <= 300000.)
        with np.load(cold / "atmosphere.npz") as data:
            saved = {key: data[key] for key in data.files}
        record = dict(case=case, cold_metadata=identity(cold / "metadata.json"),
                      cold_atmosphere=identity(cold / "atmosphere.npz"), spectrum=source,
                      selection="Every fourth evaluated wavelength, 900-300000 Angstrom",
                      atmosphere_cold_start_verified=True, interpolation="formal-pchip",
                      historical_controls_preserved=True,
                      reason="User-approved cubic default after corrected cold-start/paper comparisons")
        saved.update(config_json=json.dumps(meta["config"]), spectral_type="DA",
                     wavelength=wave[take], original_surface_flux=flux[take],
                     provenance_json=json.dumps(record))
        np.savez_compressed(output / (case + ".npz"), **saved)
        records.append(record)
    (output / "manifest.json").write_text(json.dumps(records, indent=2)+"\n")


if __name__ == "__main__":
    main()
