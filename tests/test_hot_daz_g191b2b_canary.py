"""Fresh 40-depth G191-B2B cold start: H/He NLTE host, then nine NLTE trace metals.

Slow end-to-end canary (several hours, one thread). The production version
of this model (80 depths) is the documented G191-B2B comparison; it is not run
here. The stored reference holds the six documented comparison windows of a
run of research/hot_daz_g191b2b.py with --quality standard.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

from wd_spectra._compat import trapezoid

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = Path(__file__).parent / "data/hot_daz/g191b2b-standard.npz"
# Equivalent-width windows (A) around lines of seven model elements.
LINES = {"C III 1175": (1174.6, 1176.7), "N V 1239": (1238.5, 1239.2), "O IV 1339": (1338.4, 1338.9),
         "Si IV 1394": (1393.4, 1394.1), "Fe V 1409": (1409.3, 1409.6),
         "Al III 1855": (1854.5, 1855.0), "P V 1118": (1117.7, 1118.3)}


def equivalent_widths(wavelength, flux, background):
    depth = 1.0 - flux / background
    return {name: trapezoid(depth[(wavelength >= lo) & (wavelength <= hi)],
                            wavelength[(wavelength >= lo) & (wavelength <= hi)])
            for name, (lo, hi) in LINES.items()}


@pytest.mark.canary
def test_g191b2b_40_depth_cold_start(tmp_path):
    sys.path.insert(0, str(ROOT / "research"))
    from hot_daz_g191b2b import run
    from wd_spectra.hot_trace_metals import FLUX_TOLERANCE

    # run() raises unless the H/He host cold start converges.
    result, _ = run(tmp_path / "g191b2b", quality="standard")
    assert result.converged and result.metadata["convergence_criterion"] == "flux"
    checked = [x for x in result.metadata["flux_defect_history"] if x is not None]
    assert checked and checked[-1] < FLUX_TOLERANCE

    with np.load(tmp_path / "g191b2b/spectrum.npz") as model, np.load(REFERENCE) as reference:
        wave = reference["wavelength"]
        flux = np.interp(wave, model["wavelength"], model["flux"])
        background = np.interp(wave, model["wavelength"], model["background_flux"])
        assert np.all(np.isfinite(flux)) and np.all(flux > 0)
        # The fixed point is converged to 3e-3 in flux; allow a few times that.
        assert np.max(abs(flux / reference["flux"] - 1)) < 1e-2
        np.testing.assert_allclose(background, reference["background_flux"], rtol=1e-3)
        new = equivalent_widths(wave, flux, background)
        old = equivalent_widths(wave, reference["flux"], reference["background_flux"])
    for name in LINES:
        assert old[name] > 5e-3, name
        assert abs(new[name] / old[name] - 1) < 1e-2, (name, new[name], old[name])
