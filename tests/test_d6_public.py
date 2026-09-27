"""Fast checks of the public D6 configuration, data and scoring surface."""

import numpy as np
import pytest

from wd_spectra import D6Config, compute_d6, select_physics
from wd_spectra.d6 import (
    SDSS_J1637_LOG_NUMBER_ABUNDANCE,
    continuum_d6_atmosphere,
    gray_d6_atmosphere,
    synthesize_d6_spectrum,
)
from wd_spectra.metals import IONIZATION_ENERGY_EV
from wd_spectra.models.common import ModelData
from wd_spectra.models.d6 import _atomic_inputs
from wd_spectra.opacity import optical_depth_from_mass_opacity
from wd_spectra.validation.hollands_d6 import (
    hollands2025_scores,
    load_hollands2025_digitization,
)


@pytest.fixture(scope="module")
def j1637_inputs():
    return _atomic_inputs(ModelData.default(), tuple(SDSS_J1637_LOG_NUMBER_ABUNDANCE))


def test_default_config_is_the_published_j1637_solution():
    config = D6Config()
    assert (config.effective_temperature, config.logg) == (15_680.0, 6.3)
    assert dict(config.abundances) == dict(SDSS_J1637_LOG_NUMBER_ABUNDANCE)
    assert select_physics(config).workflow == "d6"


@pytest.mark.parametrize(
    "abundances, message",
    [
        ({"C": 0.0, "He": -2.0}, "hydrogen/helium-free"),
        ({"O": 0.0}, "reference element"),
        ({"C": 0.1, "O": 0.3}, "abundance zero"),
        ({"C": 0.0, "Xx": -2.0}, "unsupported element"),
    ],
)
def test_invalid_compositions_are_rejected_before_any_solve(abundances, message):
    with pytest.raises(ValueError, match=message):
        compute_d6(D6Config(abundances=abundances))


def test_bundled_atomic_inputs_cover_the_level_resolved_continua(j1637_inputs):
    database, photoionization, topbase = j1637_inputs
    stages = set(topbase.ion_stages)
    for stage in [("C", 0), ("C", 1), ("O", 0), ("O", 1), ("Fe", 1), ("Si", 0)]:
        assert stage in stages
    # Complete ladders for the bulk donors, three stages for trace elements.
    assert [ion.charge for ion in database.ion_stages("O")] == list(range(9))
    assert [ion.charge for ion in database.ion_stages("Fe")] == [0, 1, 2]
    # Lines only for the validated low stages.
    assert all(not ion.transitions for ion in database.ion_stages("C") if ion.charge > 2)


def test_continuum_seed_is_hydrostatic_on_its_rosseland_scale(j1637_inputs):
    database, photoionization, topbase = j1637_inputs
    seed = continuum_d6_atmosphere(
        15_680.0, 6.3, database, photoionization, SDSS_J1637_LOG_NUMBER_ABUNDANCE,
        n_depth=12, n_wavelength=200, topbase_photoionization_database=topbase,
    )
    assert seed.metadata["continuum_seed_converged"]
    np.testing.assert_allclose(seed.gas_pressure, 10.0**6.3 * seed.column_mass)
    assert seed.rosseland_optical_depth[0] == pytest.approx(1.0e-8)
    assert seed.rosseland_optical_depth[-1] == pytest.approx(1.0e2)
    # Deep layers follow the gray law instead of a clipped isothermal floor.
    assert np.all(np.diff(seed.temperature) > 0.0)
    assert seed.temperature[-1] / 15_680.0 == pytest.approx(
        (0.75 * (100.0 + 2.0 / 3.0)) ** 0.25
    )


def test_formal_spectrum_is_finite_and_records_its_physics(j1637_inputs):
    database, photoionization, topbase = j1637_inputs
    atmosphere = gray_d6_atmosphere(
        15_680.0, 6.3, database, SDSS_J1637_LOG_NUMBER_ABUNDANCE, n_depth=16,
    )
    wavelength = np.linspace(6150.0, 6170.0, 41)
    spectrum = synthesize_d6_spectrum(
        atmosphere, wavelength, database, photoionization,
        SDSS_J1637_LOG_NUMBER_ABUNDANCE, maximum_metal_lines=200,
        topbase_photoionization_database=topbase, n_angle=2,
    )
    assert np.all(np.isfinite(spectrum.surface_flux_lambda))
    assert np.all(spectrum.surface_flux_lambda > 0.0)
    assert spectrum.metadata["composition"] == "hydrogen-helium-free-bulk-metals"
    assert spectrum.metadata["transfer_discretization"] == "formal-linear"


def test_hollands_scores_are_zero_for_the_digitized_model_itself():
    published = load_hollands2025_digitization()
    wavelength = np.arange(3450.0, 7620.0, 0.1)
    # Piece the four normalized Koester panels together and tilt them with a
    # smooth continuum; the local normalization must remove the tilt.
    flux = np.ones_like(wavelength)
    for panel, (lower, upper) in enumerate(
        [(3580.0, 4600.0), (4600.0, 5600.0), (5600.0, 6600.0), (6600.0, 7620.0)], start=1
    ):
        inside = (wavelength >= lower) & (wavelength < upper)
        flux[inside] = np.interp(
            wavelength[inside],
            published[f"panel{panel}_koester_wavelength"],
            published[f"panel{panel}_koester_normalized_flux"],
        )
    tilted = flux * (wavelength / 5000.0) ** -3
    scores = hollands2025_scores(wavelength, tilted, already_convolved=True)
    assert len(scores["koester_panel_rms"]) == 4
    # The authors normalized their panels differently from the log-linear
    # development metric, whose floor is therefore not zero ...
    assert 0.015 < scores["koester_mean_panel_rms"] < 0.03
    # ... while the matched metric recovers the digitized model exactly.
    assert scores["koester_mean_matched_panel_rms"] < 2.0e-3
    assert np.isfinite(scores["broad_refluxed_koester_rms_fractional"])
