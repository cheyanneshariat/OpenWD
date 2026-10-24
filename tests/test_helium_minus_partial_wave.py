"""First-principles He- free-free shape below John's (1994) 0.5063-um edge."""
import json

import numpy as np
import pytest

from wd_spectra.helium import (
    _PARTIAL_WAVE_TABLE,
    helium_minus_free_free_coefficient as k,
)
from wd_spectra.models.stellar import DZConfig

EDGE = 0.5063


@pytest.mark.parametrize("temperature", [2000.0, 3000.0, 4500.0, 6000.0, 9000.0])
def test_continuous_at_john_edge(temperature):
    below = k(np.array([EDGE * (1 - 1e-9)]), temperature, prescription="partial-wave")
    above = k(np.array([EDGE]), temperature, prescription="partial-wave")
    assert below[0] == pytest.approx(above[0], rel=1e-6)
    # the existing automatic prescription jumps here
    auto_below = k(np.array([EDGE * (1 - 1e-9)]), temperature)
    if temperature <= 6000.0:
        assert auto_below[0] / above[0] > 1.05


def test_identical_to_automatic_where_john_or_outside_range():
    wave = np.array([0.6, 1.0, 5.0, 20.0])
    for t in (3000.0, 8000.0):
        assert np.array_equal(k(wave, t, prescription="partial-wave"), k(wave, t))
    blue = np.array([0.2, 0.3, 0.45])
    for t in (12000.0, 20000.0):  # outside John's temperature range
        assert np.array_equal(k(blue, t, prescription="partial-wave"), k(blue, t))


def test_blue_shape_below_carbon_fit_and_smooth():
    wave = np.geomspace(0.2, EDGE, 40)
    for t in (3000.0, 5000.0):
        pw = k(wave, t, prescription="partial-wave")
        assert np.all(np.diff(np.log(pw)) > 0)          # rises toward the red
        assert np.all(pw <= k(wave, t, prescription="john1968") * 1.0001)


def test_table_metadata_records_john_agreement():
    with np.load(_PARTIAL_WAVE_TABLE) as data:
        meta = json.loads(str(data["metadata_json"]))
    r = meta["john1994_ratio_lambda_ge_0.5063um"]
    assert 0.94 < r["min"] and r["max"] < 1.03


def test_config_default_and_validation():
    assert DZConfig().helium_minus_free_free == "partial-wave"
    from wd_spectra.models.stellar import compute_dz
    with pytest.raises(ValueError, match="helium_minus_free_free"):
        compute_dz(DZConfig(helium_minus_free_free="bogus"), relax_atmosphere=False)
