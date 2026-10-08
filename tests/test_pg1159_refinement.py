"""Public surface of the PG1159 upper-atmosphere refinement."""

from dataclasses import replace

import numpy as np
import pytest

from wd_spectra._pg1159_ali_temperature import REFINEMENT_LINE_VELOCITY_SAMPLES_KMS
from wd_spectra.models.pg1159 import PG1159Config, validate_config


def test_refinement_is_opt_in():
    assert PG1159Config().refine_upper_atmosphere is False


def test_refinement_flag_must_be_boolean():
    validate_config(replace(PG1159Config(), refine_upper_atmosphere=True))
    with pytest.raises(ValueError, match="refine_upper_atmosphere"):
        validate_config(replace(PG1159Config(), refine_upper_atmosphere=1))


def test_refinement_stencil_resolves_the_thermal_core():
    samples = np.asarray(REFINEMENT_LINE_VELOCITY_SAMPLES_KMS)
    assert samples.size == 17
    assert np.all(np.diff(samples) > 0)
    assert np.allclose(samples, -samples[::-1])
    assert 0.0 in samples
    # At least two samples per side within 20 km/s of line centre.
    assert np.count_nonzero((abs(samples) > 0) & (abs(samples) <= 20.0)) >= 4
