"""DAB Allard Lyman-profile Teff policy; table contents are stubbed."""

import pytest
from wd_spectra import DABConfig
from wd_spectra.models import ModelData, stellar
from wd_spectra.models.daz import DAZConfig
from wd_spectra.models.stellar import (
    DAConfig,
    DZConfig,
    _allard_lyman_profiles_for_effective_temperature as allard_for,
)


@pytest.fixture
def tables(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(ModelData, "require", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        stellar, "read_allard_temperature_dependent_tables", lambda *a, **k: sentinel
    )
    return sentinel


def test_dab_default_has_no_allard_teff_cutoff():
    assert DABConfig().allard_minimum_effective_temperature is None
    # The other presets keep their documented 9000-K boundary.
    for config in (DAConfig(), DAZConfig(), DZConfig()):
        assert config.allard_minimum_effective_temperature == 9_000.0


@pytest.mark.parametrize("teff", [5_000.0, 8_000.0, 9_000.0, 20_000.0])
def test_no_minimum_selects_delivered_tables_at_every_teff(tables, teff):
    data = ModelData.default()
    selected = allard_for(
        teff, data, minimum_effective_temperature=None,
        use_fixed_low_temperature_profile=False,
    )
    assert selected is tables


def test_explicit_minimum_still_disables_allard_below_it(tables):
    data = ModelData.default()
    assert allard_for(
        8_000.0, data, minimum_effective_temperature=9_000.0,
        use_fixed_low_temperature_profile=False,
    ) is None
    assert allard_for(
        9_000.0, data, minimum_effective_temperature=9_000.0,
        use_fixed_low_temperature_profile=False,
    ) is tables


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_minimum_is_rejected(tables, bad):
    with pytest.raises(ValueError):
        allard_for(8_000.0, ModelData.default(), minimum_effective_temperature=bad)
