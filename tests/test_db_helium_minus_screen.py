"""The He- screen on real cold seeds: cool helium selects column mass, hot DBs optical depth."""

import pytest
from wd_spectra.models.stellar import DBConfig, _db_helium_minus_share


@pytest.mark.parametrize("teff,logg,column_mass", [
    (9000.0, 7.0, True),    # He- free-free dominates the outer layers
    (30000.0, 7.0, False),  # He- free-free is negligible
])
def test_helium_minus_share_separates_cool_and_hot_seeds(teff, logg, column_mass):
    share = _db_helium_minus_share(DBConfig(effective_temperature=teff, logg=logg), 40)
    assert 0.0 <= share < 1.0
    assert (share > 0.5) is column_mass
