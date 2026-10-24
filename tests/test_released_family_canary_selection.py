"""The DZ cold canary must qualify current public defaults at its saved star."""

import pytest

from wd_spectra import DZConfig
import test_released_family_canaries as canaries


def test_dz_cold_canary_uses_current_physics_without_a_stored_atmosphere(monkeypatch, tmp_path):
    class Dispatched(Exception):
        pass

    def inspect_request(config, output, *, require_convergence):
        defaults = DZConfig()
        assert config.effective_temperature == 10800.0
        assert config.logg == 8.0
        assert config.abundances["Ca"] == -8.06
        assert config.log_hydrogen_abundance == -4.05
        assert config.quality == "production"
        for key in ("dense_helium_eos", "dense_helium_free_free",
                    "helium_minus_free_free", "mg_i_b_he_unified_profile",
                    "mg_ii_he_unified_profile", "ca_ii_he_unified_profile",
                    "na_i_he_unified_profile"):
            assert getattr(config, key) == getattr(defaults, key)
        assert require_convergence is True
        assert output == tmp_path / "dz-pg1225"
        raise Dispatched

    monkeypatch.setattr(canaries, "run_model", inspect_request)
    with pytest.raises(Dispatched):
        canaries.test_released_family_cold_start("dz-pg1225", tmp_path)
