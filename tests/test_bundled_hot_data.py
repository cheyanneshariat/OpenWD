"""The installed package contains the atomic inputs required by hot models."""

from wd_spectra.models import ModelData
from wd_spectra.models.pg1159 import required_atomic_files


def test_do_dao_atomic_inputs_are_bundled():
    data = ModelData.default()
    required = (
        data.ccc_hydrogen_collisions,
        data.tlusty_source,
        data.tlusty_helium_atom,
        data.cache / "helium-stark/Tremblay26.txt",
        data.helium_ii_stark,
    )
    assert all(path.is_file() for path in required)


def test_pg1159_atomic_inputs_are_bundled():
    data = ModelData.default()
    assert all(
        path.is_file()
        for path in required_atomic_files(data, "extended54-complete")
    )
    assert (data.stout / "stout").is_dir()
