import hashlib
from pathlib import Path
import tarfile
import pytest
import research_paths
from experiment_source_archive import archive_sources
from run_cool_db import commands
import check_data


def test_data_path_is_explicit_and_independent_of_working_directory(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENWD_RESEARCH_DATA", raising=False)
    expected = research_paths.data_directory()
    monkeypatch.chdir(tmp_path)
    assert research_paths.data_directory() == expected
    monkeypatch.setenv("OPENWD_RESEARCH_DATA", str(tmp_path / "tables"))
    assert research_paths.data_directory() == tmp_path / "tables"


def test_source_archive_contains_the_actual_checkout_from_another_cwd(monkeypatch, tmp_path):
    paths = research_paths.source_paths()
    assert Path("wd_spectra/adaptive_structure.py") in paths
    assert Path("wd_spectra/_cool/run_molecular_dab_mass_experiment.py") in paths
    monkeypatch.chdir(tmp_path)
    selected = paths[:2]
    manifest = archive_sources(selected, tmp_path / "sources.tar.gz", base_directory=research_paths.REPOSITORY)
    with tarfile.open(tmp_path / "sources.tar.gz") as archive:
        for path in selected:
            content = archive.extractfile(str(path)).read()
            assert content == (research_paths.REPOSITORY / path).read_bytes()
            assert hashlib.sha256(content).hexdigest() == manifest[str(path)]


def test_dense_recipe_has_no_temperature_switch_or_supplied_state(tmp_path):
    low, audit = commands(5000, tmp_path, tmp_path / "table.npz")
    high, _ = commands(8000, tmp_path, tmp_path / "table.npz")
    assert [x.replace("5000", "8000") for x in low] == high
    assert not any("resume" in x or "predictor" in x or "refine-atmosphere" in x for x in low)
    assert "--physical-only" in low and "--no-continuations" in low
    assert "--mass-conservative-transfer" in low and "--exact-convection-tangent" in low
    assert audit[audit.index("--maximum-wavelength") + 1] == "1e9"


def test_external_data_check_fails_closed(monkeypatch, tmp_path):
    payload = b"unit-test table, not physical data"
    monkeypatch.setattr(check_data, "SHA256", {"table": hashlib.sha256(payload).hexdigest()})
    with pytest.raises(FileNotFoundError):
        check_data.verify(tmp_path)
    (tmp_path / "table").write_bytes(payload)
    check_data.verify(tmp_path)
    (tmp_path / "table").write_bytes(b"different")
    with pytest.raises(ValueError, match="differs"):
        check_data.verify(tmp_path)


def test_bundled_hnc_cache_is_the_qualified_generated_table():
    table = Path(__file__).parent / "data/molecular-hnc-electron-domain-table.npz"
    assert hashlib.sha256(table.read_bytes()).hexdigest() == "4bf3134197d7309ccb6ee227bf2a659c30438ec263e322cb020aec8371fc10e7"
