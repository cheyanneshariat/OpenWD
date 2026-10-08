"""Test notebook interfaces and failure paths without computing an atmosphere.

Synthetic recovery checks the fitting code. Saved output checks establish
execution evidence, not physical validation or calibrated abundance errors.
"""

import importlib.util
import json
from pathlib import Path
import sys
import ssl
import warnings

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from wd_spectra.models import AtmosphereConvergenceWarning, DZConfig


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "examples/fit_dz_spectrum.ipynb"
CELLS = json.loads(NOTEBOOK.read_text())["cells"]
CODE = {c["id"]: "".join(c["source"]) for c in CELLS if c["cell_type"] == "code"}


def execute(cell, namespace):
    exec(compile(CODE[cell], f"notebook:{cell}", "exec"), namespace)


def output_text(cell_id):
    cell = next(c for c in CELLS if c.get("id") == cell_id)
    return "".join("".join(o.get("text", [])) for o in cell["outputs"])


@pytest.fixture
def notebook(monkeypatch):
    monkeypatch.chdir(NOTEBOOK.parent)
    monkeypatch.setattr(sys, "path", sys.path.copy())
    namespace = {"__name__": "__main__"}
    execute("setup", namespace)
    execute("controls", namespace)
    return namespace


@pytest.fixture
def data_tools():
    # Archive preparation is optional; the notebook uses the bundled coadd.
    path = ROOT / "docs/examples/prepare_pg1225.py"
    spec = importlib.util.spec_from_file_location("prepare_pg1225", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.__dict__


def test_bundled_spectrum_loads_without_network_or_optional_fits_reader(notebook, monkeypatch):
    monkeypatch.setitem(sys.modules, "astropy", None)
    execute("data", notebook)
    assert notebook["obs_wave"].shape == (40000,)
    assert notebook["RESOLVING_POWER"] == 19540.0
    assert notebook["observation"]["license"] == "CC-BY-4.0"
    sources = notebook["observation"]["sources"]
    assert len(sources) == 4
    assert {p["programme"] for p in sources} == {"165.H-0588(A)", "167.D-0407(A)"}
    for product in notebook["observation"]["source_fits_headers"].values():
        assert "SPECSYS" in product["primary"]
        assert "em.wl;obs.atmos" in product["spectrum_table"]


def test_bundled_loader_rejects_changed_bytes_and_wrong_frame(notebook, tmp_path):
    execute("data", notebook)
    path = tmp_path / "spectrum.npz"
    path.write_bytes(notebook["DATA_FILE"].read_bytes())
    metadata = dict(notebook["observation"])
    path.with_suffix(".json").write_text(json.dumps(metadata))
    path.write_bytes(b"wrong bytes")
    with pytest.raises(RuntimeError, match="recorded checksum"):
        notebook["load_observation"](path)
    path.write_bytes(notebook["DATA_FILE"].read_bytes())
    metadata["wavelength_frame"] = "air, topocentric"
    path.with_suffix(".json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="vacuum, barycentric"):
        notebook["load_observation"](path)


def test_saved_run_is_complete_and_certified():
    counts = []
    for cell in CELLS:
        if cell["cell_type"] == "code":
            if cell["execution_count"] is not None:
                counts.append(cell["execution_count"])
            else:
                assert cell["id"] in {"observed-plot", "rv-plot"}
                assert cell["metadata"]["completion_execution"]["method"].startswith("scripted plot execution")
                assert any(o.get("data", {}).get("image/png") for o in cell["outputs"])
            assert all(o["output_type"] != "error" for o in cell["outputs"])
    assert counts == list(range(1, len(counts) + 1))
    assert "Certificate verified: True" in output_text("certificate")
    plot = next(c for c in CELLS if c.get("id") == "plot")
    assert any(o.get("data", {}).get("image/png") for o in plot["outputs"])
    for element in ("Ca", "Mg", "Fe"):
        assert f"{element}: log {element}/He = " in output_text("fit")


def test_starting_request_is_the_tested_pg1225_cold_start(notebook):
    config = notebook["start_config"]
    assert isinstance(config, DZConfig)
    with np.load(ROOT / "tests/data/spectral_regressions/dz-pg1225.npz") as saved:
        tested = json.loads(str(saved["config_json"]))
    for name in ("effective_temperature", "logg", "abundances",
                 "log_hydrogen_abundance", "quality"):
        assert getattr(config, name) == tested[name]
    assert notebook["RUN_FINAL_COLD_START"] is False
    assert set(notebook["FIT_ELEMENTS"]) <= set(config.abundances)


def test_fit_tools_use_fixed_structure_and_recover_an_injected_abundance(notebook):
    structure = object()
    calls = []

    def fake_compute_dz(config, wave, *, initial_atmosphere=None, relax_atmosphere=True):
        # One Gaussian Mg line whose depth grows with Mg/He, on a sloped continuum.
        calls.append((initial_atmosphere, relax_atmosphere))
        if not relax_atmosphere:
            warnings.warn("fixed synthesis: checkpoint request mismatch",
                          AtmosphereConvergenceWarning)
        depth = 0.5 * 10 ** (config.abundances["Mg"] + 7.27)
        flux = (1 + 1e-4 * (wave - 3840)) * (1 - depth * np.exp(-0.5 * ((wave - 3839.4) / 0.4) ** 2))

        class Result:
            spectrum = type("Spectrum", (), {"surface_flux_lambda": flux})

        return Result()

    notebook.update(compute_dz=fake_compute_dz, base=type("Base", (), {"atmosphere": structure}),
                    RESOLVING_POWER=19540.0)
    execute("fit_tools", notebook)
    assert calls == [(structure, False)]

    truth, rv = {**notebook["LITERATURE"], "Mg": -7.1}, 45.0
    wave = np.arange(3800.0, 3880.0, 0.03)
    rest, flux = notebook["element_spectra"]("Mg", truth)[0]
    model = notebook["observe"](rest, flux, rv, wave)
    notebook.update(obs_wave=wave, obs_flux=3.0 * model, obs_error=np.full_like(wave, 0.01))
    trials = np.round(np.arange(-7.4, -6.79, 0.1), 2)
    chi2 = [notebook["element_chi2"]("Mg", {**truth, "Mg": v}, rv)[0] for v in trials]
    assert trials[int(np.argmin(chi2))] == truth["Mg"]
    assert min(chi2) < 1e-6
    assert all(call == (structure, False) for call in calls)


def test_cached_observation_requires_the_pinned_checksum(data_tools, tmp_path):
    import hashlib

    payload = b"cached archive bytes"
    path = tmp_path / "test-product.fits"
    path.write_bytes(payload)
    data_tools.update(DATA_DIR=tmp_path, ESO_URL="https://example.invalid/{}",
                      download=lambda *args: pytest.fail("Cached FITS attempted network access"))
    assert data_tools["fetch"]("test-product", hashlib.sha256(payload).hexdigest()) == path
    with pytest.raises(RuntimeError, match="pinned checksum"):
        data_tools["fetch"]("test-product", "0" * 64)
    assert path.read_bytes() == payload


def test_corrupted_download_does_not_populate_the_cache(data_tools, tmp_path):
    data_tools.update(DATA_DIR=tmp_path, ESO_URL="https://example.invalid/{}",
                      download=lambda url, path: path.write_bytes(b"corrupted bytes"))
    with pytest.raises(RuntimeError, match="cache unchanged"):
        data_tools["fetch"]("test-product", "0" * 64)
    assert not (tmp_path / "test-product.fits").exists()
    assert (tmp_path / "test-product.part").read_bytes() == b"corrupted bytes"


def test_download_keeps_hostname_and_certificate_validation(data_tools, monkeypatch, tmp_path):
    import io

    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    data_tools["verified_tls_context"] = lambda: context
    calls = []

    def open_verified(url, *, context, timeout):
        assert context.check_hostname
        assert context.verify_mode == ssl.CERT_REQUIRED
        calls.append(url)
        return io.BytesIO(b"verified download")

    monkeypatch.setattr(data_tools["urllib"].request, "urlopen", open_verified)
    path = tmp_path / "product.part"
    data_tools["download"]("https://example.invalid/product", path)
    assert calls == ["https://example.invalid/product"]
    assert path.read_bytes() == b"verified download"


def test_missing_explicit_ca_path_has_a_clear_failure(data_tools, monkeypatch, tmp_path):
    monkeypatch.setenv("SSL_CERT_FILE", str(tmp_path / "missing-ca.pem"))
    with pytest.raises(RuntimeError, match="SSL_CERT_FILE points to a missing CA path"):
        data_tools["verified_tls_context"]()


def test_no_ca_store_fails_without_weakening_tls(data_tools, monkeypatch):
    from types import SimpleNamespace

    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.delenv("SSL_CERT_DIR", raising=False)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    monkeypatch.setattr(ssl, "create_default_context", lambda: context)
    monkeypatch.setattr(ssl, "get_default_verify_paths", lambda: SimpleNamespace(capath=None))
    monkeypatch.setattr(sys, "platform", "no-platform-bundle")
    monkeypatch.setitem(sys.modules, "certifi", None)
    with pytest.raises(RuntimeError, match="No CA trust store is available"):
        data_tools["verified_tls_context"]()
    assert context.check_hostname
    assert context.verify_mode == ssl.CERT_REQUIRED


def test_download_error_identifies_the_product_and_cache_destination(data_tools, monkeypatch, tmp_path):
    def failed(*args, **kwargs):
        raise OSError("certificate verify failed")

    monkeypatch.setattr(data_tools["urllib"].request, "urlopen", failed)
    data_tools["verified_tls_context"] = lambda: ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    path = tmp_path / "product.part"
    with pytest.raises(RuntimeError) as error:
        data_tools["download"]("https://example.invalid/product", path)
    assert "https://example.invalid/product" in str(error.value)
    assert str(path.with_suffix(".fits")) in str(error.value)


def test_abundance_scan_recovers_a_quadratic_and_refuses_an_edge(notebook):
    notebook.update(FIT_ELEMENTS=("Mg",), RV_KMS={"Mg": 49.0}, POLY_DEGREE=1,
                    WINDOWS={"Mg": [(3815, 3860)]})
    truth = notebook["LITERATURE"]["Mg"] + 0.13
    notebook["element_chi2"] = lambda e, a, rv: (100 + ((a[e] - truth) / 0.05) ** 2, 103)
    execute("fit", notebook)
    assert notebook["results"]["Mg"]["best"] == pytest.approx(truth)
    assert notebook["results"]["Mg"]["error"] == pytest.approx(0.05)
    notebook["element_chi2"] = lambda e, a, rv: (100 + (a[e] - truth - 2) ** 2, 103)
    with pytest.warns(RuntimeWarning, match="scan edge"):
        with pytest.raises(RuntimeError, match="scan boundary"):
            execute("fit", notebook)


def test_velocity_scan_refuses_a_boundary_minimum(notebook):
    notebook.update(FIT_ELEMENTS=("Mg",), element_spectra=lambda *args: [],
                    element_chi2=lambda e, a, rv, spectra: ((rv - 150) ** 2, 100))
    with pytest.raises(RuntimeError, match="velocity minimum is at the scan boundary"):
        execute("rv", notebook)


def test_failed_final_certificate_stops_before_replacing_the_structure(notebook):
    original = object()

    class FailedResult:
        atmosphere = type("Atmosphere", (), {"metadata": {
            "equilibrium_certificate": {"verified": False, "failures": ["local_energy"]}}})

    notebook.update(RUN_FINAL_COLD_START=True, fitted=dict(notebook["LITERATURE"]),
                    probe=np.array([3900., 4000.]), base=original,
                    compute_dz=lambda *args, **kwargs: FailedResult())
    with pytest.raises(RuntimeError, match="Fitted-composition cold start failed"):
        execute("final-cold-start", notebook)
    assert notebook["base"] is original
