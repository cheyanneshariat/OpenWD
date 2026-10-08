"""Compiled Gaunt interpolation against the retained numerical reference."""
import numpy as np
import pytest
from wd_spectra import gaunt


def native():
    if gaunt._rt is None or not hasattr(gaunt._rt, "cubic_table_interpolate"):
        pytest.skip("compiled cubic interpolation is unavailable")
    return gaunt._rt


@pytest.mark.parametrize("charge", [1.0, 2.0, 6.0, 8.0, 26.0])
def test_compiled_gaunt_matches_reference_over_broadcast_grid(monkeypatch, charge):
    compiled = native()
    wavelength = np.geomspace(0.01, 1e10, 301)[:, None]
    temperature = np.geomspace(1e3, 1e8, 47)[None, :]
    actual = gaunt.hydrogen_free_free_gaunt_factor(
        wavelength, temperature, ionic_charge=charge
    )
    monkeypatch.setattr(gaunt, "_rt", None)
    expected = gaunt.hydrogen_free_free_gaunt_factor(
        wavelength, temperature, ionic_charge=charge
    )
    np.testing.assert_allclose(actual, expected, rtol=3e-14, atol=1e-14)
    monkeypatch.setattr(gaunt, "_rt", compiled)
    assert gaunt.hydrogen_free_free_gaunt_factor(5000.0, 10000.0).shape == ()
    assert gaunt.hydrogen_free_free_gaunt_factor(np.empty(0), 10000.0).shape == (0,)


def test_compiled_cubic_interpolation_matches_cell_edges_and_clipping():
    compiled = native()
    table, xmin, ymin, step = gaunt._read_van_hoof_table()
    # Include every cell boundary, its immediate neighbours and outside-domain values.
    x = xmin + step * np.arange(table.shape[1])
    y = ymin + step * np.arange(table.shape[0])
    x = np.r_[-np.inf, x - 1e-12, x, x + 1e-12, np.inf]
    y = np.r_[-np.inf, y - 1e-12, y, y + 1e-12, np.inf]
    xx, yy = np.meshgrid(x, y)
    xx, yy = xx.ravel(), yy.ravel()
    xi, xw = gaunt._cubic_grid_indices_and_weights(xx, xmin, step, table.shape[1])
    yi, yw = gaunt._cubic_grid_indices_and_weights(yy, ymin, step, table.shape[0])
    expected = np.zeros_like(xx)
    for j in range(4):
        for i in range(4):
            expected += yw[:, j] * xw[:, i] * table[yi[:, j], xi[:, i]]
    actual = np.empty_like(xx)
    compiled.cubic_table_interpolate(table, xx, yy, xmin, ymin, step, actual)
    np.testing.assert_allclose(actual, expected, rtol=3e-14, atol=1e-14)


@pytest.mark.parametrize("case", ["shape", "dtype", "readonly", "nan", "small_table"])
def test_compiled_interpolation_rejects_invalid_buffers(case):
    compiled = native()
    table = np.ones((4, 4))
    x = np.ones(3)
    y = np.ones(3)
    out = np.empty(3)
    if case == "shape":
        out = np.empty(2)
    elif case == "dtype":
        x = x.astype(np.float32)
    elif case == "readonly":
        out.flags.writeable = False
    elif case == "nan":
        x[0] = np.nan
    elif case == "small_table":
        table = np.ones((3, 4))
    with pytest.raises((ValueError, TypeError, BufferError)):
        compiled.cubic_table_interpolate(table, x, y, 0.0, 0.0, 1.0, out)


@pytest.mark.parametrize(
    "minimum,step", [(1e308, 1e308), (1e308, 1.0), (-1e308, 6e307)]
)
def test_compiled_interpolation_rejects_unrepresentable_grid(minimum, step):
    compiled = native()
    with pytest.raises(ValueError, match="finite, distinct nodes"):
        compiled.cubic_table_interpolate(
            np.ones((4, 4)), np.zeros(1), np.zeros(1), minimum, 0.0, step, np.empty(1)
        )
