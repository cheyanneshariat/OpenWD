"""The reader, not the generic table's constructor default, chooses CIA tails."""
import numpy as np
import pytest

from wd_spectra.molecules import read_borysow_h2_h2_cia_table


def test_borysow_loader_continues_declining_optical_edge(tmp_path):
    # The 5000-K end of the production table; the last value is not rising.
    path = tmp_path / "edge.dat"
    path.write_text(
        "@TEMPERATURES\n4000 5000\n@DATA\n"
        "16460 1e-10 6.523e-10\n16480 0.9e-10 6.402e-10\n"
    )
    table = read_borysow_h2_h2_cia_table(path)
    assert table.extrapolate_high_wavenumber_tail
    wavenumber = np.array([16479.99, 16480., 16480.01, 16500.])
    actual = table.coefficient_for_wavelength_temperature(
        1e8 / wavenumber, [5000.]
    )[:, 0]
    expected = 6.402e-10 * (6.402 / 6.523)**((wavenumber - 16480) / 20)
    np.testing.assert_allclose(actual, expected, rtol=3e-14)
    assert np.all(actual > 0)
    assert np.all(np.diff(actual) < 0)


@pytest.mark.parametrize("terminal", [1e-10, 1.1e-10])
def test_borysow_loader_does_not_extrapolate_flat_or_rising_tail(tmp_path, terminal):
    path = tmp_path / "edge.dat"
    path.write_text(
        "@TEMPERATURES\n4000 5000\n@DATA\n"
        f"16460 1e-10 6.523e-10\n16480 {terminal} 6.402e-10\n"
    )
    table = read_borysow_h2_h2_cia_table(path)
    assert not table.extrapolate_high_wavenumber_tail
    np.testing.assert_array_equal(
        table.coefficient_for_wavelength_temperature([6000.], [4000., 5000.]),
        0.,
    )
