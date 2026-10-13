"""The historical-control patch must be local, reversible and EOS preserving."""
from pathlib import Path
import sys
from types import MappingProxyType, ModuleType, SimpleNamespace

import numpy as np
import pytest

import wd_spectra.metals as metals
from wd_spectra.metals import MetalLTEState

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))
from uv_atomic_control_legacy import legacy_uv_atomic_controls


@pytest.mark.parametrize("reference", ["He", "H"])
def test_legacy_scope_changes_only_helium_line_opacity_state(reference, monkeypatch):
    forwarded = []

    def opacity(atmosphere, wavelength_angstrom, atomic_database, metal_state, **kwargs):
        forwarded.append(metal_state)
        return "opacity"

    def augment(database):
        raise AssertionError("Legacy controls must disable only Fe augmentation")

    monkeypatch.setattr(metals, "metal_line_mass_absorption_coefficient", opacity)
    monkeypatch.setattr(metals, "augment_iron_ii_kurucz_uv_transitions", augment)
    alias = ModuleType("wd_spectra._legacy_control_test_alias")
    alias.opacity = opacity
    alias.augment = augment
    monkeypatch.setitem(sys.modules, alias.__name__, alias)
    state = MetalLTEState(
        reference_species=reference,
        log_number_abundance=MappingProxyType({"Fe": -6.47}),
        element_number_density=MappingProxyType({"Fe": np.asarray([1., 2.])}),
        ion_number_density=MappingProxyType({"Fe": np.ones((3, 2))}),
        partition_function=MappingProxyType({("Fe", 1): np.asarray([3., 4.])}),
        electron_density=np.asarray([5., 6.]),
        metal_electron_density=np.asarray([1., 2.]),
        trace_hydrogen_state=SimpleNamespace(neutral_h_density=np.asarray([7., 8.])),
    )
    database = object()
    with pytest.raises(RuntimeError, match="restore"):
        with legacy_uv_atomic_controls():
            assert alias.augment(database) is database
            assert alias.opacity(None, [1200.], database, metal_state=state) == "opacity"
            (passed,) = forwarded
            if reference == "He":
                assert passed is not state
                assert passed.trace_hydrogen_state is None
            else:
                assert passed is state
            for name in ("log_number_abundance", "element_number_density",
                         "ion_number_density", "partition_function",
                         "electron_density", "metal_electron_density"):
                assert getattr(passed, name) is getattr(state, name)
            assert state.trace_hydrogen_state is not None
            raise RuntimeError("restore")
    assert metals.metal_line_mass_absorption_coefficient is opacity
    assert metals.augment_iron_ii_kurucz_uv_transitions is augment
    assert alias.opacity is opacity and alias.augment is augment


def test_legacy_scope_restores_aliases_created_after_entry(monkeypatch):
    opacity = metals.metal_line_mass_absorption_coefficient
    augment = metals.augment_iron_ii_kurucz_uv_transitions
    late_alias = ModuleType("wd_spectra._legacy_late_alias")
    with pytest.raises(RuntimeError, match="restore late aliases"):
        with legacy_uv_atomic_controls():
            late_alias.opacity = metals.metal_line_mass_absorption_coefficient
            late_alias.augment = metals.augment_iron_ii_kurucz_uv_transitions
            monkeypatch.setitem(sys.modules, late_alias.__name__, late_alias)
            assert late_alias.opacity is not opacity
            assert late_alias.augment is not augment
            raise RuntimeError("restore late aliases")
    assert late_alias.opacity is opacity
    assert late_alias.augment is augment
