"""Shared public interfaces for one-shot white-dwarf models.

The low-level :mod:`wd_spectra` modules intentionally expose individual EOS,
opacity, transfer, and atmosphere operations.  This module defines the small
stable surface used by the spectral-type presets: a data manifest, a quality
level, and a uniform result that can be saved without importing validation
scripts.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass, replace
import hashlib
from importlib.resources import files
import json
import os
from pathlib import Path
from typing import Any, Literal, Mapping
import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..atmosphere import Atmosphere
from ..eos import (
    hummer_mihalas_helium_lte,
    hummer_mihalas_hydrogen_helium_lte,
    hummer_mihalas_hydrogen_lte,
)
from ..spectrum import Spectrum
FloatArray = NDArray[np.float64]
Quality = Literal["quick", "standard", "production"]
AtmosphereComposition = Literal["hydrogen", "helium", "mixed"]
ConvergenceStatus = Literal[
    "converged", "spectrum-qualified", "unconverged", "unknown"
]

_MODEL_REQUEST_FINGERPRINT_SCHEMA = 2
_MODEL_PHYSICS_REVISION = "openwd-0.1.3-qmhd-undoubled-v4"
_MODEL_FAMILY_PHYSICS_REVISIONS = {"DAH": "dah-shared-solver-v2"}


class AtmosphereConvergenceWarning(RuntimeWarning):
    """A spectrum is being returned from an unverified atmosphere."""


@dataclass(frozen=True)
class NumericalResolution:
    """Numerical work budget associated with a public quality level."""

    n_depth: int
    n_continuum: int
    maximum_iterations: int
    n_angle: int


_RESOLUTION = {
    "quick": NumericalResolution(20, 80, 2, 2),
    "standard": NumericalResolution(40, 300, 120, 3),
    "production": NumericalResolution(80, 600, 500, 4),
}


def numerical_resolution(quality: Quality) -> NumericalResolution:
    """Return numerical settings without altering the requested physics."""

    try:
        return _RESOLUTION[quality]
    except KeyError as exc:  # pragma: no cover - protects untyped callers
        raise ValueError(f"unknown quality {quality!r}") from exc


@dataclass(frozen=True)
class ModelData:
    """Locations of the atomic/profile data used by model presets.

    ``root`` is the repository or installed data workspace.  Set the
    ``OPENWD_DATA`` environment variable when calling the package away
    from the source checkout.
    """

    root: Path

    @classmethod
    def default(cls, root: str | Path | None = None) -> "ModelData":
        if root is None:
            root = os.environ.get("OPENWD_DATA")
        if root is None:
            root = files("wd_spectra").joinpath("data/runtime")
        return cls(Path(root).expanduser().resolve())

    @property
    def cache(self) -> Path:
        bundled = self.root / "cache"
        return bundled if bundled.is_dir() else self.root / ".cache"

    @property
    def allard_lyman(self) -> Path:
        return self.root / "allard_data"

    @property
    def allard_tlusty205(self) -> Path:
        """Checksum-pinned fixed Allard tables distributed with TLUSTY205."""

        return self.cache / "allard-tlusty205"

    @property
    def helium_i_stark(self) -> Path:
        return self.cache / "helium-stark/Beauchamp25_LD.txt"

    @property
    def helium_ii_stark(self) -> Path:
        return self.cache / "helium-stark/he2prf.dat"

    @property
    def ccc_hydrogen_collisions(self) -> Path:
        return self.cache / "ccc/e-H_XSEC_LS.zip"

    @property
    def tlusty_source(self) -> Path:
        return self.cache / "tlusty-source/tlusty200.f"

    @property
    def tlusty_helium_atom(self) -> Path:
        return self.cache / "tlusty-atoms/he1_14lev.dat"

    @property
    def stout(self) -> Path:
        return self.cache / "metal-opacity/atomic-line-list"

    @property
    def verner_photoionization(self) -> Path:
        return self.cache / "metal-opacity/verner-photoionization.dat"

    @property
    def barklem_neutral_h_broadening(self) -> Path:
        """ABO line-by-line neutral-H widths for metal transitions."""

        return self.cache / "metal-opacity/barklem-neutral-h-broadening.dat"

    @property
    def mg_he_red_wing(self) -> Path:
        return self.cache / "metal-opacity/mg-he-2852-red-wing.dat"

    @property
    def mg_i_he_density_profiles(self) -> Path:
        return (
            self.cache
            / "metal-opacity/unified-metal-he/mgi-2852-density-6000K.dat"
        )

    @property
    def ca_i_he_density_profiles(self) -> Path:
        return (
            self.cache
            / "metal-opacity/unified-metal-he/cai-4227-density-4000K.dat"
        )

    @property
    def ca_i_he_temperature_profiles(self) -> Path:
        return (
            self.cache
            / "metal-opacity/unified-metal-he/cai-4227-temperature-nhe5e21.dat"
        )

    @property
    def ca_ii_chianti_collisions(self) -> Path:
        """CHIANTI Ca II effective electron collision strengths."""

        return self.cache / "chianti/ca_2/ca_2.scups"

    @property
    def helium_reos3(self) -> Path:
        """Becker et al. (2014) ab-initio helium EOS table."""

        return self.cache / "dense-eos/he-reos3-table3.dat"

    @property
    def nist_asd_strong(self) -> Path:
        """Checksum-pinned NIST ASD strong-line query directory."""

        return self.cache / "nist-asd-strong"

    @property
    def sirocco_atomic(self) -> Path:
        """Checksum-pinned SIROCCO/TOPbase level and photoionization files."""

        return self.cache / "sirocco-atomic"

    @property
    def tlusty_atoms(self) -> Path:
        """Checksum-pinned public TLUSTY/Opacity Project model atoms."""

        return self.cache / "tlusty-atoms"

    @property
    def h2db_balmer_subset(self) -> Path:
        """Schimeczek--Wunner H2db Balmer transitions and state energies."""

        return self.cache / "h2db/h2db_balmer_subset.npz"

    @property
    def h2_h2_cia(self) -> Path:
        return (
            self.cache
            / "molecular-opacity/CIA_Borysow_H2H2_0060-7000K_0.6-500um.dat"
        )

    def require(self, *paths: Path, fetch_command: str | None = None) -> None:
        missing = [path for path in paths if not path.is_file() and not path.is_dir()]
        if not missing:
            return
        detail = "\n".join(f"  - {path}" for path in missing)
        hint = (
            "\nReinstall OpenWD to restore the bundled data, or point "
            "OPENWD_DATA/ModelData at a complete external data directory."
        )
        raise FileNotFoundError(f"required model data are missing:\n{detail}{hint}")


@dataclass(frozen=True)
class ModelResult:
    """Atmosphere, rest-frame spectrum, and provenance from one calculation."""

    spectral_type: str
    atmosphere: Atmosphere
    spectrum: Spectrum
    config: object
    metadata: Mapping[str, Any]
    population_state: object | None = None


def model_request_fingerprint(
    spectral_type: str,
    config: object,
    data: ModelData,
    *, physical_data_identity: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Return a stable identity for one solver-relevant public request.

    Code and table contents are part of the identity. Older path-only
    fingerprints degrade to a warm start and cannot certify fixed synthesis.
    """

    from .._provenance import model_data_identity, numerical_code_identity

    request = {
        "schema": _MODEL_REQUEST_FINGERPRINT_SCHEMA,
        "physics_revision": _MODEL_PHYSICS_REVISION + (
            ":" + _MODEL_FAMILY_PHYSICS_REVISIONS[str(spectral_type)]
            if str(spectral_type) in _MODEL_FAMILY_PHYSICS_REVISIONS else ""
        ),
        "spectral_type": str(spectral_type),
        "config": _jsonable(config),
        "data_root": str(data.root.resolve()),
        "data_sha256": model_data_identity(data),
        "code_sha256": numerical_code_identity(),
    }
    if physical_data_identity is not None:
        request["physical_data_identity"] = _jsonable(physical_data_identity)
    serialized = json.dumps(
        request,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return {
        **request,
        "sha256": hashlib.sha256(serialized).hexdigest(),
    }


def atmosphere_matches_model_request(
    atmosphere: Atmosphere | None,
    fingerprint: Mapping[str, object],
) -> bool:
    """Return whether an atmosphere proves an exact request identity."""

    if atmosphere is None:
        return False
    config = fingerprint.get("config", {})
    if not isinstance(config, Mapping) or (
        atmosphere.effective_temperature != config.get("effective_temperature")
        or atmosphere.logg != config.get("logg")
    ):
        return False
    recorded = atmosphere.metadata.get("model_request_fingerprint")
    if not isinstance(recorded, Mapping):
        return False
    return dict(recorded) == dict(fingerprint)


def fixed_synthesis_atmosphere(atmosphere: Atmosphere,
                              fingerprint: Mapping[str, object]) -> Atmosphere:
    """Retain an explicit exploratory structure, but not a mismatched claim.

    The original atmosphere diagnostics still describe the original physical
    problem. They must not certify a new composition, opacity policy or Teff.
    Never mutate the caller's structure or silently re-relax it here.
    """
    matches = atmosphere_matches_model_request(atmosphere, fingerprint)
    return replace(atmosphere, metadata={
        **atmosphere.metadata,
        "fixed_synthesis_request_verified": matches,
        "fixed_synthesis_requested_fingerprint": dict(fingerprint),
    })


def atmosphere_with_model_request_fingerprint(
    atmosphere: Atmosphere,
    fingerprint: Mapping[str, object],
) -> Atmosphere:
    """Attach immutable request provenance without mutating caller metadata."""
    metadata = dict(atmosphere.metadata)
    # A newly relaxed atmosphere is no longer a fixed-synthesis request.
    metadata.pop("fixed_synthesis_request_verified", None)
    metadata.pop("fixed_synthesis_requested_fingerprint", None)
    return replace(
        atmosphere,
        metadata={
            **metadata,
            "model_request_fingerprint": dict(fingerprint),
        },
    )


def atmosphere_convergence_status(
    atmosphere: Atmosphere,
) -> ConvergenceStatus:
    """Classify whether a structure records successful self-consistency."""

    if atmosphere.metadata.get(
        "mean_3d_temperature_differential_is_equilibrium_model"
    ) is False:
        return "unconverged"
    if atmosphere.metadata.get("fixed_synthesis_request_verified") is False:
        return "unconverged"
    from .._convergence import recorded_equilibrium_status
    return recorded_equilibrium_status(atmosphere.metadata)


def warn_if_atmosphere_not_converged(
    atmosphere: Atmosphere,
    spectral_type: str,
) -> ConvergenceStatus:
    """Warn, but do not prevent exploratory synthesis from a partial model."""

    status = atmosphere_convergence_status(atmosphere)
    if status == "converged":
        return status
    if status == "spectrum-qualified":
        detail = (
            "passes the declared spectrum-qualification checks but does not "
            "record full equilibrium convergence"
        )
    elif status == "unconverged":
        detail = "records that radiative/convective equilibrium did not converge"
    else:
        detail = "does not record a verified atmosphere-convergence status"
    if atmosphere.metadata.get("fixed_synthesis_request_verified") is False:
        detail = "does not verify equilibrium for the requested parameters and physics (checkpoint request mismatch)"
    elif (
        status != "spectrum-qualified"
        and atmosphere.metadata.get("radiative_equilibrium_solver_converged") is True
    ):
        detail = "passes the solver flux checks but does not pass all equilibrium certification checks"
    metrics = []
    certificate = atmosphere.metadata.get("equilibrium_certificate", {})
    if isinstance(certificate, Mapping) and certificate.get("failures"):
        metrics.append("unverified checks=" + ", ".join(certificate["failures"]))
    residual = atmosphere.metadata.get(
        "maximum_all_depth_total_flux_residual",
        atmosphere.metadata.get("maximum_total_flux_residual"),
    )
    if isinstance(
        residual, (int, float, np.integer, np.floating)
    ) and np.isfinite(residual):
        metrics.append(f"maximum all-depth flux residual={float(residual):.3e}")
    correction = atmosphere.metadata.get(
        "radiative_equilibrium_maximum_log_temperature_correction"
    )
    if isinstance(
        correction, (int, float, np.integer, np.floating)
    ) and np.isfinite(correction):
        metrics.append(
            "maximum log-temperature correction="
            f"{float(correction):.3e}"
        )
    terminal_reason = atmosphere.metadata.get(
        "nonlinear_solver_terminal_reason"
    )
    if isinstance(terminal_reason, str):
        metrics.append(f"nonlinear terminal reason={terminal_reason}")
    metric_text = f" ({', '.join(metrics)})" if metrics else ""
    warnings.warn(
        f"Returning a {spectral_type} spectrum from an atmosphere that {detail}"
        f"{metric_text}. The result is suitable for explicit exploratory work, "
        "but must not be treated as a converged model.",
        AtmosphereConvergenceWarning,
        stacklevel=3,
    )
    return status


def default_wavelength_grid(
    lower: float = 900.0,
    upper: float = 30_000.0,
    *,
    continuum_points: int = 2500,
    optical_step: float = 0.25,
) -> FloatArray:
    """Broad UV--IR grid with a resolved optical line region."""

    if not 0.0 < lower < upper:
        raise ValueError("wavelength bounds must be positive and increasing")
    broad = np.geomspace(lower, upper, continuum_points)
    optical_lower = max(lower, 3400.0)
    optical_upper = min(upper, 7500.0)
    optical = (
        np.arange(optical_lower, optical_upper + 0.5 * optical_step, optical_step)
        if optical_lower < optical_upper
        else np.empty(0)
    )
    return np.unique(np.r_[broad, optical])


def validate_wavelength(wavelength: ArrayLike | None) -> FloatArray:
    result = default_wavelength_grid() if wavelength is None else np.asarray(
        wavelength, dtype=np.float64
    )
    if result.ndim != 1 or result.size < 2:
        raise ValueError("wavelength must be a one-dimensional array")
    if np.any(~np.isfinite(result)) or np.any(result <= 0.0):
        raise ValueError("wavelength must contain finite positive values")
    if np.any(np.diff(result) <= 0.0):
        raise ValueError("wavelength must increase strictly")
    return np.ascontiguousarray(result)


def load_atmosphere_checkpoint(
    path: str | Path,
    effective_temperature: float,
    logg: float,
    composition: AtmosphereComposition,
    *,
    log_hydrogen_to_helium: float | None = None,
    include_molecules: bool = False,
    include_negative_hydrogen: bool = False,
    trihydrogen_ion_partition_model: str | None = None,
) -> Atmosphere:
    """Restore a validation/one-shot atmosphere from the common NPZ schemas.

    The historical validation files differ mainly in whether their thermal
    arrays carry a ``thermal_`` prefix.  EOS populations are always rebuilt
    from the requested current composition rather than trusted from a stale
    checkpoint.

    For mixed composition, ``include_molecules=True`` selects the complete
    molecular-H/He closure (including H- and Neale--Tennyson H3+). The separate
    negative-H and H3+ options above apply to the pure-hydrogen closure only.
    Changing mixed chemistry invalidates a stored convergence claim.
    """

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"atmosphere checkpoint does not exist: {source}")
    stored_metadata: dict[str, object] = {}
    with np.load(source) as saved:
        saved_teff = float(saved["effective_temperature"]) if "effective_temperature" in saved else None
        saved_logg = float(saved["logg"]) if "logg" in saved else None
        def required(*names: str) -> FloatArray:
            for name in names:
                if name in saved:
                    return np.asarray(saved[name], dtype=np.float64)
            raise KeyError(
                f"{source} has none of the required arrays {', '.join(names)}"
            )

        temperature = required("thermal_temperature", "temperature", "T")
        pressure = required("thermal_gas_pressure", "gas_pressure", "P")
        column_mass = required("column_mass", "m")
        if "rosseland_optical_depth" in saved:
            optical_depth = np.asarray(
                saved["rosseland_optical_depth"], dtype=np.float64
            )
        elif "tau" in saved:
            optical_depth = np.asarray(saved["tau"], dtype=np.float64)
        else:
            optical_depth = np.geomspace(1.0e-8, 1.0e2, temperature.size)
        saved_mass_density = (
            np.asarray(saved["thermal_mass_density"], dtype=np.float64)
            if "thermal_mass_density" in saved
            else np.asarray(saved["mass_density"], dtype=np.float64)
            if "mass_density" in saved else None
        )
        saved_electron_density = (
            np.asarray(saved["thermal_electron_density"], dtype=np.float64)
            if "thermal_electron_density" in saved
            else np.asarray(saved["electron_density"], dtype=np.float64)
            if "electron_density" in saved else None
        )
        if "atmosphere_metadata_json" in saved:
            try:
                serialized_metadata = str(
                    np.asarray(saved["atmosphere_metadata_json"]).item()
                )
                parsed_metadata = json.loads(serialized_metadata)
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"{source} contains invalid atmosphere metadata"
                ) from exc
            if not isinstance(parsed_metadata, dict):
                raise ValueError(
                    f"{source} atmosphere metadata must be a JSON object"
                )
            stored_metadata = parsed_metadata
    metadata = {
        **stored_metadata,
        "model": "restored one-shot atmosphere",
        "source_checkpoint": str(source.resolve()),
        "checkpoint_composition": composition,
    }
    def reclosed_metadata(state):
        """Changing the EOS invalidates certification even at identical T/P."""
        record = dict(metadata)
        for name, old in (("mass_density", saved_mass_density),
                          ("electron_density", saved_electron_density)):
            new = np.asarray(getattr(state, name), dtype=float)
            if old is None or old.shape != new.shape or not np.allclose(old, new, rtol=1e-8, atol=0.):
                record["radiative_equilibrium_converged"] = False
                record.pop("equilibrium_certificate", None)
                record.pop("model_request_fingerprint", None)
                record["checkpoint_eos_changed_requires_relaxation"] = True
        return record
    request = stored_metadata.get("model_request_fingerprint", {})
    old_config = request.get("config", {}) if isinstance(request, Mapping) else {}
    if not isinstance(old_config, Mapping):
        old_config = {}
    old_kind = request.get("spectral_type") if isinstance(request, Mapping) else None
    old_composition = ({"DA": "hydrogen", "DB": "helium", "DZ": "helium", "DAB": "mixed"}.get(old_kind)
                       or stored_metadata.get("checkpoint_composition"))
    changed = []
    if saved_teff != effective_temperature:
        changed.append("effective_temperature changed or unrecorded")
    if saved_logg != logg:
        changed.append("logg changed or unrecorded")
    if old_composition is not None and old_composition != composition:
        changed.append("composition changed")
    if composition == "mixed":
        old_abundance = old_config.get("log_hydrogen_to_helium",
            stored_metadata.get("log_hydrogen_to_helium"))
        if old_abundance != log_hydrogen_to_helium:
            changed.append("H/He abundance changed or unrecorded")
    if composition == "hydrogen":
        old_molecules = old_config.get("include_molecules")
        if old_molecules is None:
            old_molecules = stored_metadata.get("includes_molecular_equilibrium", False)
        if bool(old_molecules) != include_molecules:
            changed.append("molecular chemistry changed")
        old_negative = stored_metadata.get(
            "radiative_equilibrium_includes_negative_hydrogen_charge_equilibrium",
            stored_metadata.get("negative_hydrogen_in_charge_equilibrium"))
        if old_negative is None or bool(old_negative) != include_negative_hydrogen:
            changed.append("H- charge equilibrium changed or unrecorded")
        old_partition = old_config.get("h3plus_partition_model",
            stored_metadata.get("trihydrogen_ion_partition_model"))
        if old_partition == "none":
            old_partition = None
        if old_partition != trihydrogen_ion_partition_model:
            changed.append("H3+ partition changed or unrecorded")
    if changed:
        metadata["radiative_equilibrium_converged"] = False
        metadata.pop("model_request_fingerprint", None)
        metadata.pop("equilibrium_certificate", None)
        metadata["checkpoint_changed_requires_relaxation"] = changed
    if stored_metadata.get("experimental_h2_partition"):
        # This loader rebuilds populations using the requested production EOS;
        # it cannot certify an external experiment's partition function or
        # thermodynamic derivatives. Keep the structure for explicit re-solving,
        # but never inherit that experiment's convergence claim.
        metadata["radiative_equilibrium_converged"] = False
        metadata.pop("model_request_fingerprint", None)
        metadata.pop("equilibrium_certificate", None)
        metadata["checkpoint_chemistry_changed_requires_relaxation"] = True
    zeros = np.zeros_like(temperature)
    if composition == "hydrogen":
        state = hummer_mihalas_hydrogen_lte(
            temperature,
            pressure,
            correlated_microfields=True,
            include_molecules=include_molecules,
            include_negative_hydrogen=include_negative_hydrogen,
            trihydrogen_ion_partition_model=(
                trihydrogen_ion_partition_model
            ),
        )
        return Atmosphere(
            effective_temperature,
            logg,
            optical_depth,
            column_mass,
            temperature,
            pressure,
            state.mass_density,
            state.neutral_h_density,
            state.proton_density,
            state.electron_density,
            reclosed_metadata(state),
            hydrogen_lte_state=state,
        )
    if composition == "helium":
        state = hummer_mihalas_helium_lte(
            temperature,
            pressure,
            neutral_radius_scale=0.5,
            correlated_microfields=True,
        )
        return Atmosphere(
            effective_temperature,
            logg,
            optical_depth,
            column_mass,
            temperature,
            pressure,
            state.mass_density,
            zeros,
            zeros,
            state.electron_density,
            reclosed_metadata(state),
            helium_lte_state=state,
        )
    if composition == "mixed":
        if log_hydrogen_to_helium is None:
            raise ValueError("mixed checkpoints require log_hydrogen_to_helium")
        function = hummer_mihalas_hydrogen_helium_lte
        if include_molecules:
            from .._mixed_molecules import molecular_hydrogen_helium_lte
            function = molecular_hydrogen_helium_lte
        if bool(stored_metadata.get("includes_molecular_equilibrium",False)) != include_molecules:
            metadata["radiative_equilibrium_converged"] = False
            metadata.pop("model_request_fingerprint",None)
            metadata.pop("equilibrium_certificate",None)
            metadata["checkpoint_chemistry_changed_requires_relaxation"] = True
        metadata["includes_molecular_equilibrium"] = include_molecules
        metadata["mixed_chemical_model"] = ("molecular-h-he-hm" if include_molecules else "atomic-h-he-hm")
        state = function(
            temperature,
            pressure,
            log_hydrogen_to_helium,
            hydrogen_neutral_radius_scale=0.5,
            helium_neutral_radius_scale=0.5,
            correlated_microfields=True,
        )
        hydrogen = state.hydrogen_lte_state
        return Atmosphere(
            effective_temperature,
            logg,
            optical_depth,
            column_mass,
            temperature,
            pressure,
            state.mass_density,
            hydrogen.neutral_h_density,
            hydrogen.proton_density,
            state.electron_density,
            reclosed_metadata(state),
            hydrogen_lte_state=hydrogen,
            helium_lte_state=state.helium_lte_state,
        )
    raise ValueError(f"unknown atmosphere composition {composition!r}")


def _jsonable(value: object) -> object:
    if is_dataclass(value):
        return {
            item.name: _jsonable(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def save_model_result(result: ModelResult, output: str | Path) -> Path:
    """Write the common one-shot artifact set and return its directory."""

    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    spectrum = result.spectrum
    atmosphere = result.atmosphere
    np.savetxt(
        directory / "spectrum.txt",
        np.column_stack((spectrum.wavelength_angstrom, spectrum.surface_flux_lambda)),
        header="vacuum_wavelength_angstrom surface_flux_erg_s-1_cm-2_A-1",
    )
    np.savez_compressed(
        directory / "atmosphere.npz",
        effective_temperature=atmosphere.effective_temperature,
        logg=atmosphere.logg,
        rosseland_optical_depth=atmosphere.rosseland_optical_depth,
        column_mass=atmosphere.column_mass,
        temperature=atmosphere.temperature,
        gas_pressure=atmosphere.gas_pressure,
        mass_density=atmosphere.mass_density,
        electron_density=atmosphere.electron_density,
        atmosphere_metadata_json=np.asarray(
            json.dumps(
                _jsonable(atmosphere.metadata),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        ),
    )
    if result.population_state is not None:
        # Keep nested atom states without pickle; this is diagnostic output,
        # not an implicit restart input to the public cold-start API.
        arrays = {}
        def collect_population(value, prefix=""):
            if is_dataclass(value):
                return {item.name: collect_population(getattr(value, item.name), prefix + item.name + ".")
                        for item in fields(value)}
            if isinstance(value, Mapping):
                return {str(key): collect_population(item, prefix + str(key) + ".")
                        for key, item in value.items()}
            if isinstance(value, (tuple, list)):
                return [collect_population(item, prefix + str(index) + ".")
                        for index, item in enumerate(value)]
            if isinstance(value, np.ndarray):
                arrays[prefix[:-1]] = value
                return {"array": prefix[:-1], "shape": list(value.shape)}
            if isinstance(value, (float, np.floating)) and not np.isfinite(value):
                # An unmeasured population defect is absent, not zero. Keep
                # fixed-state diagnostic outputs valid strict JSON as well.
                return None
            return _jsonable(value)
        population_metadata = collect_population(result.population_state)
        arrays["metadata_json"] = np.asarray(json.dumps(
            population_metadata, sort_keys=True, allow_nan=False))
        np.savez_compressed(directory / "populations.npz", **arrays)
    record = {
        "schema": 1,
        "spectral_type": result.spectral_type,
        "config": _jsonable(result.config),
        "model_metadata": _jsonable(result.metadata),
        "atmosphere_metadata": _jsonable(atmosphere.metadata),
        "spectrum_metadata": _jsonable(spectrum.metadata),
    }
    (directory / "metadata.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    return directory
