"""Checksum-pinned, independently corroborated additions to Stout."""
from dataclasses import replace
from functools import lru_cache
import hashlib
from importlib.resources import files
from io import BytesIO
import json
from types import MappingProxyType

import numpy as np

from .constants import BOLTZMANN, ELECTRON_MASS, ELEMENTARY_CHARGE_ESU, LIGHT_SPEED, PI, PLANCK

SOURCE = "independently corroborated Kurucz supplements v1"
MANIFEST_SHA256 = "f0efa3b6012e25c4888722a154455e519e9a4857575e583e70004739b112226e"


@lru_cache(maxsize=1)
def manifest():
    payload = files("wd_spectra").joinpath(
        "data/atomic/independent-kurucz/manifest.json").read_bytes()
    if hashlib.sha256(payload).hexdigest() != MANIFEST_SHA256:
        raise ValueError("Independent Kurucz manifest checksum mismatch")
    return json.loads(payload)


@lru_cache(maxsize=7)
def catalogue(key):
    item = manifest()["ions"][key]
    payload = files("wd_spectra").joinpath(
        f"data/atomic/independent-kurucz/{key}.npz").read_bytes()
    if hashlib.sha256(payload).hexdigest() != item["sha256"]:
        raise ValueError(f"Independent Kurucz catalogue checksum mismatch: {key}")
    with np.load(BytesIO(payload), allow_pickle=False) as saved:
        records, levels = saved["records"].copy(), saved["levels"].copy()
    records.setflags(write=False)
    levels.setflags(write=False)
    return records, levels


def augment(database):
    """Preserve Stout records; add the prospective qualified set once.

    Only the two level extensions already present in the previous defaults
    are retained. Lower-strength qualified channels supply natural widths,
    without instantiating additional absorption lines below the fixed floor.
    """
    from .metals import AtomicDatabase, AtomicLevel, AtomicTransition

    ions = dict(database.ions)
    changed = False
    floor = manifest()["absorption_f_floor"]
    for ion_key, original in database.ions.items():
        key = f"{ion_key[0]}_{ion_key[1]}"
        if key not in manifest()["ions"] or SOURCE in original.source:
            continue
        rows, expected = catalogue(key)
        item = manifest()["ions"][key]
        levels = list(original.levels)

        def match(energy, weight, expected_index=None):
            # The pinned Stout atom has known level identities; nearly
            # degenerate neighbours must not make that identity ambiguous.
            exact = [l.index for l in levels if l.index == expected_index
                     and l.energy_wavenumber == energy and l.statistical_weight == weight]
            if exact:
                return exact[0]
            found = [l.index for l in levels
                     if abs(l.energy_wavenumber - energy) <= 0.1
                     and l.statistical_weight == weight]
            return found[0] if len(found) == 1 else None

        extension_ids = {l["index"] for l in item["legacy_level_extensions"]}
        # A custom/subset Stout atom must not acquire unrelated new levels.
        complete_base = all(match(e, g, int(i)) is not None for i, e, g in expected
                            if int(i) not in extension_ids)
        if complete_base:
            for record in item["legacy_level_extensions"]:
                if match(record["energy_wavenumber"], record["statistical_weight"], record["index"]) is None:
                    index = record["index"]
                    if any(l.index == index for l in levels):
                        index = max(l.index for l in levels) + 1
                    levels.append(AtomicLevel(index, record["energy_wavenumber"],
                        record["statistical_weight"], record["label"]))
        indices = {int(i): match(e, g, int(i)) for i, e, g in expected}
        existing = {(t.lower_index, t.upper_index) for t in original.transitions}
        added, weak = [], {}
        for lower, upper, wavelength, f, a, rad, stark, vdw, _ in rows:
            lo, up = indices[int(lower)], indices[int(upper)]
            if lo is None or up is None or (lo, up) in existing:
                continue
            if f < floor:
                weak[up] = weak.get(up, 0.0) + float(a)
                continue
            added.append(AtomicTransition(lo, up, float(a), "E1", float(wavelength), float(f),
                radiative_damping_rate_s=None if np.isnan(rad) else float(rad),
                electron_stark_rate_coefficient_cm3_s=None if np.isnan(stark) else float(stark),
                neutral_h_vdw_rate_coefficient_cm3_s=None if np.isnan(vdw) else float(vdw),
                independently_validated_supplement=True))
        ions[ion_key] = replace(original, levels=tuple(levels),
            transitions=original.transitions + tuple(added),
            supplemental_downward_rates=original.supplemental_downward_rates + tuple(weak.items()),
            source=f"{original.source}; {SOURCE} ({len(added)} absorption lines)")
        changed = True
    if not changed:
        return database
    return AtomicDatabase(MappingProxyType(ions), source=f"{database.source}; {SOURCE}")


def screen(atmosphere, wavelength, state, selected, background):
    """Conservative all-depth LTE Doppler-peak screen, additions only.

    A normalized Voigt profile cannot exceed its thermal Gaussian peak.
    Protect explicit unified profiles and strong ground-state resonances.
    Callers with explicit line keys or NLTE departure coefficients bypass
    this screen. The production ranking and total line budget are unchanged.
    """
    background = np.asarray(background, dtype=float)
    if background.shape != (len(wavelength), atmosphere.n_depth):
        raise ValueError("supplement background must match wavelength and depth axes")
    if np.any(~np.isfinite(background)) or np.any(background < 0):
        raise ValueError("supplement background must be finite and non-negative")
    original = [(i, t) for i, t in selected if not t.independently_validated_supplement]
    additions = [(i, t) for i, t in selected if t.independently_validated_supplement]
    if not additions:
        return selected
    backgrounds = np.maximum(background, 1e-30)
    minimum_background = np.min(backgrounds, axis=0)
    levels = {(i.element, i.charge): {l.index: l for l in i.levels} for i, t in additions}
    temperature = atmosphere.temperature
    coefficient = PI * ELEMENTARY_CHARGE_ESU**2 / (ELECTRON_MASS * LIGHT_SPEED)
    retained = []
    for offset in range(0, len(additions), 1024):
        chunk = additions[offset:offset + 1024]
        centers = np.array([t.wavelength_vacuum_angstrom for i, t in chunk])
        strength = np.array([t.absorption_oscillator_strength for i, t in chunk])
        lower = [levels[i.element, i.charge][t.lower_index] for i, t in chunk]
        energy = np.array([l.energy_wavenumber for l in lower]) * PLANCK * LIGHT_SPEED
        weight = np.array([l.statistical_weight for l in lower])
        stage = np.array([state.ion_number_density[i.element][i.charge]
            / state.partition_function[i.element, i.charge] for i, t in chunk])
        mass = np.array([i.atomic_mass_u for i, t in chunk]) * 1.66053906892e-24
        population = stage * weight[:, None] * np.exp(
            -energy[:, None] / (BOLTZMANN * temperature[None, :])) / atmosphere.mass_density[None, :]
        sigma = np.sqrt(BOLTZMANN * temperature[None, :] / mass[:, None]) / (centers[:, None] * 1e-8)
        stimulated = -np.expm1(-PLANCK * LIGHT_SPEED /
            (centers[:, None] * 1e-8 * BOLTZMANN * temperature[None, :]))
        peak = coefficient * strength[:, None] * population * stimulated / (np.sqrt(2 * np.pi) * sigma)
        local = np.column_stack([np.interp(centers, wavelength, backgrounds[:, depth])
                                 for depth in range(atmosphere.n_depth)])
        outside = (centers < wavelength[0]) | (centers > wavelength[-1])
        local[outside] = minimum_background
        ratio = np.max(peak / local, axis=1)
        protected = np.array([t.helium_unified_profile is not None or
            (t.absorption_oscillator_strength >= 0.1 and l.energy_wavenumber <= 500)
            for (i, t), l in zip(chunk, lower)])
        retained.extend(pair for pair, keep in zip(chunk,
            (ratio >= manifest()["screening_peak_ratio"]) | ~np.isfinite(ratio) | protected) if keep)
    return original + retained
