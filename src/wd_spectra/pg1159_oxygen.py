"""Production assembly of the extended PG 1159 O VI model atom."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import re
from types import MappingProxyType
from typing import Mapping

import numpy as np

from .light_metal_nlte import (
    TlustyPhotoionizationThresholdData,
    read_chianti_term_collision_strengths,
    read_norad_oxygen_vi_photoionization_data,
    read_sirocco_topbase_photoionization_data,
    read_tmad_structure_model_atom,
    rydberg_angular_momentum_mixing_collision_data,
    rydberg_quadrupole_angular_momentum_mixing_collision_data,
)
from .pg1159 import PG1159NLTEModel


EXTENDED_OVI_LEVELS_PER_CHARGE = MappingProxyType(
    {2: 47, 3: 83, 4: 105, 5: 54, 6: 1}
)
_TLUSTY_RESOLVED_HIGH_L_INDICES = (9, 13, 14)  # O VI 4f, 5f, 5g.


def _resolved_oxygen_vi_bound_free_hybrid(
    topbase: TlustyPhotoionizationThresholdData,
    tlusty: TlustyPhotoionizationThresholdData,
) -> TlustyPhotoionizationThresholdData:
    """Retain resolved TLUSTY/OP high-l terms absent from TOPbase s/p/d."""

    indices = np.asarray(_TLUSTY_RESOLVED_HIGH_L_INDICES, dtype=np.int64)
    return replace(
        topbase,
        threshold_frequency_hz=np.ascontiguousarray(np.concatenate((
            topbase.threshold_frequency_hz,
            tlusty.threshold_frequency_hz[indices],
        ))),
        statistical_weight=np.ascontiguousarray(np.concatenate((
            topbase.statistical_weight,
            tlusty.statistical_weight[indices],
        ))),
        level_label=topbase.level_label + tuple(
            tlusty.level_label[index] for index in indices
        ),
        threshold_cross_section_cm2=np.ascontiguousarray(np.concatenate((
            topbase.threshold_cross_section_cm2,
            tlusty.threshold_cross_section_cm2[indices],
        ))),
        log_frequency_ratio=topbase.log_frequency_ratio + tuple(
            tlusty.log_frequency_ratio[index] for index in indices
        ),
        log_cross_section_megabar=topbase.log_cross_section_megabar + tuple(
            tlusty.log_cross_section_megabar[index] for index in indices
        ),
        photoionization_formula=topbase.photoionization_formula + tuple(
            tlusty.photoionization_formula[index] for index in indices
        ),
        photoionization_parameters=topbase.photoionization_parameters + tuple(
            tlusty.photoionization_parameters[index] for index in indices
        ),
        source=(
            topbase.source
            + "; resolved TLUSTY/OP high-l complements: 4f, 5f, 5g"
        ),
    )


def _supplement_oxygen_vi_bound_free(
    primary: TlustyPhotoionizationThresholdData,
    supplement: TlustyPhotoionizationThresholdData,
    population_ion,
    *,
    minimum_principal_quantum_number: int,
    minimum_angular_momentum: int,
) -> TlustyPhotoionizationThresholdData:
    """Append source-resolved high-l terms absent from the primary table."""

    angular_momentum_by_letter = {
        letter: value for value, letter in enumerate("SPDFGHIKLMNOQ")
    }
    selected = []
    primary_labels = {label.strip().upper() for label in primary.level_label}
    level_by_label = {
        level.label.strip().upper(): level for level in population_ion.levels
    }
    for index, label in enumerate(supplement.level_label):
        normalized = label.strip().upper()
        level = level_by_label.get(normalized)
        if level is None or normalized in primary_labels:
            continue
        match = re.search(r"(\d{2})([SPDFGHIKLMNOQ])", normalized)
        if match is None:
            continue
        n = int(match.group(1))
        angular_momentum = angular_momentum_by_letter[match.group(2)]
        if n >= minimum_principal_quantum_number and (
            angular_momentum >= minimum_angular_momentum
        ):
            selected.append(index)
    if not selected:
        return primary
    indices = np.asarray(selected, dtype=np.int64)
    return replace(
        primary,
        threshold_frequency_hz=np.ascontiguousarray(np.concatenate((
            primary.threshold_frequency_hz,
            supplement.threshold_frequency_hz[indices],
        ))),
        statistical_weight=np.ascontiguousarray(np.concatenate((
            primary.statistical_weight,
            supplement.statistical_weight[indices],
        ))),
        level_label=primary.level_label + tuple(
            supplement.level_label[index] for index in indices
        ),
        threshold_cross_section_cm2=np.ascontiguousarray(np.concatenate((
            primary.threshold_cross_section_cm2,
            supplement.threshold_cross_section_cm2[indices],
        ))),
        log_frequency_ratio=primary.log_frequency_ratio + tuple(
            supplement.log_frequency_ratio[index] for index in indices
        ),
        log_cross_section_megabar=primary.log_cross_section_megabar + tuple(
            supplement.log_cross_section_megabar[index] for index in indices
        ),
        photoionization_formula=primary.photoionization_formula + tuple(
            supplement.photoionization_formula[index] for index in indices
        ),
        photoionization_parameters=primary.photoionization_parameters + tuple(
            supplement.photoionization_parameters[index] for index in indices
        ),
        source=(
            primary.source
            + "; high-l level-specific complements from "
            + supplement.source
        ),
    )


def pg1159_model_with_extended_oxygen_vi(
    model: PG1159NLTEModel,
    oxygen_tmad_atom: str | Path,
    *,
    sirocco_level_data: str | Path | None = None,
    sirocco_photoionization_data: str | Path | None = None,
    oxygen_vi_chianti_scups: str | Path | None = None,
    oxygen_vi_norad_energy: str | Path | None = None,
    oxygen_vi_norad_partial_photoionization: str | Path | None = None,
    resolved_bound_free: bool = True,
    include_angular_momentum_mixing: bool = True,
    include_quadrupole_angular_momentum_mixing: bool = False,
    levels_per_charge: Mapping[int, int] = EXTENDED_OVI_LEVELS_PER_CHARGE,
    minimum_mixed_principal_quantum_number: int = 7,
    maximum_mixed_principal_quantum_number: int = 10,
    minimum_mixed_angular_momentum: int = 3,
    minimum_norad_principal_quantum_number: int = 6,
    minimum_norad_angular_momentum: int = 3,
) -> PG1159NLTEModel:
    """Return a target-independent 54-level O VI PG 1159 model.

    The assembly promotes the public TMAD O VI atom, optionally replaces its
    shell-superlevel bound-free matches with resolved SIROCCO/TOPbase s/p/d
    data while preserving TLUSTY/OP 4f/5f/5g cross sections.  When supplied,
    CHIANTI effective electron-collision strengths replace g-bar fallbacks for
    their mapped O VI term pairs.  The production form also adds physical
    PSM20 heavy-ion dipole l-mixing links and can add the BTM quadrupole links
    of Deliporanidou et al. (2025).  Both use local plasma collider densities;
    no target parameter, observed line, or empirical rate multiplier enters
    this function.
    """

    oxygen = read_tmad_structure_model_atom(
        oxygen_tmad_atom,
        model.atomic_database,
        target_nlte_levels_per_charge=levels_per_charge,
    )
    oxygen_vi = oxygen.atomic_database.ions[("O", 5)]
    if len(oxygen_vi.levels) != 54:
        raise RuntimeError(
            f"extended O VI atom must contain 54 levels, got {len(oxygen_vi.levels)}"
        )
    thresholds = dict(model.oxygen_photoionization_threshold_data or {})
    if resolved_bound_free:
        if sirocco_level_data is None or sirocco_photoionization_data is None:
            raise ValueError(
                "resolved O VI bound-free assembly requires both SIROCCO files"
            )
        if 5 not in thresholds:
            raise ValueError("the base model has no TLUSTY O VI bound-free data")
        topbase = read_sirocco_topbase_photoionization_data(
            sirocco_level_data,
            sirocco_photoionization_data,
            oxygen_vi,
            line_collision_data=thresholds[5],
        )
        if len(topbase.level_label) != 20:
            raise RuntimeError(
                "expected 20 resolved TOPbase O VI s/p/d terms, got "
                f"{len(topbase.level_label)}"
            )
        thresholds[5] = _resolved_oxygen_vi_bound_free_hybrid(
            topbase, thresholds[5]
        )
    if (oxygen_vi_norad_energy is None) != (
        oxygen_vi_norad_partial_photoionization is None
    ):
        raise ValueError(
            "NORAD O VI assembly requires both the energy and partial "
            "photoionization files"
        )
    if oxygen_vi_norad_energy is not None:
        if 5 not in thresholds:
            raise ValueError("the base model has no O VI bound-free data")
        norad = read_norad_oxygen_vi_photoionization_data(
            oxygen_vi_norad_energy,
            oxygen_vi_norad_partial_photoionization,
            oxygen_vi,
            line_collision_data=thresholds[5],
        )
        thresholds[5] = _supplement_oxygen_vi_bound_free(
            thresholds[5],
            norad,
            oxygen_vi,
            minimum_principal_quantum_number=(
                minimum_norad_principal_quantum_number
            ),
            minimum_angular_momentum=minimum_norad_angular_momentum,
        )
    exact_electron_collisions = (
        read_chianti_term_collision_strengths(
            oxygen_vi_chianti_scups,
            oxygen,
            "O",
            5,
        )
        if oxygen_vi_chianti_scups is not None
        else {}
    )
    mixing = (
        rydberg_angular_momentum_mixing_collision_data(
            oxygen.atomic_database,
            "O",
            5,
            minimum_principal_quantum_number=(
                minimum_mixed_principal_quantum_number
            ),
            maximum_principal_quantum_number=(
                maximum_mixed_principal_quantum_number
            ),
            minimum_angular_momentum=minimum_mixed_angular_momentum,
        )
        if include_angular_momentum_mixing
        else {}
    )
    quadrupole_mixing = (
        rydberg_quadrupole_angular_momentum_mixing_collision_data(
            oxygen.atomic_database,
            "O",
            5,
            minimum_principal_quantum_number=(
                minimum_mixed_principal_quantum_number
            ),
            maximum_principal_quantum_number=(
                maximum_mixed_principal_quantum_number
            ),
            minimum_angular_momentum=minimum_mixed_angular_momentum,
        )
        if include_quadrupole_angular_momentum_mixing
        else {}
    )
    collisions = dict(oxygen.collision_data)
    collisions.update(exact_electron_collisions)
    collisions.update(mixing)
    collisions.update(quadrupole_mixing)
    return replace(
        model,
        oxygen_levels_per_charge=oxygen.levels_per_charge,
        oxygen_population_atomic_database=oxygen.atomic_database,
        oxygen_formal_level_mapping=oxygen.formal_level_mapping,
        oxygen_formal_lte_parent_mapping=oxygen.formal_lte_parent_mapping,
        oxygen_continuum_parent_mapping=oxygen.continuum_parent_mapping,
        oxygen_lte_level_reservoir=oxygen.lte_level_reservoir,
        oxygen_lte_bound_bound_couplings=oxygen.lte_bound_bound_couplings,
        oxygen_effective_dielectronic_couplings=(
            oxygen.effective_dielectronic_couplings
        ),
        oxygen_collision_data=MappingProxyType(collisions),
        oxygen_photoionization_threshold_data=MappingProxyType(thresholds),
    )


__all__ = [
    "EXTENDED_OVI_LEVELS_PER_CHARGE",
    "pg1159_model_with_extended_oxygen_vi",
]
