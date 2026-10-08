"""Small readers used to compare OpenWD with published grids."""

from .koester import (
    KoesterSpectrum,
    air_to_vacuum,
    physical_air_to_vacuum,
    read_svo_koester_ascii,
)

__all__ = [
    "KoesterSpectrum",
    "air_to_vacuum",
    "physical_air_to_vacuum",
    "read_svo_koester_ascii",
]

from .observed import (
    ObservedSpectrum, convolve_constant_resolving_power,
    convolve_variable_log_gaussian, local_normalize,
    normalized_profile_metrics, read_sdss_spectrum, read_eso_phase3_spectrum,
)

__all__ += ["ObservedSpectrum", "convolve_constant_resolving_power",
            "convolve_variable_log_gaussian", "local_normalize",
            "normalized_profile_metrics", "read_sdss_spectrum", "read_eso_phase3_spectrum"]
