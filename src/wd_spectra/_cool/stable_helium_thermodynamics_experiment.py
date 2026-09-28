"""Cancel the analytic ideal-gas terms before differencing the He EOS.

Same HM ion/excitation populations and same finite-temperature stencil. No
neutral-gas switch: the analytic translational term is separated at every T.
The REOS branch is deliberately outside this experiment.
"""
from contextlib import contextmanager, ExitStack
from unittest.mock import patch
import numpy as np
from wd_spectra import eos
from wd_spectra.constants import BOLTZMANN, HELIUM_MASS


def stable_helium_thermodynamics(temperature, gas_pressure, **options):
    if options.pop("helium_reos3_table", None) is not None:
        raise ValueError(
            "Stable ideal-HM derivative experiment does not implement REOS derivatives"
        )
    t, p = np.broadcast_arrays(
        np.asarray(temperature, float), np.asarray(gas_pressure, float)
    )
    h = 2e-4
    cold_t, hot_t = t * np.exp(-h), t * np.exp(h)
    cold = eos.hummer_mihalas_helium_lte(cold_t, p, **options)
    hot = eos.hummer_mihalas_helium_lte(hot_t, p, **options)
    central = eos.hummer_mihalas_helium_lte(t, p, **options)

    def reaction_energy(temp, state):
        _, _, _, e0, e1 = eos._helium_partition_functions(
            temp,
            state.electron_density,
            state.neutral_he_density,
            maximum_level=options.get("maximum_level", eos.HM_MAX_BOUND_LEVEL),
            neutral_radius_scale=options.get(
                "neutral_radius_scale", eos.HM_HELIUM_NEUTRAL_RADIUS_SCALE
            ),
            correlated_microfields=options.get("correlated_microfields", False),
        )
        return (
            state.neutral_he_density * e0
            + state.singly_ionized_he_density
            * (eos.HELIUM_FIRST_IONIZATION_ENERGY + e1)
            + state.doubly_ionized_he_density
            * (eos.HELIUM_FIRST_IONIZATION_ENERGY + eos.HELIUM_SECOND_IONIZATION_ENERGY)
        ) / state.mass_density

    zc, zh, z0 = (s.mean_ion_charge for s in (cold, hot, central))
    cp = 2.5 * BOLTZMANN / HELIUM_MASS * (
        1 + (hot_t * zh - cold_t * zc) / (hot_t - cold_t)
    ) + (reaction_energy(hot_t, hot) - reaction_energy(cold_t, cold)) / (hot_t - cold_t)
    expansion = 1 + (np.log1p(zh) - np.log1p(zc)) / (2 * h)
    ad = BOLTZMANN * (1 + z0) * expansion / (HELIUM_MASS * cp)
    return eos.HeliumThermodynamics(cp, expansion, ad)


@contextmanager
def stable_helium_thermal_experiment():
    from wd_spectra import atmosphere, convection

    with ExitStack() as stack:
        for module in (eos, atmosphere, convection):
            if hasattr(module, "hummer_mihalas_helium_thermodynamics"):
                stack.enter_context(
                    patch.object(
                        module,
                        "hummer_mihalas_helium_thermodynamics",
                        stable_helium_thermodynamics,
                    )
                )
        yield
