"""Build the optional C radiative-transfer backend."""

import os
from setuptools import Extension, setup

optimization_flags = ["/O2"] if os.name == "nt" else ["-O3"]

setup(
    ext_modules=[
        Extension(
            "wd_spectra._rt",
            sources=["csrc/rt_core.c", "csrc/stark_profiles.c"],
            optional=True,
            extra_compile_args=optimization_flags,
        )
    ]
)
