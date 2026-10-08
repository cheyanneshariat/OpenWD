"""Private refractive DQ runtime.

The process-local solver adapters in this package are entered only by the DQ
worker. Public callers use :mod:`wd_spectra.models.dq`, which launches an
isolated interpreter. Importing OpenWD never installs these adapters.
"""
