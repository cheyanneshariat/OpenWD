"""Compatibility entry point; implementation lives in the installed package."""
import importlib
import runpy
import sys

if __name__ == "__main__":
    runpy.run_module("wd_spectra._cool.dense_mesh_hierarchy", run_name="__main__")
else:
    sys.modules[__name__] = importlib.import_module("wd_spectra._cool.dense_mesh_hierarchy")
