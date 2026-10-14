"""Scoped recovery of the previous Kurucz defaults for immutable controls.

This is a research/test helper, never a production configuration option.
"""
from contextlib import contextmanager, ExitStack
import sys
from unittest.mock import patch

import wd_spectra.metals as metals


@contextmanager
def previous_kurucz_defaults():
    """Recover the full Cr II and selected Fe II reader before qualification.

    The existing Si II configuration switch still applies to the bare atom.
    Resolve augmenters at call time so older nested scopes can disable Fe II.
    Restore aliases imported during the scope as well as existing bindings.
    """
    reader = metals.read_stout_atomic_database

    def previous_reader(*args, **kwargs):
        enabled = kwargs.pop('include_default_supplements', True)
        database = reader(*args, **kwargs, include_default_supplements=False)
        if enabled:
            database = metals.augment_chromium_ii_kurucz_transitions(database)
            database = metals.augment_iron_ii_kurucz_uv_transitions(database)
        return database

    with ExitStack() as stack:
        for module in tuple(sys.modules.values()):
            if module is None or not getattr(module, '__name__', '').startswith('wd_spectra'):
                continue
            for name, value in tuple(vars(module).items()):
                if value is reader:
                    stack.enter_context(patch.object(module, name, previous_reader))
        try:
            yield
        finally:
            for module in tuple(sys.modules.values()):
                if module is None or not getattr(module, '__name__', '').startswith('wd_spectra'):
                    continue
                for name, value in tuple(vars(module).items()):
                    if value is previous_reader:
                        setattr(module, name, reader)
