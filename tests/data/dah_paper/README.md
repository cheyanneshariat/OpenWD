# Frozen DAH paper controls

These eight spectra are the unscaled Kurucz/Griem predictions accepted for the
paper figure on 2026-09-29, before promotion to the public API. The NPZ files
contain the input atmosphere arrays and predicted surface flux per Angstrom;
there are no observations or calibration corrections in these controls.
The manifest preserves stellar parameters and original source/data hashes.

Fixed-state tests require the public default to reproduce these spectra with
atmosphere iteration forbidden. Separate cold-start tests create new atmospheres
from the same stellar parameters and check all five equilibrium gates. The
certificate applies to the underlying nonmagnetic DA structure. Never regenerate
these references automatically inside a test.
