"""Guard against NumPy APIs removed in the supported NumPy 2.x range.

CI runs NumPy >= 2.4 on Python 3.12, where ``np.trapz`` no longer exists; the
local Python 3.9 environment still has it, so a stray call passes locally and
fails in CI.  Use ``wd_spectra._compat.trapezoid`` instead.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REMOVED = re.compile(r"\b(?:np|numpy)\.(?:trapz|in1d|row_stack|product|cumproduct|alltrue|sometrue)\s*\(")


def test_no_removed_numpy_functions():
    offenders = []
    for directory in ("src", "tests", "tools"):
        for path in (ROOT / directory).rglob("*.py"):
            if path.name == "_compat.py":
                continue
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if REMOVED.search(line) and not line.lstrip().startswith("#"):
                    offenders.append(f"{path.relative_to(ROOT)}:{number}: {line.strip()}")
    assert not offenders, "NumPy 2 removed these functions:\n" + "\n".join(offenders)
