#!/usr/bin/env python3
"""Recover the 6000-K Mg I b--He unified profiles from vector figure data.

Allard, Leininger, Gadea, Brousseau-Couture & Dufour (2016), A&A 588, A142
(doi:10.1051/0004-6361/201527826; open copy https://hal.science/hal-01318327)
plot absolute cross sections of the J0 and J2 components of the Mg b triplet
(3s3p 3P_J - 3s4s 3S1) in Figs. 9-11 at T = 6000 K; Fig. 9 also shows J1,
which is indistinguishable from J0.  The page-5 content stream stores the
curves as vector polylines, so the plotted values are recovered without
raster digitization:

  Fig. 9   n_He = 1e21             detuning -100..1000 cm^-1  (J0, J1, J2)
  Fig. 10  n_He = 1.75, 2, 3e21     detuning -100..1600 cm^-1  (J0, J2)
  Fig. 11  n_He = 2, 3, 4e21        air wavelength 4700-5220 A (J0, J2)

Positive detuning is toward the blue.  The 2e21 and 3e21 curves appear in both
Figs. 10 and 11; Fig. 11 maps onto Fig. 10 when its wavelengths are taken as
air wavelengths about the laboratory air line centres, and the script checks
this to 1% of the peak.  This is figure-resolution data, not the authors'
full opacity tables (available from N. F. Allard on request).

usage: extract_allard_mgib_figure_profiles.py PAPER_PDF OUTPUT_TABLE
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import re
import zlib

import numpy as np

AIR_CENTER = {0: 5167.322, 2: 5183.604}  # NIST air wavelengths, Angstrom
# page-5 polylines: (figure, J, n_He) -> index in stroke order of the stream
CURVES = {
    ("f9", 0, 1.0e21): 77, ("f9", 1, 1.0e21): 78, ("f9", 2, 1.0e21): 79,
    ("f10", 0, 1.75e21): 138, ("f10", 2, 1.75e21): 139,
    ("f10", 0, 2.0e21): 140, ("f10", 2, 2.0e21): 141,
    ("f10", 0, 3.0e21): 142, ("f10", 2, 3.0e21): 143,
    ("f11", 0, 2.0e21): 198, ("f11", 0, 3.0e21): 199, ("f11", 0, 4.0e21): 200,
    ("f11", 2, 2.0e21): 201, ("f11", 2, 3.0e21): 202, ("f11", 2, 4.0e21): 203,
}
# Axis calibrations from the tick marks (PDF points):
#  Fig. 9 : detuning 0 at x=360.62, 18.273 pt per 100 cm^-1; sigma 0 at y=621.62, 20.107 pt per 1e-16
#  Fig. 10: detuning 0 at x=354.17, 11.8257 pt per 100 cm^-1; sigma 0 at y=402.24, 46.863 pt per 1e-16
#  Fig. 11: 4700 A at x=342.95, 19.7435 pt per 50 A; sigma 0 at y=181.47, 47.89 pt per 1e-16
CAL = {
    "f9": lambda x, y: ((x - 360.62) / 18.273 * 100.0, (y - 621.62) / 20.107 * 1e-16),
    "f10": lambda x, y: ((x - 354.17) / 11.8257 * 100.0, (y - 402.24) / 46.863 * 1e-16),
    "f11": lambda x, y: (4700.0 + (x - 342.95) / 19.7435 * 50.0, (y - 181.47) / 47.89 * 1e-16),
}
TOKEN = re.compile(r"\[[^\]]*\]|\((?:\\.|[^\\)])*\)|/[^\s/\[\]()<>]+|[-+]?\d*\.?\d+|[A-Za-z'\"*]+")


def _page5_stream(pdf: bytes) -> str:
    for match in re.finditer(rb"stream\r?\n", pdf):
        start = match.end()
        stop = pdf.find(b"endstream", start)
        try:
            text = zlib.decompress(pdf[start:stop]).decode("latin1")
        except zlib.error:
            continue
        if "(8000K)" in text and "D \\(ea" in text:
            return text
    raise RuntimeError("Figs. 8-11 content stream not found")


def _polylines(text: str) -> list[np.ndarray]:
    lines, current, stack = [], None, []
    for token in TOKEN.findall(text):
        if token == "m":
            y, x = float(stack.pop()), float(stack.pop())
            if current and len(current) > 1:
                lines.append(np.array(current))
            current, stack = [(x, y)], []
        elif token == "l":
            y, x = float(stack.pop()), float(stack.pop())
            current = (current or []) + [(x, y)]
            stack = []
        elif token in ("S", "s", "f", "n", "B", "b"):
            if current and len(current) > 1 and token in ("S", "s"):
                lines.append(np.array(current))
            current, stack = None, []
        elif re.fullmatch(r"[A-Za-z'\"*]+", token):
            stack = []
        else:
            stack.append(token)
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paper_pdf", type=Path)
    parser.add_argument("output_table", type=Path)
    args = parser.parse_args()
    pdf = args.paper_pdf.read_bytes()
    lines = _polylines(_page5_stream(pdf))
    curves = {}
    for (figure, j, density), index in CURVES.items():
        points = lines[index]
        x, y = CAL[figure](points[:, 0], points[:, 1])
        if figure == "f11":
            x = 1.0e8 / x - 1.0e8 / AIR_CENTER[j]   # air wavenumber detuning, cm^-1
        order = np.argsort(x)
        x, y = x[order], np.maximum(y[order], 0.0)
        keep = np.r_[True, np.diff(x) > 0]
        curves[figure, j, density] = (x[keep], y[keep])
    # Fig. 11 must reproduce Fig. 10 where both are plotted.
    for j in (0, 2):
        for density in (2.0e21, 3.0e21):
            x11, y11 = curves["f11", j, density]
            x10, y10 = curves["f10", j, density]
            inside = (x11 > x10[0]) & (x11 < x10[-1])
            rms = np.sqrt(np.mean((np.interp(x11[inside], x10, y10) - y11[inside]) ** 2))
            if rms > 0.01 * y10.max():
                raise RuntimeError(f"Fig. 10/11 mismatch J{j} n={density:g}: rms {rms:.2e}")
    # Keep the widest detuning coverage for each (J, n): Fig. 11 for 2-4e21.
    selected = {}
    for (figure, j, density), values in curves.items():
        if figure == "f11" or (j, density) not in selected:
            selected[j, density] = values
    rows = [np.column_stack((np.full(x.size, density), np.full(x.size, j), x, y))
            for (j, density), (x, y) in sorted(selected.items())]
    header = (
        "Mg I b (3s3p 3P_J - 3s4s 3S1) + He absolute cross sections recovered from "
        "Allard et al. 2016 A&A 588 A142 Figs. 9-11 vector paths\n"
        f"T=6000 K; source PDF sha256 {hashlib.sha256(pdf).hexdigest()}\n"
        "detuning = sigma - sigma_0 in cm^-1 (positive blue); J1 plotted only at 1e21\n"
        "helium_density_cm-3 J detuning_cm-1 cross_section_cm2"
    )
    args.output_table.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(args.output_table, np.concatenate(rows), fmt=("%.4e", "%d", "%.4f", "%.6e"),
               header=header)


if __name__ == "__main__":
    main()
