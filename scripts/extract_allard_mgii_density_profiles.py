#!/usr/bin/env python3
"""Recover the 6000-K Mg II h/k + He density sequence from vector figure data.

Allard, Guillon, Alekseev & Kielkopf (2016), A&A 593, A13
(doi:10.1051/0004-6361/201628781; open copy https://hal.science/hal-01384506)
plot absolute cross sections of the P3/2 (k, Fig. 5) and P1/2 (h, Fig. 6)
components at T = 6000 K for n_He = 1e20, 1e21 and 1e22 cm^-3, on log axes.
The figures are Form XObjects whose curves are stored as vector polylines; the
black curves (their preferred "GG" potentials) are recovered here, ordered by
height (top to bottom 1e22, 1e21, 1e20 as in the captions).  Points pinned at
the lower axis limit are dropped.  This is figure-resolution data, not the
authors' full tables.

usage: extract_allard_mgii_density_profiles.py PAPER_PDF OUTPUT_TABLE
Output columns: n_He, component (3/2 = k, 1/2 = h), wavelength (A), sigma (cm^2).
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import re
import zlib

import numpy as np

TOKEN = re.compile(r"\[[^\]]*\]|\((?:\\.|[^\\)])*\)|/[^\s/\[\]()<>]+|[-+]?\d*\.?\d+|[A-Za-z'\"*]+")
# (XObject stream marker, calibration) for Fig. 5 (k) and Fig. 6 (h):
# x tick at 0.20 (0.28) micron = 1131.5 (1328.6), 453.4 (470.6) units per 0.02 micron;
# log10 sigma = -21 (-20) at y = 418, 612 (856.8) units per decade.
FIGURES = {
    "k": dict(axis="-21", x0=1131.5, lam0=2000.0, dx=453.4, y_floor=-21.0, dy=612.0),
    "h": dict(axis="-20", x0=1328.6, lam0=2800.0, dx=470.6, y_floor=-20.0, dy=856.8),
}


def _streams(pdf: bytes):
    for match in re.finditer(rb"stream\r?\n", pdf):
        start = match.end()
        stop = pdf.find(b"endstream", start)
        try:
            yield zlib.decompress(pdf[start:stop]).decode("latin1")
        except zlib.error:
            continue


def _black_polylines(text: str) -> list[np.ndarray]:
    lines, current, stack, color = [], None, [], None
    for token in TOKEN.findall(text):
        if token in ("SCN", "SC", "RG"):
            color = " ".join(stack[-3:])
            stack = []
        elif token == "m":
            y, x = float(stack.pop()), float(stack.pop())
            if current and len(current) > 100 and color == "0 0 0":
                lines.append(np.array(current))
            current, stack = [(x, y)], []
        elif token == "l":
            y, x = float(stack.pop()), float(stack.pop())
            current = (current or []) + [(x, y)]
            stack = []
        elif token in ("S", "s", "f", "n", "B", "b"):
            if current and len(current) > 100 and token in ("S", "s") and color == "0 0 0":
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
    rows = []
    for component, cal in FIGURES.items():
        # Fig. 5 spans 0.20-0.44 micron with ticks to 1e-14; Fig. 6 0.28-0.52.
        def is_figure(s):
            if component == "k":
                return "(0.44)" in s and "(-14)" in s and "(-21)" in s and "(0.52)" not in s
            return "(0.52)" in s and "(-20)" in s and "(-15)" in s and "(0.2)" not in s
        stream = next(s for s in _streams(pdf) if is_figure(s))
        curves = []
        for points in _black_polylines(stream):
            wavelength = cal["lam0"] + (points[:, 0] - cal["x0"]) / cal["dx"] * 200.0
            log_sigma = cal["y_floor"] + (points[:, 1] - 418.0) / cal["dy"]
            order = np.argsort(wavelength)
            wavelength, log_sigma = wavelength[order], log_sigma[order]
            keep = np.r_[True, np.diff(wavelength) > 0] & (log_sigma > cal["y_floor"] + 0.05)
            curves.append((wavelength[keep], log_sigma[keep]))
        if len(curves) != 3:
            raise RuntimeError(f"expected three black curves for {component}, found {len(curves)}")
        curves.sort(key=lambda c: -np.interp(3300.0, c[0], c[1]))
        for (wavelength, log_sigma), density in zip(curves, (1e22, 1e21, 1e20)):
            j = 1.5 if component == "k" else 0.5
            rows.append(np.column_stack((np.full(wavelength.size, density), np.full(wavelength.size, j),
                                         wavelength, 10.0 ** log_sigma)))
    header = (
        "Mg II h (J=1/2) and k (J=3/2) + He absolute cross sections recovered from "
        "Allard et al. 2016 A&A 593 A13 Figs. 5-6 vector paths (GG potentials)\n"
        f"T=6000 K; source PDF sha256 {hashlib.sha256(pdf).hexdigest()}\n"
        "helium_density_cm-3 upper_J wavelength_angstrom cross_section_cm2"
    )
    args.output_table.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(args.output_table, np.concatenate(rows), fmt=("%.4e", "%.1f", "%.3f", "%.6e"), header=header)


if __name__ == "__main__":
    main()
