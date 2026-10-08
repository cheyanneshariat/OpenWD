#!/usr/bin/env python3
"""Recover the red model curves of Williams et al. (2026) Figure 1 from the PDF.

    python research/extract_williams2026_figure1.py PAPER.pdf --output CURVES.npz

Figure 1 of arXiv:2610.07161v1 (page 3) is vector graphics.  Each panel holds
one red polyline (the published Koester-model fit).  Its vertices are mapped to
wavelength and flux with a linear fit to that panel's own tick marks, located
by matching the tick labels.  Requires PyMuPDF.  The result is a recovered
plotted curve, with the plot's sampling and line clipping, not a released
model spectrum.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pymupdf

PANELS = ("fuse_1003", "fuse_1050", "stis_1366", "stis_1434", "stis_1452", "stis_1640")


def _number(text: str) -> float | None:
    try:
        return float(text.replace("−", "-"))
    except ValueError:
        return None


def extract(pdf: Path, page_number: int = 2) -> dict:
    page = pymupdf.open(pdf)[page_number]
    drawings = page.get_drawings()
    frames = sorted(
        (d["rect"] for d in drawings
         if d.get("fill") == (1.0, 1.0, 1.0) and 120 < d["rect"].width < 140),
        key=lambda r: (round(r.y0), r.x0),
    )
    if len(frames) != 6:
        raise ValueError(f"expected six Figure 1 panels, found {len(frames)}")
    red = [d for d in drawings if d.get("color") == (1.0, 0.0, 0.0) and len(d["items"]) > 50]
    ticks = [
        d for d in drawings if d.get("color") == (0.0, 0.0, 0.0) and len(d["items"]) == 1
        and d["items"][0][0] == "l"
    ]
    words = [(w[0], w[1], w[2], w[3], _number(w[4])) for w in page.get_text("words")]
    result, record = {}, {}
    for name, frame in zip(PANELS, frames):
        path = next(d for d in red if frame.contains(d["rect"].tl + (0.5, 0.5)))
        points = [path["items"][0][1]] + [item[2] for item in path["items"]]
        px = np.array([p.x for p in points]); py = np.array([p.y for p in points])

        def axis_map(horizontal: bool):
            labels = [w for w in words if w[4] is not None and (
                (horizontal and frame.x0 - 15 < (w[0] + w[2]) / 2 < frame.x1 + 15
                 and -3 < w[1] - frame.y1 < 8)
                or (not horizontal and 0 < frame.x0 - w[2] < 8
                    and frame.y0 - 6 < (w[1] + w[3]) / 2 < frame.y1 + 6))]
            pairs = []
            for x0, y0, x1, y1, value in labels:
                centre = (x0 + x1) / 2 if horizontal else (y0 + y1) / 2
                marks = []
                for d in ticks:
                    a, b = d["items"][0][1], d["items"][0][2]
                    if horizontal and abs(a.x - b.x) < 0.01 and abs(b.y - a.y) < 6 \
                            and abs(max(a.y, b.y) - frame.y1) < 0.6:
                        marks.append(a.x)
                    if not horizontal and abs(a.y - b.y) < 0.01 and abs(b.x - a.x) < 6 \
                            and abs(min(a.x, b.x) - frame.x0) < 0.6:
                        marks.append(a.y)
                if marks:
                    nearest = min(marks, key=lambda m: abs(m - centre))
                    if abs(nearest - centre) < 3:
                        pairs.append((nearest, value))
            pairs = sorted(set(pairs))
            if len(pairs) < 2:
                raise ValueError(f"{name}: too few tick marks")
            coordinate, value = np.array(pairs).T
            slope, offset = np.polyfit(coordinate, value, 1)
            residual = float(np.max(np.abs(slope * coordinate + offset - value)))
            return slope, offset, residual, pairs

        xs, xo, xr, xp = axis_map(True)
        ys, yo, yr, yp = axis_map(False)
        wave, flux = xs * px + xo, ys * py + yo
        order = np.argsort(wave)
        result[f"{name}_wavelength"] = wave[order]
        result[f"{name}_flux_1e-12"] = flux[order]
        record[name] = {"vertices": int(px.size), "x_ticks": xp, "y_ticks": yp,
                        "x_fit_max_residual": xr, "y_fit_max_residual": yr}
    return result, record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    curves, record = extract(args.pdf)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **curves)
    args.output.with_suffix(".json").write_text(json.dumps(
        {"source": str(args.pdf), "page_index": 2, "panels": record}, indent=2, default=float) + "\n")
    print(json.dumps(record, indent=1, default=float))


if __name__ == "__main__":
    main()
