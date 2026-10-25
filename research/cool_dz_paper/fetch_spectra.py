#!/usr/bin/env python3
"""Download the public spectra of the cool DZ/DAZ paper comparison.

Files are written to DATA/<key>/: ESO Phase 3 X-shooter products from the ESO
archive, SDSS DR17 "lite" spectra, and single DESI DR1 coadd rows (read with
HTTP range requests, so only the target's rows of the healpix coadd file are
transferred).  Existing files are kept.  Run with ``python -I``.

usage: python -I research/cool_dz_paper/fetch_spectra.py DATA [KEY ...]
"""

import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
USER_AGENT = {"User-Agent": "openwd-cool-dz-paper/1.0"}


def get(url, byte_range=None, timeout=300):
    headers = dict(USER_AGENT)
    if byte_range is not None:
        headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1] - 1}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout) as response:
        data = response.read()
    if byte_range is not None and len(data) != byte_range[1] - byte_range[0]:
        raise IOError(f"short range read {len(data)} from {url}")
    return data


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_bytes(data)
    tmp.replace(path)


def fetch_eso(directory, dataset_id):
    out = directory / f"{dataset_id.replace(':', '_')}.fits"
    if not out.exists():
        save(out, get(f"https://dataportal.eso.org/dataPortal/file/{dataset_id}"))
    return out


def fetch_sdss(directory, plate_mjd_fiber):
    plate, mjd, fiber = plate_mjd_fiber.split("-")
    out = directory / f"spec-{plate}-{mjd}-{fiber}.fits"
    if out.exists():
        return out
    for url in (
        f"https://data.sdss.org/sas/dr17/sdss/spectro/redux/26/spectra/lite/{plate}/spec-{plate}-{mjd}-{fiber}.fits",
        f"https://data.sdss.org/sas/dr17/eboss/spectro/redux/v5_13_2/spectra/lite/{plate}/spec-{plate}-{mjd}-{fiber}.fits",
    ):
        try:
            save(out, get(url))
            return out
        except Exception as exc:  # try the next reduction
            print("  SDSS miss", url, exc)
    raise IOError(f"SDSS spectrum {plate_mjd_fiber} not found")


def _parse_header(url, offset):
    cards, block = {}, 0
    while True:
        raw = get(url, (offset + 2880 * block, offset + 2880 * (block + 1))).decode("ascii")
        block += 1
        for i in range(36):
            card = raw[80 * i: 80 * (i + 1)]
            if card[:8].strip() == "END":
                return cards, offset + 2880 * block
            if card[8:10] == "= ":
                value = card[10:].split("/")[0].strip()
                if value.startswith("'"):
                    value = card[10:].strip()[1:].split("'")[0].strip()
                cards[card[:8].strip()] = value


def _hdus(url):
    """Yield (cards, data_offset, data_size) for every HDU of a remote FITS file."""
    offset = 0
    while True:
        try:
            cards, data_offset = _parse_header(url, offset)
        except Exception:
            return
        naxis = int(cards.get("NAXIS", 0))
        size = abs(int(cards.get("BITPIX", 8))) // 8 if naxis else 0
        for i in range(1, naxis + 1):
            size *= int(cards[f"NAXIS{i}"])
        size += int(cards.get("PCOUNT", 0)) if naxis else 0
        yield cards, data_offset, size
        offset = data_offset + ((size + 2879) // 2880) * 2880


def fetch_desi(directory, targetid, survey, program, healpix):
    out = directory / f"desi-{targetid}.npz"
    if out.exists():
        return out
    url = (f"https://data.desi.lbl.gov/public/dr1/spectro/redux/iron/healpix/"
           f"{survey}/{program}/{healpix // 100}/{healpix}/coadd-{survey}-{program}-{healpix}.fits")
    row, arrays = None, {}
    for cards, offset, _ in _hdus(url):
        name = cards.get("EXTNAME", "")
        if name == "FIBERMAP":
            nrow, rowlen = int(cards["NAXIS2"]), int(cards["NAXIS1"])
            if cards["TTYPE1"] != "TARGETID" or cards["TFORM1"] != "K":
                raise ValueError(f"unexpected FIBERMAP layout in {url}")
            table = get(url, (offset, offset + nrow * rowlen))
            ids = np.frombuffer(table, dtype=np.uint8).reshape(nrow, rowlen)[:, :8].copy().view(">i8").ravel()
            hits = np.flatnonzero(ids == targetid)
            if hits.size != 1:
                raise ValueError(f"TARGETID {targetid} rows {hits} in {url}")
            row = int(hits[0])
        elif name.endswith("_WAVELENGTH"):
            n = int(cards["NAXIS1"])
            arrays[name] = np.frombuffer(get(url, (offset, offset + 8 * n)), dtype=">f8").astype(float)
        elif name[:2] in ("B_", "R_", "Z_") and name.split("_", 1)[1] in ("FLUX", "IVAR", "MASK"):
            if row is None:
                raise ValueError("FIBERMAP must precede the spectra")
            n = int(cards["NAXIS1"])
            bytes_per_value = abs(int(cards["BITPIX"])) // 8
            dtype = {-32: ">f4", 32: ">i4", 16: ">i2"}[int(cards["BITPIX"])]
            start = offset + row * n * bytes_per_value
            arrays[name] = np.frombuffer(get(url, (start, start + n * bytes_per_value)), dtype=dtype).astype(float)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, url=url, targetid=targetid, row=row, **{k.lower(): v for k, v in arrays.items()})
    return out


def main(data, keys):
    objects = json.loads((HERE / "objects.json").read_text())["objects"]
    for key in keys or list(objects):
        entry, directory = objects[key], data / key
        print("==", key, entry["name"])
        for dataset_id in entry.get("eso", []):
            print("  ESO", fetch_eso(directory, dataset_id))
        if "sdss" in entry:
            print("  SDSS", fetch_sdss(directory, entry["sdss"]))
        if "desi" in entry:
            print("  DESI", fetch_desi(directory, *entry["desi"]))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main(Path(sys.argv[1]), sys.argv[2:])
