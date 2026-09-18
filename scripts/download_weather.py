"""Fetch daily climate records for a grid of points across the Northern Territory.

Source is SILO's Data Drill (Queensland Government), which takes Bureau of Meteorology
station observations and interpolates them onto a 0.05 degree grid -- the same cell size
this project already uses. Requesting the 'BoM Only' dataset keeps the inputs to Bureau
observations rather than a blend of suppliers.

Interpolated grid values are used in preference to a nearest-station join because the NT
has very few stations for its size: a join would put several hundred kilometres between
many cells and their assigned station, and that error would be invisible in the results.
The interpolation carries its own error, but it is a documented, published product and
the error is at least uniform across the Territory.

Points are sampled every 0.5 degrees (about 50 km). Climate varies smoothly at that
scale in the NT, and 470-odd requests is a reasonable thing to ask of a public service
where 46,000 would not be.

Licence: CC BY 4.0, attributing SILO as the source. Data Drill asks for a contact
address with each request, which is what the username field carries.
"""

from __future__ import annotations

import os
import ssl
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import certifi
import geopandas as gpd
import numpy as np
from shapely.geometry import Point

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pyrantis.schema import RAW, STATE

OUT = RAW / "silo"
ENDPOINT = "https://www.longpaddock.qld.gov.au/cgi-bin/silo/DataDrillDataset.php"
# Data Drill asks for a contact address with every request. It is read from the
# environment rather than written here so that a personal address does not travel with a
# public repository; SILO still gets a real contact, it just is not published alongside
# the code.
CONTACT = os.environ.get("SILO_CONTACT", "")
# The reference tag must be a bare alphanumeric token. A comment containing spaces is
# blocked by the site firewall with a generic "Request Rejected" page, and one containing
# punctuation is rejected by the API itself as an invalid value.
REF = "PYRANTIS"

STEP = 0.5                            # degrees between sampled points
START, FINISH = "19990101", "20251231"

WORKERS = 4                           # polite concurrency against a public service
RETRIES = 3
MIN_BYTES = 50_000                    # a real 27-year record is well over a megabyte

SSL_CTX = ssl.create_default_context(cafile=certifi.where())


def nt_points() -> list[tuple[float, float]]:
    """Sampled points lying inside the Territory, plus a small coastal tolerance."""
    g = gpd.read_file(RAW / "boundary" / "au_states.geojson")
    nt = g[g["STATE_NAME"].str.contains("Northern", case=False)].geometry.union_all()
    # Buffer outward so points just offshore still anchor the coastal cells, which is
    # where much of the early-season burning happens.
    nt_buf = nt.buffer(0.25)

    lo, la = nt.bounds[0], nt.bounds[1]
    lons = np.arange(np.floor(lo / STEP) * STEP, nt.bounds[2] + STEP, STEP)
    lats = np.arange(np.floor(la / STEP) * STEP, nt.bounds[3] + STEP, STEP)
    pts = [
        (round(float(y), 3), round(float(x), 3))
        for y in lats for x in lons
        if nt_buf.contains(Point(x, y))
    ]
    return pts


def fetch(lat: float, lon: float) -> str:
    dest = OUT / f"silo_{lat:+07.2f}_{lon:+07.2f}.txt"
    if dest.exists() and dest.stat().st_size > MIN_BYTES:
        return "skip"

    q = urllib.parse.urlencode({
        "lat": lat, "lon": lon, "start": START, "finish": FINISH,
        "format": "alldata", "username": CONTACT, "password": "apirequest",
        "comment": REF,
    })
    for attempt in range(1, RETRIES + 1):
        try:
            req = urllib.request.Request(f"{ENDPOINT}?{q}", headers={"User-Agent": "PYRANTIS/1.0"})
            with urllib.request.urlopen(req, timeout=180, context=SSL_CTX) as r:
                body = r.read()
            if len(body) < MIN_BYTES:
                raise ValueError(f"short response, {len(body)} bytes")
            dest.write_bytes(body)
            return "ok"
        except Exception as e:                                  # noqa: BLE001
            if attempt == RETRIES:
                return f"FAIL {lat},{lon}: {e}"
            time.sleep(3 * attempt)
    return "unreachable"


def main() -> None:
    if not CONTACT or "@" not in CONTACT:
        raise SystemExit(
            "Set SILO_CONTACT to an email address before running, for example:\n"
            "  SILO_CONTACT=you@example.com .venv/bin/python scripts/download_weather.py\n"
            "SILO's Data Drill requires a contact address with each request.")
    OUT.mkdir(parents=True, exist_ok=True)
    pts = nt_points()
    print(f"{len(pts)} sample points across the {STATE} at {STEP} degree spacing",
          flush=True)

    done = fails = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for res in ex.map(lambda p: fetch(*p), pts):
            done += 1
            if res.startswith("FAIL"):
                fails += 1
                print(res, flush=True)
            if done % 25 == 0:
                print(f"  {done}/{len(pts)}  {time.time() - t0:.0f}s elapsed", flush=True)

    got = sorted(OUT.glob("silo_*.txt"))
    mb = sum(f.stat().st_size for f in got) / 1e6
    print(f"\n{len(got)} point records, {mb:.0f} MB, {fails} failures, "
          f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
