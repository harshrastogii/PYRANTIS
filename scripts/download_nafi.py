#!/usr/bin/env python3
"""Download NAFI annual fire-scar shapefiles.

NAFI (North Australia and Rangelands Fire Information, Charles Darwin University)
publishes 250 m burnt-area mapping for each year from 2000. Each annual shapefile
carries a `month` attribute — the month covering the largest part of the mapping
period in which the polygon was detected as burnt — and a `region` code, which lets us
select the Northern Territory without a separate boundary layer.

Downloads are skipped when the file is already present and non-trivial in size, so the
script is safe to re-run and resumes after an interruption.

    python scripts/download_nafi.py --from 2000 --to 2025
"""
from __future__ import annotations

import argparse
import ssl
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import certifi

# The GeoTIFF is the better source for this task, not the shapefile. Each pixel is
# tagged with the month it was detected as burnt (January = 1 ... December = 12, and 0
# for unburnt), at 250 m, so the early/late label is a direct numpy comparison rather
# than an overlay of roughly a hundred thousand polygons per year. It is also 2.7 MB a
# year against 16 MB.
BASE = "https://firenorth.org.au/nafi3/downloads/firescars/{year}/{year}%20firescar%20image%20files.zip"
OUT = Path(__file__).resolve().parent.parent / "data" / "raw" / "nafi"
# NAFI's web server rejects the default urllib agent.
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
MIN_BYTES = 100_000            # anything smaller is an error page, not a shapefile
RETRIES = 3

# A bare Python install on macOS has no CA bundle, so urllib rejects every HTTPS
# request with CERTIFICATE_VERIFY_FAILED even though curl succeeds. Point it at
# certifi's bundle rather than disabling verification.
SSL_CTX = ssl.create_default_context(cafile=certifi.where())


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fetch(year: int) -> tuple[bool, str]:
    dest = OUT / f"firescars_{year}_tif.zip"
    if dest.exists() and dest.stat().st_size > MIN_BYTES:
        return True, f"{year}  already present ({dest.stat().st_size/1e6:.1f} MB)"

    url = BASE.format(year=year)
    last = ""
    for attempt in range(1, RETRIES + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=180, context=SSL_CTX) as r, \
                 open(dest, "wb") as fh:
                size = 0
                while chunk := r.read(1 << 16):
                    fh.write(chunk)
                    size += len(chunk)
            if size < MIN_BYTES:
                dest.unlink(missing_ok=True)
                last = f"response too small ({size} bytes)"
            else:
                return True, f"{year}  {size/1e6:6.1f} MB"
        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
            last = str(exc)[:90]
        if attempt < RETRIES:
            time.sleep(3 * attempt)
    return False, f"{year}  FAILED: {last}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", type=int, default=2000)
    ap.add_argument("--to", dest="end", type=int, default=2025)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    ok, failed = 0, []
    for year in range(args.start, args.end + 1):
        good, msg = fetch(year)
        log(msg)
        if good:
            ok += 1
        else:
            failed.append(year)

    total = sum(f.stat().st_size for f in OUT.glob("*.zip"))
    log(f"done: {ok} of {args.end - args.start + 1} years, {total/1e9:.2f} GB on disk")
    if failed:
        log(f"failed years: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
