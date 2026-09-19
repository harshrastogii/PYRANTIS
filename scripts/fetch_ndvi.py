"""Pull MODIS NDVI from Earth Engine onto this project's exact grid.

Everything the models have been told about fuel so far is rainfall wearing a disguise.
Rain is a decent proxy for how much grass grew, but NDVI is a measurement of it: the
satellite is looking at the greenness of the ground that will or will not carry a fire in
August. The literature on Australian savanna burning puts fuel load ahead of every other
predictor, and this is the first time the project has had any.

Two things make this cheap. Earth Engine does the compositing server side, so nothing
larger than the answer is ever transferred. And the answer is small: the Territory at 0.05
degrees is 301 by 181 pixels, so a whole year is a 54,000 element array.

The grid is pinned with an explicit crsTransform rather than a scale, so the pixels land
exactly on the cells the rest of the project uses instead of nearly on them.

Information cut-off: every composite ends on 30 April, the same rule the weather follows.
"""

from __future__ import annotations

import io
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pyrantis.schema import CELL_DEG, INTERIM, PROC

PROJECT = "terraiq-firewatch-and-avian"
OUT = PROC / "ndvi.parquet"
# MOD13Q1 is the 250 m, 16-day vegetation index product. It starts in February 2000, so
# the first full wet season it can describe is the one ending in April 2001.
COLLECTION = "MODIS/061/MOD13Q1"
FIRST_YEAR = 2001


def log(m=""):
    print(m, flush=True)


def grid_spec():
    lab = pd.read_parquet(INTERIM / "cell_year_labels.parquet")
    geo = lab[["row", "col", "lon", "lat"]].drop_duplicates(["row", "col"])
    nrow, ncol = int(geo["row"].max()) + 1, int(geo["col"].max()) + 1
    # Row and column zero of the raster sit outside the Territory, so neither appears in
    # the table. The corner is worked back from any cell that does, using the fact that
    # the grid is regular.
    r0 = geo.iloc[0]
    half = CELL_DEG / 2
    north = float(r0["lat"]) + int(r0["row"]) * CELL_DEG + half
    west = float(r0["lon"]) - int(r0["col"]) * CELL_DEG - half
    return nrow, ncol, west, north


def fetch_year(ee, year, nrow, ncol, west, north):
    """April mean, wet-season mean, peak and trough NDVI, as 2-D arrays on the grid."""
    region = ee.Geometry.Rectangle(
        [west, north - nrow * CELL_DEG, west + ncol * CELL_DEG, north],
        proj="EPSG:4326", geodesic=False)

    def scaled(a, b):
        # NDVI ships as an integer scaled by 10,000.
        return (ee.ImageCollection(COLLECTION).filterDate(a, b)
                .select("NDVI").map(lambda i: i.multiply(0.0001)))

    wet = scaled(f"{year - 1}-11-01", f"{year}-05-01")
    apr = scaled(f"{year}-04-01", f"{year}-05-01")

    img = ee.Image.cat([
        apr.mean().rename("ndvi_apr"),
        wet.mean().rename("ndvi_wet_mean"),
        wet.max().rename("ndvi_wet_max"),
        wet.min().rename("ndvi_wet_min"),
    ])

    # The output grid is pinned by asking for an exact pixel count over an exact
    # rectangle, which lands each pixel on one analysis cell. Setting a crsTransform
    # instead made Earth Engine evaluate the reprojection at the MODIS native 250 m and
    # refuse the job as too large; asking for dimensions lets it downsample through its
    # own pyramid, which averages rather than samples for continuous bands.
    url = img.getDownloadURL({"region": region, "crs": "EPSG:4326",
                              "dimensions": f"{ncol}x{nrow}", "format": "NPY"})
    r = requests.get(url, timeout=600)
    r.raise_for_status()
    body = r.content
    if body[:2] == b"PK":                     # a zip of one .npy per band
        with zipfile.ZipFile(io.BytesIO(body)) as z:
            arrs = {n.split(".")[-2]: np.load(io.BytesIO(z.read(n))) for n in z.namelist()}
        return arrs
    a = np.load(io.BytesIO(body), allow_pickle=True)
    return {n: a[n] for n in a.dtype.names}


def main():
    import ee
    ee.Initialize(project=PROJECT)

    nrow, ncol, west, north = grid_spec()
    log(f"grid {nrow} x {ncol}, north-west corner {north:.4f} {west:.4f}, "
        f"cell {CELL_DEG} deg")

    lab = pd.read_parquet(INTERIM / "cell_year_labels.parquet")
    geo = lab[["cell_id", "row", "col"]].drop_duplicates("cell_id")
    last = int(lab["year"].max()) + 1          # include the forecast year

    frames, t0 = [], time.time()
    for year in range(FIRST_YEAR, last + 1):
        try:
            arrs = fetch_year(ee, year, nrow, ncol, west, north)
        except Exception as e:                                    # noqa: BLE001
            log(f"  {year}  FAILED  {type(e).__name__}: {str(e)[:160]}")
            continue
        d = {"cell_id": geo["cell_id"].to_numpy(), "year": year}
        r, c = geo["row"].to_numpy(), geo["col"].to_numpy()
        for name, a in arrs.items():
            a = np.asarray(a, dtype="float32")
            if a.shape != (nrow, ncol):
                log(f"  {year}  shape {a.shape}, expected {(nrow, ncol)}")
            d[name] = a[np.clip(r, 0, a.shape[0] - 1), np.clip(c, 0, a.shape[1] - 1)]
        frames.append(pd.DataFrame(d))
        got = frames[-1]
        log(f"  {year}  april NDVI mean {np.nanmean(got['ndvi_apr']):.3f}  "
            f"peak {np.nanmean(got['ndvi_wet_max']):.3f}  "
            f"({time.time() - t0:.0f}s elapsed)")

    assert frames, "no years fetched"
    df = pd.concat(frames, ignore_index=True)

    # Greenness anomaly against each cell's own training-period normal, so a cell in the
    # Top End is compared with itself rather than with the desert.
    clim = (df[df["year"].between(2005, 2019)]
            .groupby("cell_id")[["ndvi_apr", "ndvi_wet_max"]].mean()
            .rename(columns=lambda c: c + "_clim"))
    df = df.merge(clim, on="cell_id", how="left")
    df["ndvi_apr_anom"] = df["ndvi_apr"] - df["ndvi_apr_clim"]
    df["ndvi_peak_anom"] = df["ndvi_wet_max"] - df["ndvi_wet_max_clim"]
    # How much the country greened up over the wet season, which is the growth itself
    # rather than the standing total.
    df["ndvi_greenup"] = df["ndvi_wet_max"] - df["ndvi_wet_min"]

    PROC.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    cols = [c for c in df.columns if c.startswith("ndvi")]
    log(f"\nwrote {OUT.relative_to(ROOT)}  {len(df):,} rows, {len(cols)} columns, "
        f"{df['year'].min()}-{df['year'].max()}")
    log(df[cols].describe().T[["mean", "std", "min", "max"]].round(3).to_string())


if __name__ == "__main__":
    main()
