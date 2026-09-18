#!/usr/bin/env python3
"""Turn NAFI's annual month-coded rasters into a tidy cell-by-year table.

Each NAFI pixel carries the month it was detected as burnt (1-12, or 0 for unburnt) at
250 m. This script clips to the Northern Territory, aggregates 20 x 20 pixel blocks
into analysis cells, and labels each cell-year as unburnt, early or late.

Labelling rule, in order:
  * fewer than MIN_VALID_FRACTION of the cell inside the NT and NAFI's extent -> dropped
  * burnt fraction below MIN_BURNT_FRACTION                                   -> unburnt
  * otherwise the season holding the larger burnt area                        -> early / late

    python scripts/build_grid.py
"""
from __future__ import annotations

import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
from rasterio.windows import Window, from_bounds

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pyrantis.schema import (BLOCK, CELL_DEG, CLASSES, FIRST_YEAR, LAST_YEAR,  # noqa: E402
                             LATE_SEASON_FIRST_MONTH, MIN_BURNT_FRACTION,
                             MIN_VALID_FRACTION, STATE, UNBURNT_VALUE)

RAW = ROOT / "data" / "raw" / "nafi"
OUT = ROOT / "data" / "interim"


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def nt_geometry():
    import geopandas as gpd
    g = gpd.read_file(ROOT / "data" / "raw" / "boundary" / "au_states.geojson")
    nt = g[g["STATE_NAME"] == STATE]
    if nt.empty:
        raise SystemExit(f"{STATE} not found in the boundary file")
    return nt.geometry.union_all()


def open_year(year: int):
    """Yield an open rasterio dataset for a year, reading straight from the zip."""
    z = RAW / f"firescars_{year}_tif.zip"
    if not z.exists():
        return None
    with zipfile.ZipFile(z) as zf:
        tif = next((n for n in zf.namelist() if n.lower().endswith(".tif")), None)
        if tif is None:
            return None
        return rasterio.open(f"zip://{z}!/{tif}")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    nt = nt_geometry()
    minx, miny, maxx, maxy = nt.bounds
    log(f"{STATE} bounds {minx:.3f} {miny:.3f} {maxx:.3f} {maxy:.3f}")

    frames, mask_valid, grid_meta = [], None, None

    for year in range(FIRST_YEAR, LAST_YEAR + 1):
        ds = open_year(year)
        if ds is None:
            log(f"{year}  no raster yet, skipped")
            continue
        with ds:
            win = from_bounds(minx, miny, maxx, maxy, ds.transform)
            # snap the window to whole analysis cells so every block is complete
            col_off = int(np.floor(win.col_off // BLOCK * BLOCK))
            row_off = int(np.floor(win.row_off // BLOCK * BLOCK))
            width = int(np.ceil(win.width / BLOCK) * BLOCK)
            height = int(np.ceil(win.height / BLOCK) * BLOCK)
            width = min(width, ds.width - col_off)
            height = min(height, ds.height - row_off)
            width -= width % BLOCK
            height -= height % BLOCK
            window = Window(col_off, row_off, width, height)
            arr = ds.read(1, window=window)
            transform = ds.window_transform(window)

            if mask_valid is None:
                # inside-NT mask, computed once: the grid is identical every year
                inside = ~geometry_mask([nt], out_shape=arr.shape,
                                        transform=transform, invert=False)
                ny, nx = arr.shape[0] // BLOCK, arr.shape[1] // BLOCK
                valid_px = inside.reshape(ny, BLOCK, nx, BLOCK).sum(axis=(1, 3))
                mask_valid = valid_px / (BLOCK * BLOCK)
                grid_meta = (transform, ny, nx, inside)
                log(f"grid {ny} x {nx} cells of {CELL_DEG:.3f} deg "
                    f"({int((mask_valid >= MIN_VALID_FRACTION).sum()):,} inside the NT)")

        transform, ny, nx, inside = grid_meta
        burnt = (arr != UNBURNT_VALUE) & inside
        early = burnt & (arr < LATE_SEASON_FIRST_MONTH)
        late = burnt & (arr >= LATE_SEASON_FIRST_MONTH)

        blk = lambda a: a.reshape(ny, BLOCK, nx, BLOCK).sum(axis=(1, 3))  # noqa: E731
        n_early, n_late = blk(early), blk(late)
        n_valid = blk(inside)
        n_burnt = n_early + n_late

        with np.errstate(invalid="ignore", divide="ignore"):
            frac_burnt = np.where(n_valid > 0, n_burnt / n_valid, np.nan)
            frac_early = np.where(n_valid > 0, n_early / n_valid, np.nan)
            frac_late = np.where(n_valid > 0, n_late / n_valid, np.nan)

        label = np.full((ny, nx), CLASSES[0], dtype=object)
        is_burnt = frac_burnt >= MIN_BURNT_FRACTION
        label[is_burnt & (n_early >= n_late)] = "early"
        label[is_burnt & (n_late > n_early)] = "late"

        keep = mask_valid >= MIN_VALID_FRACTION
        rows, cols = np.nonzero(keep)
        lon = transform.c + (cols * BLOCK + BLOCK / 2) * transform.a
        lat = transform.f + (rows * BLOCK + BLOCK / 2) * transform.e

        frames.append(pd.DataFrame({
            "cell_id": rows * nx + cols,
            "row": rows.astype("int32"), "col": cols.astype("int32"),
            "year": np.int16(year),
            "lon": lon.astype("float32"), "lat": lat.astype("float32"),
            "valid_frac": mask_valid[keep].astype("float32"),
            "frac_burnt": frac_burnt[keep].astype("float32"),
            "frac_early": frac_early[keep].astype("float32"),
            "frac_late": frac_late[keep].astype("float32"),
            "label": pd.Categorical(label[keep], categories=CLASSES),
        }))
        counts = frames[-1]["label"].value_counts()
        log(f"{year}  {len(frames[-1]):,} cells   "
            + "  ".join(f"{c}={int(counts.get(c,0)):,}" for c in CLASSES))

    if not frames:
        log("no rasters found: run scripts/download_nafi.py first")
        return 1

    df = pd.concat(frames, ignore_index=True)
    dest = OUT / "cell_year_labels.parquet"
    df.to_parquet(dest, index=False)
    log(f"wrote {dest.relative_to(ROOT)}  {len(df):,} rows, "
        f"{df['cell_id'].nunique():,} cells, {df['year'].nunique()} years")
    log("class balance overall: " + ", ".join(
        f"{c} {100*(df['label']==c).mean():.1f}%" for c in CLASSES))
    return 0


if __name__ == "__main__":
    sys.exit(main())
