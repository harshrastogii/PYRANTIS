"""Month-by-month wet-season weather, instead of one seasonal average.

The models currently see the wet season as a handful of totals: rain, mean temperature,
mean vapour pressure deficit. That throws away the shape of the season, and the shape is
what grows grass. Six hundred millimetres falling steadily from November leaves a very
different fuel load from the same six hundred arriving in one March week.

This writes one row per cell-year holding each month from November through April
separately, which gives a recurrent model an actual sequence to read rather than a
pre-averaged summary.

Nothing here reaches past 30 April, so the same information cut-off applies as everywhere
else in the project.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import griddata

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pyrantis.schema import PROC, RAW

OUT = PROC / "weather_monthly.parquet"
# November through April, in the order they happen.
SEASON = [(-1, 11), (-1, 12), (0, 1), (0, 2), (0, 3), (0, 4)]
VARS = ["rain", "tmax", "vpd", "radn"]


def main() -> None:
    os.environ.setdefault("SILO_DIR", "silo2026")
    import scripts.build_weather_features as bwf

    feat = pd.read_parquet(PROC / "cell_year_features.parquet")
    years = np.sort(feat["year"].unique())
    years = np.append(years, years[-1] + 1)          # include the forecast year
    cells = feat[["cell_id", "lon", "lat"]].drop_duplicates("cell_id")

    files = sorted(bwf.SILO.glob("silo_*.txt"))
    assert files, f"no SILO records in {bwf.SILO}"
    print(f"reading {len(files)} point records for {years[0]}-{years[-1]}", flush=True)

    rows = []
    for i, f in enumerate(files, 1):
        m = re.match(r"silo_([+-][\d.]+)_([+-][\d.]+)\.txt", f.name)
        lat, lon = float(m.group(1)), float(m.group(2))
        d = bwf.read_point(f)
        for y in years:
            rec = {"year": int(y), "lat": lat, "lon": lon}
            ok = True
            for k, (off, mon) in enumerate(SEASON):
                lo = pd.Timestamp(int(y) + off, mon, 1)
                hi = lo + pd.offsets.MonthEnd(1)
                s = d.loc[lo:hi]
                if len(s) < 20:
                    ok = False
                    break
                rec[f"m{k}_rain"] = s["rain"].sum()
                rec[f"m{k}_tmax"] = s["tmax"].mean()
                rec[f"m{k}_vpd"] = s["vpd"].mean()
                rec[f"m{k}_radn"] = s["radn"].mean()
            if ok:
                rows.append(rec)
        if i % 100 == 0:
            print(f"  {i}/{len(files)}", flush=True)

    pts = pd.DataFrame(rows)
    cols = [f"m{k}_{v}" for k in range(len(SEASON)) for v in VARS]
    print(f"{len(pts):,} point-years, {len(cols)} monthly columns", flush=True)

    target = cells[["lon", "lat"]].to_numpy()
    out = []
    for y in years:
        py = pts[pts["year"] == y]
        if not len(py):
            continue
        src = py[["lon", "lat"]].to_numpy()
        d = {"cell_id": cells["cell_id"].to_numpy(), "year": int(y)}
        for c in cols:
            v = py[c].to_numpy()
            ok = np.isfinite(v)
            z = griddata(src[ok], v[ok], target, method="linear")
            gap = ~np.isfinite(z)
            if gap.any():
                z[gap] = griddata(src[ok], v[ok], target[gap], method="nearest")
            d[c] = z.astype("float32")
        out.append(pd.DataFrame(d))

    grid = pd.concat(out, ignore_index=True)
    grid.to_parquet(OUT, index=False)
    print(f"\nwrote {OUT.relative_to(ROOT)}  {len(grid):,} rows, {len(cols)} columns")
    print(grid[cols[:8]].describe().T[["mean", "std", "min", "max"]].round(1).to_string())


if __name__ == "__main__":
    main()
