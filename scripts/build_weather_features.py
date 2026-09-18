"""Turn daily SILO climate records into per-cell, per-year fire-weather features.

THE INFORMATION CUT-OFF

Every feature here is restricted to weather observed on or before 30 April of the target
year. The NT fire season runs roughly May to October, and the early/late split falls on
31 July. Using August weather to decide whether a cell burnt late would not be prediction,
it would be description: the model would be told about the conditions during the fire it
is supposed to be anticipating, and its accuracy would not survive contact with a real
forecasting task.

Cutting at the end of April keeps one consistent rule with the fire-history features and
makes the result operationally meaningful: everything the model uses is known by the time
the wet season ends, so it could genuinely be run in May to say which parts of the
Territory are heading for a late-season fire year. The cost is accuracy, and that cost is
reported rather than avoided.

WHAT THE FEATURES REPRESENT

Savanna fire is fuel-limited in the arid south and curing-limited in the north. A big wet
season grows grass that will burn months later -- the 2011 fires in central Australia
followed the 2010-11 La Nina -- while the timing of the last rain sets how early that
grass dries enough to carry fire. So the features split into fuel growth (rainfall over
this and the previous wet season) and drying (time since meaningful rain, vapour pressure
deficit, temperature).

Point records on a 0.5 degree sample grid are interpolated to the 0.05 degree analysis
cells linearly, falling back to nearest-neighbour outside the convex hull of the points.
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

from pyrantis.schema import MIN_HISTORY_YEARS, PROC, RAW

SILO = RAW / os.environ.get("SILO_DIR", "silo")
OUT = PROC / "cell_year_features_weather.parquet"

COLS = ["date", "doy", "date2", "tmax", "smx", "tmin", "smn", "rain", "srn",
        "evap", "sev", "radn", "ssl", "vp", "svp", "rhmax", "rhmin", "fao56",
        "mlake", "mpot", "mact", "mwet", "span", "ssp", "evsp", "ses", "mslp", "sp"]

CUTOFF_MONTH, CUTOFF_DAY = 4, 30       # information available by 30 April
WET_START_MONTH = 11                   # wet season runs November to April

# Climatological means are computed over the training period only, so that an anomaly
# feature in a test year is not measured against an average that already contains it.
CLIM_YEARS = range(2000, 2020)


def read_point(path: Path) -> pd.DataFrame:
    txt = path.read_text(errors="replace").splitlines()
    start = next(i for i, l in enumerate(txt) if l.lstrip().startswith("Date ")) + 2
    df = pd.read_csv(io_lines(txt[start:]), sep=r"\s+", names=COLS, header=None,
                     engine="python")
    df = df[["date", "tmax", "tmin", "rain", "radn", "vp"]].copy()
    for c in ("tmax", "tmin", "rain", "radn", "vp"):
        v = pd.to_numeric(df[c], errors="coerce")
        df[c] = v.mask((v <= -99) | (v >= 999))
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["date"]).set_index("date")

    # Vapour pressure deficit at the daily maximum temperature: how thirsty the air gets
    # at the hottest part of the day, which is what drives fuel drying.
    es = 0.6108 * np.exp(17.27 * df["tmax"] / (df["tmax"] + 237.3))     # kPa
    df["vpd"] = (es - df["vp"] / 10.0).clip(lower=0)                    # vp is hPa
    return df


def io_lines(lines):
    import io as _io
    return _io.StringIO("\n".join(lines))


def point_features(df: pd.DataFrame, years: np.ndarray) -> pd.DataFrame:
    rows = []
    for y in years:
        cutoff = pd.Timestamp(y, CUTOFF_MONTH, CUTOFF_DAY)

        wet = df.loc[pd.Timestamp(y - 1, WET_START_MONTH, 1):cutoff]
        wet_prev = df.loc[pd.Timestamp(y - 2, WET_START_MONTH, 1):
                          pd.Timestamp(y - 1, CUTOFF_MONTH, CUTOFF_DAY)]
        # The dry season before this one: May to October of the previous year.
        dry_prev = df.loc[pd.Timestamp(y - 1, 5, 1):pd.Timestamp(y - 1, 10, 31)]
        apr = df.loc[pd.Timestamp(y, CUTOFF_MONTH, 1):cutoff]

        if len(wet) < 100:
            continue

        # Days between the last meaningful rain and the start of the fire season. This
        # is the curing clock: the longer the grass has been drying by 1 May, the sooner
        # it can carry a fire.
        recent = df.loc[pd.Timestamp(y - 1, 7, 1):cutoff]
        wet_days = recent.index[recent["rain"] >= 5.0]
        since_rain = (cutoff - wet_days[-1]).days if len(wet_days) else 305

        heavy = recent.index[recent["rain"] >= 10.0]
        onset = (cutoff - heavy[-1]).days if len(heavy) else 305

        rows.append({
            "year": y,
            "wx_wet_rain": wet["rain"].sum(),
            "wx_wet_rain_prev": wet_prev["rain"].sum(),
            "wx_dry_rain_prev": dry_prev["rain"].sum(),
            "wx_apr_rain": apr["rain"].sum(),
            "wx_days_since_rain": float(since_rain),
            "wx_days_since_heavy": float(onset),
            "wx_wet_tmax": wet["tmax"].mean(),
            "wx_wet_vpd": wet["vpd"].mean(),
            "wx_apr_vpd": apr["vpd"].mean(),
            "wx_wet_radn": wet["radn"].mean(),
            "wx_dry_tmax_prev": dry_prev["tmax"].mean(),
            "wx_dry_vpd_prev": dry_prev["vpd"].mean(),
        })
    return pd.DataFrame(rows)


def main() -> None:
    feat = pd.read_parquet(PROC / "cell_year_features.parquet")
    years = np.sort(feat["year"].unique())

    files = sorted(SILO.glob("silo_*.txt"))
    assert files, "no SILO records found; run scripts/download_weather.py first"
    print(f"reading {len(files)} point records", flush=True)

    frames = []
    for i, f in enumerate(files, 1):
        m = re.match(r"silo_([+-][\d.]+)_([+-][\d.]+)\.txt", f.name)
        lat, lon = float(m.group(1)), float(m.group(2))
        pf = point_features(read_point(f), years)
        pf["lat"], pf["lon"] = lat, lon
        frames.append(pf)
        if i % 100 == 0:
            print(f"  {i}/{len(files)}", flush=True)

    pts = pd.concat(frames, ignore_index=True)
    wx_cols = [c for c in pts.columns if c.startswith("wx_")]

    # Rainfall anomalies against the training-period climatology at each point.
    clim = (pts[pts["year"].isin(CLIM_YEARS)]
            .groupby(["lat", "lon"])[["wx_wet_rain", "wx_dry_rain_prev"]]
            .mean().rename(columns=lambda c: c + "_clim").reset_index())
    pts = pts.merge(clim, on=["lat", "lon"], how="left")
    pts["wx_wet_rain_anom"] = pts["wx_wet_rain"] - pts["wx_wet_rain_clim"]
    pts["wx_wet_rain_ratio"] = pts["wx_wet_rain"] / pts["wx_wet_rain_clim"].replace(0, np.nan)
    wx_cols += ["wx_wet_rain_anom", "wx_wet_rain_ratio"]

    print(f"{len(pts):,} point-years, {len(wx_cols)} weather features", flush=True)

    # Interpolate each feature onto the analysis cells, one year at a time.
    cells = feat[["cell_id", "lon", "lat"]].drop_duplicates("cell_id")
    target = cells[["lon", "lat"]].to_numpy()
    out = []
    for y in years:
        py = pts[pts["year"] == y]
        src = py[["lon", "lat"]].to_numpy()
        d = {"cell_id": cells["cell_id"].to_numpy(), "year": y}
        for c in wx_cols:
            v = py[c].to_numpy()
            ok = np.isfinite(v)
            z = griddata(src[ok], v[ok], target, method="linear")
            # Points outside the convex hull of the sample grid -- the coastal fringe --
            # fall back to the nearest sampled point rather than being dropped.
            gap = ~np.isfinite(z)
            if gap.any():
                z[gap] = griddata(src[ok], v[ok], target[gap], method="nearest")
            d[c] = z
        out.append(pd.DataFrame(d))
    grid = pd.concat(out, ignore_index=True)

    merged = feat.merge(grid, on=["cell_id", "year"], how="inner")
    assert len(merged) == len(feat), f"join changed row count: {len(feat)} -> {len(merged)}"
    assert merged[wx_cols].isna().sum().sum() == 0, "weather features contain gaps"

    merged.to_parquet(OUT, index=False)
    print(f"\nwrote {OUT.relative_to(ROOT)}  {len(merged):,} rows, "
          f"{len(wx_cols)} weather features")
    print(merged[wx_cols].describe().T[["mean", "std", "min", "max"]].round(1).to_string())


if __name__ == "__main__":
    main()
