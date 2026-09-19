"""Forecast the coming fire season, before anyone can mark it.

NAFI has not published fire scars for the current year, so this produces a genuine
prediction rather than a retrospective one: there is no answer sheet to check it against
yet. Accuracy quoted anywhere in this project comes from 2023-2025, which the model never
trained on; nothing here can improve that number, and nothing here should be read as if
it had been verified.

The inputs are exactly what would be available on 1 May: fire history through last year,
and the wet season that just ended. The fire-history features come from the same function
the training pipeline uses, so the forecast cannot quietly compute them differently.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pyrantis.schema import CLASSES, INTERIM, PROC, RAW, ROOT as PROJ
from scripts.build_features import features_at, to_grid, neighbour_mean
from scripts.train_models import (FIRE_FEATURES, EXTRA_EXCLUDE, EXTRA_NAMED)

SEED = 42
TARGET = int(os.environ.get("FORECAST_YEAR", 2026))
SILO_DIR = os.environ.get("SILO_DIR", "silo2026")


def weather_for(year: int, cells: pd.DataFrame) -> pd.DataFrame:
    """The same weather features the models train on, for one year, on the same grid."""
    os.environ["SILO_DIR"] = SILO_DIR
    import importlib
    import scripts.build_weather_features as bwf
    importlib.reload(bwf)

    files = sorted((RAW / SILO_DIR).glob("silo_*.txt"))
    assert files, f"no SILO records in {SILO_DIR}"
    print(f"reading {len(files)} point records for {year}", flush=True)

    import re
    frames = []
    for f in files:
        m = re.match(r"silo_([+-][\d.]+)_([+-][\d.]+)\.txt", f.name)
        pf = bwf.point_features(bwf.read_point(f), np.array([year]))
        if not len(pf):
            continue
        pf["lat"], pf["lon"] = float(m.group(1)), float(m.group(2))
        frames.append(pf)
    pts = pd.concat(frames, ignore_index=True)

    # Climatology over the same training-period years the main pipeline uses, so the
    # anomaly features mean the same thing here as they do there.
    clim_frames = []
    for f in files:
        m = re.match(r"silo_([+-][\d.]+)_([+-][\d.]+)\.txt", f.name)
        cf = bwf.point_features(bwf.read_point(f), np.array(list(bwf.CLIM_YEARS)))
        if not len(cf):
            continue
        cf["lat"], cf["lon"] = float(m.group(1)), float(m.group(2))
        clim_frames.append(cf)
    clim = (pd.concat(clim_frames, ignore_index=True)
            .groupby(["lat", "lon"])[["wx_wet_rain", "wx_dry_rain_prev"]]
            .mean().rename(columns=lambda c: c + "_clim").reset_index())

    pts = pts.merge(clim, on=["lat", "lon"], how="left")
    pts["wx_wet_rain_anom"] = pts["wx_wet_rain"] - pts["wx_wet_rain_clim"]
    pts["wx_wet_rain_ratio"] = pts["wx_wet_rain"] / pts["wx_wet_rain_clim"].replace(0, np.nan)

    wx = [c for c in pts.columns if c.startswith("wx_") and not c.endswith("_clim")]
    target = cells[["lon", "lat"]].to_numpy()
    src = pts[["lon", "lat"]].to_numpy()
    out = {"cell_id": cells["cell_id"].to_numpy()}
    for c in wx:
        v = pts[c].to_numpy()
        ok = np.isfinite(v)
        z = griddata(src[ok], v[ok], target, method="linear")
        gap = ~np.isfinite(z)
        if gap.any():
            z[gap] = griddata(src[ok], v[ok], target[gap], method="nearest")
        out[c] = z
    return pd.DataFrame(out)


def main() -> None:
    lab = pd.read_parquet(INTERIM / "cell_year_labels.parquet")
    years = np.sort(lab["year"].unique())
    assert TARGET == years[-1] + 1, \
        f"{TARGET} is not the year after the record ends ({years[-1]})"

    nrow, ncol = lab["row"].max() + 1, lab["col"].max() + 1
    frac_burnt = to_grid(lab, "frac_burnt", nrow, ncol, years)
    frac_early = to_grid(lab, "frac_early", nrow, ncol, years)
    frac_late = to_grid(lab, "frac_late", nrow, ncol, years)
    l = lab["label"].to_numpy()
    burnt = to_grid(lab.assign(v=(l != "unburnt").astype("f4")), "v", nrow, ncol, years)
    early = to_grid(lab.assign(v=(l == "early").astype("f4")), "v", nrow, ncol, years)
    late = to_grid(lab.assign(v=(l == "late").astype("f4")), "v", nrow, ncol, years)

    # t = len(years) means "the year after the record", so every slice [:t] is the whole
    # record and [t-1] is last year. The same function the training pipeline calls.
    f = features_at(len(years), burnt, early, late, frac_burnt, frac_early, frac_late)

    inside = ~np.isnan(frac_burnt[0])
    rr, cc = np.where(inside)
    fc = pd.DataFrame({"row": rr, "col": cc, "year": TARGET})
    for k, v in f.items():
        fc[k] = v[rr, cc] if isinstance(v, np.ndarray) else v

    geo = lab[["cell_id", "row", "col", "lon", "lat"]].drop_duplicates("cell_id")
    fc = fc.merge(geo, on=["row", "col"], how="inner")
    fc = fc.dropna(subset=["nbr_frac_burnt_prev"]).reset_index(drop=True)
    print(f"{len(fc):,} cells carried into the {TARGET} forecast")

    fc = fc.merge(weather_for(TARGET, fc[["cell_id", "lon", "lat"]]), on="cell_id")

    train = pd.read_parquet(PROC / "cell_year_features_full.parquet")
    features = (FIRE_FEATURES
                + [c for c in train.columns if c.startswith("wx_")]
                + [c for c in train.columns
                   if (c.startswith("ndvi_") and c not in EXTRA_EXCLUDE)
                   or c in EXTRA_NAMED])

    # The forecast year needs the same long-memory columns the training table carries,
    # computed the same way: windows ending the year before the target.
    l = lab["label"].to_numpy()
    for w in (15, 20, 25):
        t = len(years)
        lo = max(0, t - w)
        n = t - lo
        fc[f"freq_{w}"] = np.nansum(burnt[lo:t], axis=0)[fc["row"], fc["col"]] / n
        fc[f"late_freq_{w}"] = np.nansum(late[lo:t], axis=0)[fc["row"], fc["col"]] / n

    # Elevation and greenness are per-cell lookups; greenness for the forecast year is
    # already in the table Earth Engine wrote, since it only reads up to 30 April.
    static = (train[["cell_id", "elevation"]].drop_duplicates("cell_id"))
    fc = fc.merge(static, on="cell_id", how="left")

    ndvi = pd.read_parquet(PROC / "ndvi.parquet")
    ncols = [c for c in features if c.startswith("ndvi_")]
    fc = fc.merge(ndvi[["cell_id", "year"] + ncols], on=["cell_id", "year"], how="left")

    missing = [c for c in features if c not in fc.columns]
    assert not missing, f"forecast is missing {missing}"
    gaps = fc[features].isna().mean().max()
    print(f"{len(features)} features, worst column {gaps*100:.2f}% missing", flush=True)

    y = train["label"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    # Every labelled year is used. There is no held-out year to protect here: the score
    # this forecast is reported against was measured elsewhere, on 2023-2025, by a model
    # that never saw them.
    X = train[features].to_numpy("float32")

    t0 = time.time()
    model = HistGradientBoostingClassifier(
        max_iter=500, learning_rate=0.07, max_leaf_nodes=63, l2_regularization=1.0,
        min_samples_leaf=40, early_stopping=False, class_weight="balanced",
        random_state=SEED)
    model.fit(X, y)
    print(f"trained gradient boosting on {len(X):,} cell-years in "
          f"{time.time() - t0:.0f}s", flush=True)

    Xfc = fc[features].to_numpy("float32")
    prob = model.predict_proba(Xfc)
    pred = prob.argmax(1)

    fc["pred"] = pred
    for i, cls in enumerate(CLASSES):
        fc[f"p_{cls}"] = prob[:, i]

    share = {cls: round(float((pred == i).mean() * 100), 1) for i, cls in enumerate(CLASSES)}
    hist = {cls: round(float((l == cls).mean() * 100), 1) for cls in CLASSES}

    np.savez_compressed(PROJ / "reports" / f"forecast_{TARGET}.npz",
                        cell_id=fc["cell_id"].to_numpy(), row=fc["row"].to_numpy(),
                        col=fc["col"].to_numpy(), pred=pred.astype("int8"),
                        prob=prob.astype("float32"))
    json.dump({"year": TARGET, "share": share, "long_run_share": hist,
               "n_cells": int(len(fc)),
               "mean_confidence": round(float(prob.max(1).mean()), 3),
               "model": "gradient boosting (balanced)",
               "trained_through": int(years[-1]),
               "note": "No published fire scars exist for this year; this forecast is "
                       "unverified."},
              open(PROJ / "reports" / f"forecast_{TARGET}.json", "w"), indent=2)

    print(f"\n{TARGET} forecast, share of the Territory")
    for cls in CLASSES:
        print(f"  {cls:<9}{share[cls]:>6.1f}%   (2000-{years[-1]} average {hist[cls]:.1f}%)")
    print(f"\nmean confidence in the chosen class: {prob.max(1).mean():.3f}")


if __name__ == "__main__":
    main()
