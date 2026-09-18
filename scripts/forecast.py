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
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pyrantis.schema import CLASSES, INTERIM, PROC, RAW, ROOT as PROJ
from scripts.build_features import features_at, to_grid, neighbour_mean
from scripts.train_models import (FIRE_FEATURES, SEQ_LEN, build_sequences, BestMacroF1)

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

    train = pd.read_parquet(PROC / "cell_year_features_weather.parquet")
    features = FIRE_FEATURES + [c for c in train.columns if c.startswith("wx_")]
    missing = [c for c in features if c not in fc.columns]
    assert not missing, f"forecast is missing {missing}"

    y = train["label"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    # Operationally you would use every labelled year. The last two are kept back only
    # to decide when to stop training, never to report a score.
    tr = (train["year"] <= years[-1] - 2).to_numpy()
    va = (train["year"] > years[-1] - 2).to_numpy()

    X = train[features].to_numpy("float32")
    scaler = StandardScaler().fit(X[tr])
    Str, Sva = scaler.transform(X[tr]), scaler.transform(X[va])
    Sfc = scaler.transform(fc[features].to_numpy("float32")).astype("float32")

    seq = build_sequences(lab, train)
    Qtr, Qva = seq[tr], seq[va]

    # The forecast year's sequence is the ten years up to and including last year.
    yidx = {yy: i for i, yy in enumerate(years)}
    chan = np.full((len(years), nrow, ncol, 6), np.nan, dtype="float32")
    yi = lab["year"].map(yidx).to_numpy()
    r, c = lab["row"].to_numpy(), lab["col"].to_numpy()
    for k, col in enumerate(("frac_burnt", "frac_early", "frac_late")):
        chan[yi, r, c, k] = lab[col].to_numpy()
    for k, cls in enumerate(CLASSES):
        chan[yi, r, c, 3 + k] = (l == cls).astype("f4")
    Qfc = np.zeros((len(fc), SEQ_LEN, 6), dtype="float32")
    for pos in range(SEQ_LEN):
        src = len(years) - (SEQ_LEN - pos)
        Qfc[:, pos] = chan[src, fc["row"].to_numpy(), fc["col"].to_numpy()]
    Qfc = np.nan_to_num(Qfc, nan=0.0)

    import tensorflow as tf
    from tensorflow import keras
    tf.keras.utils.set_random_seed(SEED)

    seq_in = keras.layers.Input((SEQ_LEN, 6), name="history")
    sta_in = keras.layers.Input((len(features),), name="static")
    h = keras.layers.Dropout(0.2)(keras.layers.LSTM(64)(seq_in))
    g = keras.layers.Dense(32, activation="relu")(sta_in)
    z = keras.layers.Dense(32, activation="relu")(keras.layers.Concatenate()([h, g]))
    model = keras.Model([seq_in, sta_in], keras.layers.Dense(3, activation="softmax")(z))
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    cw = dict(enumerate(compute_class_weight("balanced",
                                             classes=np.arange(3), y=y[tr])))
    t0 = time.time()
    model.fit([Qtr, Str], y[tr], validation_data=([Qva, Sva], y[va]),
              epochs=25, batch_size=4096, verbose=0, class_weight=cw,
              callbacks=[BestMacroF1([Qva, Sva], y[va])])
    print(f"trained on {tr.sum():,} cell-years in {time.time() - t0:.0f}s")

    prob = model.predict([Qfc, Sfc], verbose=0, batch_size=8192)
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
               "trained_through": int(years[-1] - 2),
               "note": "No published fire scars exist for this year; this forecast is "
                       "unverified."},
              open(PROJ / "reports" / f"forecast_{TARGET}.json", "w"), indent=2)

    print(f"\n{TARGET} forecast, share of the Territory")
    for cls in CLASSES:
        print(f"  {cls:<9}{share[cls]:>6.1f}%   (2000-{years[-1]} average {hist[cls]:.1f}%)")
    print(f"\nmean confidence in the chosen class: {prob.max(1).mean():.3f}")


if __name__ == "__main__":
    main()
