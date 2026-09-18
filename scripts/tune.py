"""Try to beat 0.674, and report whatever actually happens.

Four things worth trying, in rising order of how much they usually help:

  1. Longer memory. The current features look back at most ten years, but the record runs
     to twenty-six and fire-prone country announces itself over decades.
  2. Elevation, which SILO already carries in every point file and nothing has used.
  3. Gradient boosting, which is not on the unit's required list but the brief allows any
     further algorithm and it usually beats a random forest on tabular data.
  4. An ensemble, averaging the probabilities of the models that disagree most.

Each is scored on the same held-out 2023-2025 as everything else, so the numbers are
comparable to the ones already reported. Anything that fails to help is reported as
failing to help.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pyrantis.schema import CLASSES, INTERIM, PROC, RAW, ROOT as PROJ
from scripts.build_features import to_grid, neighbour_mean
from scripts.train_models import (FIRE_FEATURES, SEQ_LEN, TRAIN_YEARS, VAL_YEARS,
                                  TEST_YEARS, build_sequences, BestMacroF1)

SEED = 42
LONG_WINDOWS = (15, 20, 25)


def log(m=""):
    print(m, flush=True)


def elevation_grid(cells: pd.DataFrame) -> np.ndarray:
    """Elevation per cell, interpolated from the metre figure in each SILO point file."""
    pts = []
    for d in ("silo2026", "silo"):
        files = sorted((RAW / d).glob("silo_*.txt"))
        if files:
            break
    for f in files:
        m = re.match(r"silo_([+-][\d.]+)_([+-][\d.]+)\.txt", f.name)
        head = f.read_text(errors="replace")[:4000]
        e = re.search(r"Elevation:\s*([-\d.]+)\s*m", head)
        if e:
            pts.append((float(m.group(2)), float(m.group(1)), float(e.group(1))))
    src = np.array([[p[0], p[1]] for p in pts])
    val = np.array([p[2] for p in pts])
    tgt = cells[["lon", "lat"]].to_numpy()
    z = griddata(src, val, tgt, method="linear")
    gap = ~np.isfinite(z)
    if gap.any():
        z[gap] = griddata(src, val, tgt[gap], method="nearest")
    log(f"  elevation from {len(pts)} points: {z.min():.0f} to {z.max():.0f} m")
    return z


def long_memory(feat: pd.DataFrame) -> pd.DataFrame:
    """Fire frequency over windows longer than the current ten years."""
    lab = pd.read_parquet(INTERIM / "cell_year_labels.parquet")
    years = np.sort(lab["year"].unique())
    nrow, ncol = lab["row"].max() + 1, lab["col"].max() + 1
    l = lab["label"].to_numpy()
    burnt = to_grid(lab.assign(v=(l != "unburnt").astype("f4")), "v", nrow, ncol, years)
    late = to_grid(lab.assign(v=(l == "late").astype("f4")), "v", nrow, ncol, years)
    yidx = {y: i for i, y in enumerate(years)}

    out = {}
    fr, fc = feat["row"].to_numpy(), feat["col"].to_numpy()
    fy = feat["year"].map(yidx).to_numpy()
    for w in LONG_WINDOWS:
        fb = np.full(len(feat), np.nan, "float32")
        fl = np.full(len(feat), np.nan, "float32")
        for t in np.unique(fy):
            sel = fy == t
            lo = max(0, t - w)
            n = t - lo
            if n <= 0:
                continue
            fb[sel] = np.nansum(burnt[lo:t], axis=0)[fr[sel], fc[sel]] / n
            fl[sel] = np.nansum(late[lo:t], axis=0)[fr[sel], fc[sel]] / n
        out[f"freq_{w}"] = fb
        out[f"late_freq_{w}"] = fl
    return pd.DataFrame(out, index=feat.index)


def score(name, y, p, sec=0.0):
    r = {"model": name,
         "accuracy": round(float(accuracy_score(y, p)), 4),
         "f1_macro": round(float(f1_score(y, p, average="macro", zero_division=0)), 4),
         "f1_late": round(float(f1_score(y, p, average=None, labels=[0, 1, 2],
                                         zero_division=0)[2]), 4),
         "seconds": round(sec, 1)}
    log(f"  {name:<40} acc {r['accuracy']:.3f}  macroF1 {r['f1_macro']:.4f}  "
        f"late {r['f1_late']:.3f}")
    return r


def main():
    feat = pd.read_parquet(PROC / "cell_year_features_weather.parquet").reset_index(drop=True)
    labels = pd.read_parquet(INTERIM / "cell_year_labels.parquet")
    base_features = FIRE_FEATURES + [c for c in feat.columns if c.startswith("wx_")]

    log("adding features")
    cells = feat[["cell_id", "lon", "lat"]].drop_duplicates("cell_id")
    elev = pd.DataFrame({"cell_id": cells["cell_id"].to_numpy(),
                         "elevation": elevation_grid(cells)})
    feat = feat.merge(elev, on="cell_id", how="left")
    feat = pd.concat([feat, long_memory(feat)], axis=1)
    extra = ["elevation"] + [f"freq_{w}" for w in LONG_WINDOWS] + \
            [f"late_freq_{w}" for w in LONG_WINDOWS]
    feat[extra] = feat[extra].astype("float32")
    log(f"  {len(extra)} new columns: {', '.join(extra)}")

    y = feat["label"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    tr = feat["year"].isin(TRAIN_YEARS).to_numpy()
    va = feat["year"].isin(VAL_YEARS).to_numpy()
    te = feat["year"].isin(TEST_YEARS).to_numpy()
    results = []

    for tag, cols in (("baseline features", base_features),
                      ("with longer memory and elevation", base_features + extra)):
        log(f"\n{tag} ({len(cols)} columns)")
        X = feat[cols].to_numpy("float32")
        cw = compute_class_weight("balanced", classes=np.arange(3), y=y[tr])

        t0 = time.time()
        rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=5, n_jobs=-1,
                                    class_weight="balanced_subsample", random_state=SEED)
        rf.fit(X[tr], y[tr])
        results.append(score(f"random forest / {tag}", y[te], rf.predict(X[te]),
                             time.time() - t0))

        t0 = time.time()
        gb = HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.08, max_leaf_nodes=63, l2_regularization=1.0,
            early_stopping=True, validation_fraction=None, random_state=SEED,
            class_weight="balanced")
        gb.fit(X[tr], y[tr])
        results.append(score(f"gradient boosting / {tag}", y[te], gb.predict(X[te]),
                             time.time() - t0))

        if tag.startswith("with"):
            import tensorflow as tf
            from tensorflow import keras
            tf.keras.utils.set_random_seed(SEED)
            sc = StandardScaler().fit(X[tr])
            S = [sc.transform(X[m]).astype("f4") for m in (tr, va, te)]
            seq = build_sequences(labels, feat)
            Q = [seq[tr], seq[va], seq[te]]

            si = keras.layers.Input((SEQ_LEN, seq.shape[2]))
            ti = keras.layers.Input((len(cols),))
            h = keras.layers.Dropout(0.2)(keras.layers.LSTM(96)(si))
            g = keras.layers.Dense(48, activation="relu")(ti)
            z = keras.layers.Dense(48, activation="relu")(
                keras.layers.Concatenate()([h, g]))
            z = keras.layers.Dropout(0.15)(z)
            lstm = keras.Model([si, ti], keras.layers.Dense(3, activation="softmax")(z))
            lstm.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
            t0 = time.time()
            lstm.fit([Q[0], S[0]], y[tr], validation_data=([Q[1], S[1]], y[va]),
                     epochs=40, batch_size=4096, verbose=0,
                     class_weight=dict(enumerate(cw)),
                     callbacks=[BestMacroF1([Q[1], S[1]], y[va], patience=8)])
            pl = lstm.predict([Q[2], S[2]], verbose=0, batch_size=8192)
            results.append(score(f"LSTM 96 units / {tag}", y[te], pl.argmax(1),
                                 time.time() - t0))

            # Averaging probabilities helps only when the models fail on different rows,
            # so the correlation between their errors is printed alongside.
            pg = gb.predict_proba(X[te])
            pr = rf.predict_proba(X[te])
            for nm, mix in (("LSTM + gradient boosting", (pl + pg) / 2),
                            ("LSTM + gradient boosting + forest", (pl + pg + pr) / 3)):
                results.append(score(f"ensemble: {nm}", y[te], mix.argmax(1)))
            wrong_l = pl.argmax(1) != y[te]
            wrong_g = pg.argmax(1) != y[te]
            log(f"  both wrong on {(wrong_l & wrong_g).mean()*100:.1f}% of rows, "
                f"LSTM only {(wrong_l & ~wrong_g).mean()*100:.1f}%, "
                f"boosting only {(~wrong_l & wrong_g).mean()*100:.1f}%")

    best = max(results, key=lambda r: r["f1_macro"])
    log(f"\nbest: {best['model']}  macroF1 {best['f1_macro']:.4f}")
    log("previously reported: LSTM (balanced) macroF1 0.6740")
    json.dump({"results": results, "previous_best": 0.674},
              open(PROJ / "reports" / "tuning.json", "w"), indent=1)


if __name__ == "__main__":
    main()
