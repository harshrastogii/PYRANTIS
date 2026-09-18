"""Test the model the way it would actually be used: train on the past, predict forward.

A single 2023-2025 test told us the model works once. It did not tell us whether it keeps
working, or how fast it decays when the last year it saw recedes. This walks a cutoff
through the record and, from each cutoff, predicts every remaining year.

For a cutoff C the model trains on 2005 to C-1, uses C to decide when to stop, then
predicts C+1, C+2 and so on to the end of the record. Reading down a column gives the
usual one-year-ahead score at many different points in history. Reading across a row shows
what happens as the forecast reaches further from the last year of fire history it saw.

Every prediction uses that target year's own wet season, which is known by the end of
April, so a four-year-ahead forecast is still using weather nobody had to guess.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_class_weight

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pyrantis.schema import CLASSES, INTERIM, PROC, ROOT as PROJ
from scripts.train_models import FIRE_FEATURES, SEQ_LEN, build_sequences, BestMacroF1

SEED = 42
FIRST_CUTOFF = 2014          # leaves ten years of history before the first model trains
LAST_YEAR = 2025


def log(m=""):
    print(m, flush=True)


def scores(y, p):
    return {
        "accuracy": round(float(accuracy_score(y, p)), 4),
        "f1_macro": round(float(f1_score(y, p, average="macro", zero_division=0)), 4),
        "f1_late": round(float(f1_score(y, p, average=None, labels=[0, 1, 2],
                                        zero_division=0)[2]), 4),
        "confusion": confusion_matrix(y, p, labels=[0, 1, 2]).tolist(),
        "n": int(len(y)),
    }


def main():
    feat = pd.read_parquet(PROC / "cell_year_features_weather.parquet").reset_index(drop=True)
    labels = pd.read_parquet(INTERIM / "cell_year_labels.parquet")
    features = FIRE_FEATURES + [c for c in feat.columns if c.startswith("wx_")]

    y_all = feat["label"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    yr = feat["year"].to_numpy()
    X = feat[features].to_numpy("float32")
    seq = build_sequences(labels, feat)

    import tensorflow as tf
    from tensorflow import keras

    cutoffs = list(range(FIRST_CUTOFF, LAST_YEAR))
    out = {"cutoffs": cutoffs, "features": len(features), "folds": []}
    t_start = time.time()

    for C in cutoffs:
        tr = yr <= C - 1
        va = yr == C
        horizons = [h for h in range(C + 1, LAST_YEAR + 1)]
        log(f"\n=== cutoff {C}: train 2005-{C-1} ({tr.sum():,}), tune {C}, "
            f"predict {horizons[0]}-{horizons[-1]} ===")

        Xtr, ytr = X[tr], y_all[tr]
        scaler = StandardScaler().fit(Xtr)
        Str, Sva = scaler.transform(Xtr).astype("f4"), scaler.transform(X[va]).astype("f4")
        cw = compute_class_weight("balanced", classes=np.arange(3), y=ytr)
        cwd = dict(enumerate(cw))

        fitted = {}

        for name, mk in [
            ("decision tree", lambda: DecisionTreeClassifier(
                max_depth=12, min_samples_leaf=50, class_weight="balanced", random_state=SEED)),
            ("random forest", lambda: RandomForestClassifier(
                n_estimators=200, min_samples_leaf=5, n_jobs=-1,
                class_weight="balanced_subsample", random_state=SEED)),
            ("naive bayes", lambda: GaussianNB(priors=np.full(3, 1 / 3))),
        ]:
            t0 = time.time()
            m = mk(); m.fit(Xtr, ytr)
            fitted[name] = ("plain", m)
            log(f"  fitted {name} in {time.time()-t0:.0f}s")

        keras.utils.set_random_seed(SEED)
        ann = keras.Sequential([
            keras.layers.Input((len(features),)),
            keras.layers.Dense(128, activation="relu"), keras.layers.Dropout(0.2),
            keras.layers.Dense(64, activation="relu"), keras.layers.Dropout(0.2),
            keras.layers.Dense(3, activation="softmax")])
        ann.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
        t0 = time.time()
        ann.fit(Str, ytr, validation_data=(Sva, y_all[va]), epochs=40, batch_size=4096,
                verbose=0, class_weight=cwd, callbacks=[BestMacroF1(Sva, y_all[va])])
        fitted["multilayer ANN"] = ("scaled", ann)
        log(f"  fitted multilayer ANN in {time.time()-t0:.0f}s")

        keras.utils.set_random_seed(SEED)
        si = keras.layers.Input((SEQ_LEN, seq.shape[2])); ti = keras.layers.Input((len(features),))
        h = keras.layers.Dropout(0.2)(keras.layers.LSTM(64)(si))
        g = keras.layers.Dense(32, activation="relu")(ti)
        z = keras.layers.Dense(32, activation="relu")(keras.layers.Concatenate()([h, g]))
        lstm = keras.Model([si, ti], keras.layers.Dense(3, activation="softmax")(z))
        lstm.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
        t0 = time.time()
        lstm.fit([seq[tr], Str], ytr, validation_data=([seq[va], Sva], y_all[va]),
                 epochs=25, batch_size=4096, verbose=0, class_weight=cwd,
                 callbacks=[BestMacroF1([seq[va], Sva], y_all[va])])
        fitted["LSTM"] = ("seq", lstm)
        log(f"  fitted LSTM in {time.time()-t0:.0f}s")

        for H in horizons:
            te = yr == H
            yte = y_all[te]
            Ste = scaler.transform(X[te]).astype("f4")
            row = {"cutoff": C, "target": H, "horizon": H - C, "models": {}}

            # Baseline: whatever the cell did the year before the target.
            prev = np.zeros(te.sum(), dtype=int)
            sub = feat.loc[te]
            prev[sub["prev_late"].to_numpy() == 1] = 2
            prev[sub["prev_early"].to_numpy() == 1] = 1
            row["models"]["same as last year"] = scores(yte, prev)
            row["models"]["majority class"] = scores(
                yte, np.full(te.sum(), int(np.bincount(ytr).argmax())))

            for name, (kind, m) in fitted.items():
                if kind == "plain":
                    p = m.predict(X[te])
                elif kind == "scaled":
                    p = m.predict(Ste, verbose=0, batch_size=8192).argmax(1)
                else:
                    p = m.predict([seq[te], Ste], verbose=0, batch_size=8192).argmax(1)
                row["models"][name] = scores(yte, p)

            best = max((k for k in row["models"] if k not in
                        ("same as last year", "majority class")),
                       key=lambda k: row["models"][k]["f1_macro"])
            log(f"    {H} (+{H-C}):  best {best} "
                f"acc {row['models'][best]['accuracy']:.3f} "
                f"F1 {row['models'][best]['f1_macro']:.3f}  |  "
                f"last-year baseline F1 {row['models']['same as last year']['f1_macro']:.3f}")
            out["folds"].append(row)

        json.dump(out, open(PROJ / "reports" / "walk_forward.json", "w"), indent=1)

    log(f"\ndone in {(time.time()-t_start)/60:.0f} min, {len(out['folds'])} fold-years")

    df = pd.DataFrame([{"cutoff": f["cutoff"], "target": f["target"], "horizon": f["horizon"],
                        **{f"{k}": v["f1_macro"] for k, v in f["models"].items()}}
                       for f in out["folds"]])
    log("\nmacro-F1 by how far ahead the forecast reaches")
    log(df.groupby("horizon").mean(numeric_only=True).drop(columns=["cutoff", "target"])
          .round(3).to_string())


if __name__ == "__main__":
    main()
