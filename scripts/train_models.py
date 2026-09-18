"""Train and compare five classifiers on the NT fire-regime problem.

The split is by year, never at random. A cell is strongly correlated with itself across
time and with its neighbours across space, so a random split would put a cell's 2019 in
training and its neighbour's 2019 in test and report an accuracy the model could never
repeat in practice. Training is 2005-2019, validation 2020-2022, test 2023-2025, and the
test years are not consulted until the end.

Two baselines sit alongside the models, because accuracy on a problem that is 69%
unburnt means very little on its own: always predicting the majority class, and
predicting whatever the cell did last year. Persistence is the one that matters. A model
that cannot beat "the same as last year" has learned nothing about fire.

Every model is run twice, once as-is and once with classes weighted inversely to their
frequency. Unweighted models buy accuracy by under-calling the two fire classes, which
are the only classes anyone cares about, so the pair is reported rather than the flattering
half of it.

The sequence model is handed the raw year-by-year history instead of the engineered
summaries. That makes the comparison worth something: it asks whether an LSTM can recover
from the sequence itself what the other models are given pre-computed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_class_weight

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pyrantis.schema import CLASSES, INTERIM, PROC, ROOT as PROJ

SEED = 42
TRAIN_YEARS = range(2005, 2020)
VAL_YEARS = range(2020, 2023)
TEST_YEARS = range(2023, 2026)
SEQ_LEN = 10

# Named explicitly rather than derived by exclusion, so that adding a column to the
# feature table can never silently add a predictor. frac_burnt, frac_early and frac_late
# describe the target year itself and must never appear here.
FIRE_FEATURES = [
    "freq_3", "freq_5", "freq_10",
    "late_freq_3", "late_freq_5", "late_freq_10",
    "yrs_since_burnt", "never_burnt",
    "yrs_since_late", "never_late",
    "prev_burnt", "prev_early", "prev_late",
    "prev_frac_burnt", "prev_frac_early", "prev_frac_late",
    "prev2_burnt",
    "nbr_frac_burnt_prev", "nbr_frac_early_prev", "nbr_frac_late_prev",
    "lon", "lat",
]
FORBIDDEN = {"frac_burnt", "frac_early", "frac_late", "label"}


def log(m: str = "") -> None:
    print(m, flush=True)


def build_sequences(labels: pd.DataFrame, feat: pd.DataFrame) -> np.ndarray:
    """Per cell-year, the previous SEQ_LEN years of observed fire state.

    Six channels a year: burnt, early and late fractions, and the three classes as
    indicators. The sequence runs oldest-first and stops one year short of the target.
    """
    years = np.sort(labels["year"].unique())
    nrow, ncol = labels["row"].max() + 1, labels["col"].max() + 1
    yidx = {y: i for i, y in enumerate(years)}

    chan = np.full((len(years), nrow, ncol, 6), np.nan, dtype="float32")
    yi = labels["year"].map(yidx).to_numpy()
    r, c = labels["row"].to_numpy(), labels["col"].to_numpy()
    lab = labels["label"].to_numpy()
    for k, col in enumerate(("frac_burnt", "frac_early", "frac_late")):
        chan[yi, r, c, k] = labels[col].to_numpy()
    for k, cls in enumerate(CLASSES):
        chan[yi, r, c, 3 + k] = (lab == cls).astype("f4")

    out = np.zeros((len(feat), SEQ_LEN, 6), dtype="float32")
    fr, fc = feat["row"].to_numpy(), feat["col"].to_numpy()
    fy = feat["year"].map(yidx).to_numpy()
    for pos in range(SEQ_LEN):
        src = fy - (SEQ_LEN - pos)            # oldest first, never the target year
        ok = src >= 0
        out[ok, pos] = chan[src[ok], fr[ok], fc[ok]]
    return np.nan_to_num(out, nan=0.0)


def score(name: str, y_true, y_pred, seconds: float) -> dict:
    lbl = list(range(len(CLASSES)))
    return {
        "model": name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_per_class": dict(zip(CLASSES, [
            float(f) for f in f1_score(y_true, y_pred, average=None, labels=lbl, zero_division=0)])),
        "confusion": confusion_matrix(y_true, y_pred, labels=lbl).tolist(),
        "train_seconds": round(seconds, 1),
    }


class BestMacroF1(object):
    """Early stopping on validation macro-F1 rather than loss.

    The validation years are quiet ones, so cross-entropy on them rises from the first
    epoch and loss-based stopping cuts training off almost immediately. Macro-F1 keeps
    scoring the thing we actually care about -- whether the two fire classes are being
    called -- and does not collapse when the class mix shifts between periods.
    """

    def __new__(cls, Xva, yva, patience=6):
        from tensorflow import keras

        class _CB(keras.callbacks.Callback):
            def on_train_begin(self, logs=None):
                self.best, self.wait, self.weights, self.best_epoch = -1.0, 0, None, 0

            def on_epoch_end(self, epoch, logs=None):
                p = self.model.predict(Xva, verbose=0, batch_size=8192).argmax(1)
                f = f1_score(yva, p, average="macro", zero_division=0)
                logs = logs or {}
                logs["val_macro_f1"] = f
                if f > self.best:
                    self.best, self.wait = f, 0
                    self.weights, self.best_epoch = self.model.get_weights(), epoch
                else:
                    self.wait += 1
                    if self.wait >= patience:
                        self.model.stop_training = True

            def on_train_end(self, logs=None):
                if self.weights is not None:
                    self.model.set_weights(self.weights)
                log(f"    best val macro-F1 {self.best:.3f} at epoch {self.best_epoch + 1}")

        return _CB()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weather", action="store_true",
                    help="include the SILO weather features")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    # One tag for every artefact this run writes, so the fire-history and weather runs
    # never overwrite each other's importances or results.
    tag = args.tag or ("_weather" if args.weather else "_firehistory")

    path = PROC / ("cell_year_features_weather.parquet" if args.weather
                   else "cell_year_features.parquet")
    feat = pd.read_parquet(path).reset_index(drop=True)
    labels = pd.read_parquet(INTERIM / "cell_year_labels.parquet")

    features = list(FIRE_FEATURES)
    if args.weather:
        wx = [c for c in feat.columns if c.startswith("wx_")]
        assert wx, "no weather columns found in the feature table"
        features += wx
        log(f"including {len(wx)} weather features")

    assert not (set(features) & FORBIDDEN), "a target-year column is in the feature list"
    missing = [f for f in features if f not in feat.columns]
    assert not missing, f"features missing from the table: {missing}"

    y_all = feat["label"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    tr = feat["year"].isin(TRAIN_YEARS).to_numpy()
    va = feat["year"].isin(VAL_YEARS).to_numpy()
    te = feat["year"].isin(TEST_YEARS).to_numpy()

    log(f"{len(features)} features   train {tr.sum():,}  val {va.sum():,}  test {te.sum():,}")
    log(f"train {min(TRAIN_YEARS)}-{max(TRAIN_YEARS)}   val {min(VAL_YEARS)}-{max(VAL_YEARS)}"
        f"   test {min(TEST_YEARS)}-{max(TEST_YEARS)}")

    X = feat[features].to_numpy("float32")
    Xtr, Xva, Xte = X[tr], X[va], X[te]
    ytr, yva, yte = y_all[tr], y_all[va], y_all[te]

    results = []

    def add(r):
        results.append(r)
        log(f"  {r['model']:<34} acc {r['accuracy']:.3f}  macro-F1 {r['f1_macro']:.3f}"
            f"  ({r['train_seconds']}s)")

    log("\nbaselines")
    add(score("majority class", yte,
              np.full_like(yte, int(np.bincount(ytr).argmax())), 0.0))
    prev = np.zeros(len(feat), dtype=int)
    prev[feat["prev_late"].to_numpy() == 1] = CLASSES.index("late")
    prev[feat["prev_early"].to_numpy() == 1] = CLASSES.index("early")
    add(score("same as last year", yte, prev[te], 0.0))

    log("\nclassical models")
    specs = [
        ("decision tree", lambda w: DecisionTreeClassifier(
            max_depth=12, min_samples_leaf=50, class_weight=w, random_state=SEED)),
        ("random forest", lambda w: RandomForestClassifier(
            n_estimators=200, min_samples_leaf=5, n_jobs=-1,
            class_weight=("balanced_subsample" if w else None), random_state=SEED)),
        ("naive bayes", lambda w: GaussianNB(
            priors=(np.full(len(CLASSES), 1 / len(CLASSES)) if w else None))),
    ]
    for name, make in specs:
        for weighted in (False, True):
            m = make("balanced" if weighted else None)
            t0 = time.time()
            m.fit(Xtr, ytr)
            r = score(f"{name}{' (balanced)' if weighted else ''}",
                      yte, m.predict(Xte), time.time() - t0)
            add(r)
            if name == "random forest" and weighted:
                imp = sorted(zip(features, m.feature_importances_), key=lambda t: -t[1])
                json.dump({k: float(v) for k, v in imp},
                          open(PROJ / "reports" / f"feature_importance{tag}.json", "w"),
                          indent=2)
                log("    top: " + ", ".join(f"{k} {v:.3f}" for k, v in imp[:6]))

    import tensorflow as tf
    from tensorflow import keras

    tf.keras.utils.set_random_seed(SEED)
    scaler = StandardScaler().fit(Xtr)               # fitted on training years only
    Str, Sva, Ste = (scaler.transform(a).astype("float32") for a in (Xtr, Xva, Xte))

    cw = compute_class_weight("balanced", classes=np.arange(len(CLASSES)), y=ytr)
    cw = dict(enumerate(cw))

    def fit_nn(build, Atr, Ava, Ate, name, weighted, epochs):
        keras.utils.set_random_seed(SEED)
        m = build()
        m.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
        t0 = time.time()
        m.fit(Atr, ytr, validation_data=(Ava, yva), epochs=epochs, batch_size=4096,
              verbose=0, class_weight=(cw if weighted else None),
              callbacks=[BestMacroF1(Ava, yva)])
        add(score(f"{name}{' (balanced)' if weighted else ''}", yte,
                  m.predict(Ate, verbose=0, batch_size=8192).argmax(1), time.time() - t0))

    log("\nmultilayer ANN")
    def ann():
        return keras.Sequential([
            keras.layers.Input((len(features),)),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(len(CLASSES), activation="softmax"),
        ])
    for w in (False, True):
        fit_nn(ann, Str, Sva, Ste, "multilayer ANN", w, epochs=40)

    log("\nLSTM over the year sequences")
    seq = build_sequences(labels, feat)
    log(f"  sequence tensor {seq.shape}, {seq.nbytes / 1e6:.0f} MB")
    Qtr, Qva, Qte = seq[tr], seq[va], seq[te]

    # Two inputs, not one. The recurrent branch reads the year-by-year fire history; the
    # dense branch reads the same static columns every other model gets, weather included.
    # Without the second branch the LSTM would be the only model judged blind to weather,
    # and its score would say more about what it was withheld than about the architecture.
    def lstm():
        seq_in = keras.layers.Input((SEQ_LEN, seq.shape[2]), name="history")
        sta_in = keras.layers.Input((len(features),), name="static")
        h = keras.layers.LSTM(64)(seq_in)
        h = keras.layers.Dropout(0.2)(h)
        g = keras.layers.Dense(32, activation="relu")(sta_in)
        z = keras.layers.Concatenate()([h, g])
        z = keras.layers.Dense(32, activation="relu")(z)
        out = keras.layers.Dense(len(CLASSES), activation="softmax")(z)
        return keras.Model([seq_in, sta_in], out)

    for w in (False, True):
        fit_nn(lstm, [Qtr, Str], [Qva, Sva], [Qte, Ste], "LSTM", w, epochs=25)

    out = PROJ / "reports" / f"results{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({
        "split": {"train": list(TRAIN_YEARS), "val": list(VAL_YEARS), "test": list(TEST_YEARS)},
        "features": features,
        "weather_included": bool(args.weather),
        "class_order": list(CLASSES),
        "results": results,
    }, open(out, "w"), indent=2)

    log("\n" + "=" * 88)
    log(f"{'model':<34}{'acc':>7}{'mP':>8}{'mR':>8}{'mF1':>8}{'F1 unburnt':>12}"
        f"{'F1 early':>10}{'F1 late':>9}")
    log("-" * 88)
    for r in sorted(results, key=lambda r: -r["f1_macro"]):
        p = r["f1_per_class"]
        log(f"{r['model']:<34}{r['accuracy']:>7.3f}{r['precision_macro']:>8.3f}"
            f"{r['recall_macro']:>8.3f}{r['f1_macro']:>8.3f}"
            f"{p['unburnt']:>12.3f}{p['early']:>10.3f}{p['late']:>9.3f}")
    log("=" * 88)
    log(f"\nwrote {out.relative_to(PROJ)}")


if __name__ == "__main__":
    main()
