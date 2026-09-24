"""Robustness checks the headline comparison did not make.

  1. overfitting   -- macro-F1 on the training, validation and test years for every model
  2. ranking       -- ROC-AUC and PR-AUC on the held-out years, from predicted probabilities
  3. importance    -- permutation importance on the held-out years, for the random forest
                      (to check its built-in impurity ranking) and for gradient boosting
  4. tuning        -- a randomised or grid search for every model, scored on the
                      validation years only, then refit and tested once
  5. random CV     -- repeated stratified 5-fold cross-validation on the training years,
                      set against the forward-in-time score, to show why the project
                      does not use it

Everything reuses the definitions in train_models.py, so the models here are the models
in the report. The test years are only ever used to score a model that has already been
chosen. Results are written after each stage, so a slow stage never loses earlier ones.
"""
from __future__ import annotations

import json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (average_precision_score, f1_score, make_scorer,
                             roc_auc_score)
from sklearn.model_selection import (PredefinedSplit, RandomizedSearchCV, GridSearchCV,
                                     RepeatedStratifiedKFold, cross_val_score)
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_class_weight

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_models as T
from pyrantis.schema import CLASSES, INTERIM, PROC, ROOT as PROJ

OUT = PROJ / "reports" / "robustness.json"
RNG = np.random.default_rng(T.SEED)
R = json.loads(OUT.read_text()) if OUT.exists() else {}


def save():
    OUT.write_text(json.dumps(R, indent=2))


def log(m=""):
    print(time.strftime("%H:%M:%S"), m, flush=True)


F1 = make_scorer(f1_score, average="macro", zero_division=0)
mf1 = lambda y, p: float(f1_score(y, p, average="macro", zero_division=0))

# ---------------------------------------------------------------- data, as --full
feat = pd.read_parquet(PROC / "cell_year_features_full.parquet").reset_index(drop=True)
labels = pd.read_parquet(INTERIM / "cell_year_labels.parquet")
features = list(T.FIRE_FEATURES)
features += [c for c in feat.columns if c.startswith("wx_")]
features += [c for c in feat.columns if (c.startswith("ndvi_") and c not in T.EXTRA_EXCLUDE)
             or c in T.EXTRA_NAMED]
assert len(features) == 50 and not (set(features) & T.FORBIDDEN)
y = feat["label"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
tr = feat["year"].isin(T.TRAIN_YEARS).to_numpy()
va = feat["year"].isin(T.VAL_YEARS).to_numpy()
te = feat["year"].isin(T.TEST_YEARS).to_numpy()
X = feat[features].to_numpy("float32")
Xtr, Xva, Xte, ytr, yva, yte = X[tr], X[va], X[te], y[tr], y[va], y[te]
log(f"{len(features)} features  train {tr.sum():,}  val {va.sum():,}  test {te.sum():,}")


def rank_metrics(yt, P):
    out = {"roc_auc_macro": float(roc_auc_score(yt, P, multi_class="ovr", average="macro"))}
    aps = []
    for k, c in enumerate(CLASSES):
        out[f"roc_auc_{c}"] = float(roc_auc_score(yt == k, P[:, k]))
        ap = float(average_precision_score(yt == k, P[:, k]))
        out[f"pr_auc_{c}"] = ap
        out[f"prevalence_{c}"] = float((yt == k).mean())
        aps.append(ap)
    out["pr_auc_macro"] = float(np.mean(aps))
    return out


# the balanced models exactly as train_models.py builds them
CLASSICAL = {
    "decision tree": DecisionTreeClassifier(max_depth=12, min_samples_leaf=50,
                                            class_weight="balanced", random_state=T.SEED),
    "random forest": RandomForestClassifier(n_estimators=200, min_samples_leaf=5, n_jobs=-1,
                                            class_weight="balanced_subsample", random_state=T.SEED),
    "naive bayes": GaussianNB(priors=np.full(len(CLASSES), 1 / len(CLASSES))),
    "gradient boosting": HistGradientBoostingClassifier(
        max_iter=500, learning_rate=0.07, max_leaf_nodes=63, l2_regularization=1.0,
        min_samples_leaf=40, early_stopping=False, class_weight="balanced", random_state=T.SEED),
}

# ================================================================ 1 + 2 + 3
if "models" not in R:
    R["models"] = {}
fitted = {}
for name, est in CLASSICAL.items():
    if name in R["models"] and name not in ("random forest", "gradient boosting"):
        continue
    log(f"[1-2] {name}")
    m = clone(est); t0 = time.time(); m.fit(Xtr, ytr); fitted[name] = m
    Ptr, Pva, Pte = m.predict_proba(Xtr), m.predict_proba(Xva), m.predict_proba(Xte)
    R["models"][name] = {
        "f1_train": mf1(ytr, Ptr.argmax(1)), "f1_val": mf1(yva, Pva.argmax(1)),
        "f1_test": mf1(yte, Pte.argmax(1)), **rank_metrics(yte, Pte),
        "seconds": round(time.time() - t0, 1)}
    log(f"      train {R['models'][name]['f1_train']:.3f}  val {R['models'][name]['f1_val']:.3f}"
        f"  test {R['models'][name]['f1_test']:.3f}  ROC-AUC {R['models'][name]['roc_auc_macro']:.3f}"
        f"  PR-AUC {R['models'][name]['pr_auc_macro']:.3f}")
    save()

# permutation importance on a fixed sample of the held-out years
if "permutation" not in R:
    idx = RNG.choice(np.flatnonzero(te), 40000, replace=False)
    Xp, yp = X[idx], y[idx]
    R["permutation"] = {}
    for name in ("random forest", "gradient boosting"):
        log(f"[3] permutation importance: {name}")
        pi = permutation_importance(fitted[name], Xp, yp, scoring=F1, n_repeats=5,
                                    random_state=T.SEED, n_jobs=1)
        order = np.argsort(-pi.importances_mean)
        R["permutation"][name] = [{"feature": features[i], "mean": float(pi.importances_mean[i]),
                                   "sd": float(pi.importances_std[i])} for i in order]
        log("      top: " + ", ".join(f"{features[i]} {pi.importances_mean[i]:.3f}" for i in order[:6]))
        save()
    imp = sorted(zip(features, fitted["random forest"].feature_importances_), key=lambda t: -t[1])
    R["impurity_random_forest"] = [{"feature": k, "importance": float(v)} for k, v in imp]
    save()

# neural networks, as train_models.py builds them, with probabilities kept
import tensorflow as tf
from tensorflow import keras
scaler = StandardScaler().fit(Xtr)
Str, Sva, Ste = (scaler.transform(a).astype("float32") for a in (Xtr, Xva, Xte))
cw = dict(enumerate(compute_class_weight("balanced", classes=np.arange(len(CLASSES)), y=ytr)))
seq = T.build_sequences(labels, feat)
Qtr, Qva, Qte = seq[tr], seq[va], seq[te]


def ann_builder(units=(128, 64), drop=0.2):
    def b():
        L = [keras.layers.Input((len(features),))]
        for u in units:
            L += [keras.layers.Dense(u, activation="relu"), keras.layers.Dropout(drop)]
        L += [keras.layers.Dense(len(CLASSES), activation="softmax")]
        return keras.Sequential(L)
    return b


def lstm_builder(units=64, drop=0.2):
    def b():
        s_in = keras.layers.Input((T.SEQ_LEN, seq.shape[2])); x_in = keras.layers.Input((len(features),))
        h = keras.layers.Dropout(drop)(keras.layers.LSTM(units)(s_in))
        g = keras.layers.Dense(32, activation="relu")(x_in)
        z = keras.layers.Dense(32, activation="relu")(keras.layers.Concatenate()([h, g]))
        return keras.Model([s_in, x_in], keras.layers.Dense(len(CLASSES), activation="softmax")(z))
    return b


def fit_nn(builder, Atr, Ava, epochs, lr=1e-3):
    keras.utils.set_random_seed(T.SEED)
    m = builder()
    m.compile(optimizer=keras.optimizers.Adam(lr), loss="sparse_categorical_crossentropy")
    cb = T.BestMacroF1(Ava, yva)
    m.fit(Atr, ytr, validation_data=(Ava, yva), epochs=epochs, batch_size=4096, verbose=0,
          class_weight=cw, callbacks=[cb])
    return m, cb


for name, builder, A, epochs in (("multilayer ANN", ann_builder(), (Str, Sva, Ste), 40),
                                 ("LSTM", lstm_builder(), ([Qtr, Str], [Qva, Sva], [Qte, Ste]), 25)):
    if name in R["models"]:
        continue
    log(f"[1-2] {name}")
    t0 = time.time()
    m, cb = fit_nn(builder, A[0], A[1], epochs)
    P = [m.predict(a, verbose=0, batch_size=8192) for a in A]
    R["models"][name] = {"f1_train": mf1(ytr, P[0].argmax(1)), "f1_val": mf1(yva, P[1].argmax(1)),
                         "f1_test": mf1(yte, P[2].argmax(1)), **rank_metrics(yte, P[2]),
                         "best_epoch": int(cb.best_epoch + 1), "seconds": round(time.time() - t0, 1)}
    r = R["models"][name]
    log(f"      train {r['f1_train']:.3f}  val {r['f1_val']:.3f}  test {r['f1_test']:.3f}"
        f"  ROC-AUC {r['roc_auc_macro']:.3f}  PR-AUC {r['pr_auc_macro']:.3f}")
    save()

# ================================================================ 4. tuning
# Fit on (a sample of) the training years, score on the validation years, and nothing
# else: PredefinedSplit gives exactly that single forward-in-time fold.
R.setdefault("tuning", {})
itr = RNG.choice(np.flatnonzero(tr), 200000, replace=False)
iva = RNG.choice(np.flatnonzero(va), 100000, replace=False)
Xs = np.vstack([X[itr], X[iva]]); ys = np.concatenate([y[itr], y[iva]])
split = PredefinedSplit(np.r_[np.full(len(itr), -1), np.zeros(len(iva), int)])

SPACES = {
    "decision tree": ("random", 25, {"max_depth": [6, 8, 10, 12, 15, 20, None],
                                     "min_samples_leaf": [10, 25, 50, 100, 250, 500],
                                     "criterion": ["gini", "entropy"]}),
    "naive bayes": ("grid", None, {"var_smoothing": list(np.logspace(-12, -2, 11))}),
    "random forest": ("random", 10, {"n_estimators": [100, 200, 400],
                                     "max_depth": [None, 20, 30],
                                     "min_samples_leaf": [1, 5, 20, 50],
                                     "max_features": ["sqrt", 0.3, 0.5]}),
    "gradient boosting": ("random", 14, {"learning_rate": [0.03, 0.05, 0.07, 0.1],
                                         "max_leaf_nodes": [31, 63, 127],
                                         "min_samples_leaf": [20, 40, 100],
                                         "l2_regularization": [0.0, 1.0, 3.0],
                                         "max_iter": [300, 500, 800]}),
}
for name, (kind, n_iter, space) in SPACES.items():
    if name in R["tuning"]:
        continue
    log(f"[4] tuning {name} ({kind} search)")
    base = clone(CLASSICAL[name])
    if kind == "grid":
        s = GridSearchCV(base, space, scoring=F1, cv=split, n_jobs=1, refit=False)
    else:
        s = RandomizedSearchCV(base, space, n_iter=n_iter, scoring=F1, cv=split,
                               random_state=T.SEED, n_jobs=(1 if name == "random forest" else -1),
                               refit=False)
    t0 = time.time(); s.fit(Xs, ys)
    best = s.best_params_
    final = clone(CLASSICAL[name]).set_params(**best).fit(Xtr, ytr)
    R["tuning"][name] = {
        "search": kind, "candidates": len(s.cv_results_["params"]),
        "best_params": {k: (v if not isinstance(v, np.generic) else v.item()) for k, v in best.items()},
        "val_f1_best": float(s.best_score_),
        "test_f1_tuned": mf1(yte, final.predict(Xte)),
        "test_f1_manual": R["models"][name]["f1_test"],
        "seconds": round(time.time() - t0, 1)}
    t = R["tuning"][name]
    log(f"      best {t['best_params']}  val {t['val_f1_best']:.3f}  "
        f"test tuned {t['test_f1_tuned']:.3f} vs manual {t['test_f1_manual']:.3f}")
    save()

NN_GRID = {
    "multilayer ANN": [dict(units=u, drop=d, lr=l) for u in ((64, 32), (128, 64), (256, 128))
                       for d in (0.2, 0.4) for l in (1e-3,)] + [dict(units=(128, 64), drop=0.2, lr=3e-4)],
    "LSTM": [dict(units=u, drop=d, lr=1e-3) for u in (32, 64, 96) for d in (0.2,)] + [dict(units=64, drop=0.4, lr=1e-3)],
}
for name, grid in NN_GRID.items():
    if name in R["tuning"]:
        continue
    log(f"[4] tuning {name} (grid of {len(grid)})")
    t0 = time.time(); trials = []
    for g in grid:
        if name == "LSTM":
            b = lstm_builder(g["units"], g["drop"]); A = ([Qtr, Str], [Qva, Sva], [Qte, Ste]); ep = 25
        else:
            b = ann_builder(g["units"], g["drop"]); A = (Str, Sva, Ste); ep = 40
        m, cb = fit_nn(b, A[0], A[1], ep, g["lr"])
        trials.append((cb.best, g, m, A))
        log(f"      {g}  val {cb.best:.3f}")
    bval, bg, bm, A = max(trials, key=lambda t: t[0])
    R["tuning"][name] = {
        "search": "grid", "candidates": len(grid),
        "best_params": {k: (list(v) if isinstance(v, tuple) else v) for k, v in bg.items()},
        "val_f1_best": float(bval),
        "test_f1_tuned": mf1(yte, bm.predict(A[2], verbose=0, batch_size=8192).argmax(1)),
        "test_f1_manual": R["models"][name]["f1_test"], "seconds": round(time.time() - t0, 1)}
    save()

# ================================================================ 5. random k-fold
if "random_cv" not in R:
    log("[5] repeated stratified 5-fold CV on the training years (gradient boosting)")
    isub = RNG.choice(np.flatnonzero(tr), 240000, replace=False)
    gb = clone(CLASSICAL["gradient boosting"])
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=T.SEED)
    sc = cross_val_score(gb, X[isub], y[isub], scoring=F1, cv=cv, n_jobs=1)
    fwd = clone(CLASSICAL["gradient boosting"]).fit(X[isub], y[isub])
    R["random_cv"] = {"folds": len(sc), "mean": float(sc.mean()), "sd": float(sc.std(ddof=1)),
                      "forward_val_f1": mf1(yva, fwd.predict(Xva)),
                      "forward_test_f1": mf1(yte, fwd.predict(Xte)), "rows": len(isub)}
    log(f"      random CV {sc.mean():.3f} +/- {sc.std(ddof=1):.3f}   vs forward: val "
        f"{R['random_cv']['forward_val_f1']:.3f}  test {R['random_cv']['forward_test_f1']:.3f}")
    save()

log("done")
