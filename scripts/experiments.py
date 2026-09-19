"""One change at a time, measured against a fixed benchmark, kept only if it helps.

The rule this file exists to enforce: nothing is adopted because it sounds like it should
work. Every idea is a run, every run is scored on the same held-out years with the same
metric, and the result is written down next to the incumbent whether it won or lost. A
change that does not beat the incumbent is recorded as a loss and the incumbent stands.

The benchmark is fixed: train 2005-2019, tune 2020-2022, score once on 2023-2025, ranked
on macro-averaged F1 so the rare late-season class counts as much as the common unburnt
one. Accuracy is reported too, but a model cannot win on it -- on a problem that is 69%
unburnt, accuracy rewards a model for ignoring fire.

Run `python scripts/experiments.py --list` to see the queue, or pass names to run a subset.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pyrantis.schema import CLASSES, INTERIM, PROC, RAW, ROOT as PROJ
from scripts.build_features import to_grid
from scripts.train_models import (FIRE_FEATURES, SEQ_LEN, TRAIN_YEARS, VAL_YEARS,
                                  TEST_YEARS, build_sequences, BestMacroF1)

SEED = 42
LEDGER = PROJ / "reports" / "experiments.json"
INCUMBENT = 0.6740          # LSTM (balanced), reported in results_weather.json

MONTHS_IN_SEASON = 6
MONTH_VARS = ["rain", "tmax", "vpd", "radn"]


def log(m=""):
    print(m, flush=True)


# =============================================================== feature blocks
def block_longmem(feat, labels):
    """Fire frequency over 15, 20 and 25 years. The record is 26 years long and the
    existing features look back at most 10 of them."""
    years = np.sort(labels["year"].unique())
    nrow, ncol = labels["row"].max() + 1, labels["col"].max() + 1
    l = labels["label"].to_numpy()
    burnt = to_grid(labels.assign(v=(l != "unburnt").astype("f4")), "v", nrow, ncol, years)
    late = to_grid(labels.assign(v=(l == "late").astype("f4")), "v", nrow, ncol, years)
    yidx = {y: i for i, y in enumerate(years)}
    fr, fc = feat["row"].to_numpy(), feat["col"].to_numpy()
    fy = feat["year"].map(yidx).to_numpy()
    out = {}
    for w in (15, 20, 25):
        a = np.full(len(feat), np.nan, "float32")
        b = np.full(len(feat), np.nan, "float32")
        for t in np.unique(fy):
            sel = fy == t
            lo = max(0, t - w)
            n = t - lo
            if n <= 0:
                continue
            a[sel] = np.nansum(burnt[lo:t], axis=0)[fr[sel], fc[sel]] / n
            b[sel] = np.nansum(late[lo:t], axis=0)[fr[sel], fc[sel]] / n
        out[f"freq_{w}"], out[f"late_freq_{w}"] = a, b
    return pd.DataFrame(out, index=feat.index)


def block_elevation(feat, labels):
    """Elevation, which every SILO point file has carried all along."""
    pts = []
    for d in ("silo2026", "silo"):
        files = sorted((RAW / d).glob("silo_*.txt"))
        if files:
            break
    for f in files:
        m = re.match(r"silo_([+-][\d.]+)_([+-][\d.]+)\.txt", f.name)
        e = re.search(r"Elevation:\s*([-\d.]+)\s*m", f.read_text(errors="replace")[:4000])
        if e:
            pts.append((float(m.group(2)), float(m.group(1)), float(e.group(1))))
    from scipy.interpolate import griddata
    src = np.array([[p[0], p[1]] for p in pts])
    val = np.array([p[2] for p in pts])
    cells = feat[["cell_id", "lon", "lat"]].drop_duplicates("cell_id")
    z = griddata(src, val, cells[["lon", "lat"]].to_numpy(), method="linear")
    gap = ~np.isfinite(z)
    if gap.any():
        z[gap] = griddata(src, val, cells[["lon", "lat"]].to_numpy()[gap], method="nearest")
    m = dict(zip(cells["cell_id"].to_numpy(), z))
    return pd.DataFrame({"elevation": feat["cell_id"].map(m).astype("float32")},
                        index=feat.index)


def block_spatial(feat, labels):
    """What the wider district did, at two scales and two lags.

    The existing neighbour features look one cell out and one year back. Fire in this
    country runs over tens of kilometres and burning programmes are planned across whole
    districts, so a 5 km ring is too tight a window to see either.
    """
    years = np.sort(labels["year"].unique())
    nrow, ncol = labels["row"].max() + 1, labels["col"].max() + 1
    yidx = {y: i for i, y in enumerate(years)}
    grids = {c: to_grid(labels, c, nrow, ncol, years)
             for c in ("frac_burnt", "frac_early", "frac_late")}
    inside = ~np.isnan(grids["frac_burnt"][0])

    def boxmean(a, size):
        # Cells outside the Territory are excluded from both sums, so a coastal cell
        # averages over the land actually around it rather than over empty sea.
        filled = np.nan_to_num(a, nan=0.0)
        num = ndimage.uniform_filter(filled, size, mode="constant")
        den = ndimage.uniform_filter(inside.astype("float32"), size, mode="constant")
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(den > 0.05, num / den, np.nan).astype("float32")

    fr, fc = feat["row"].to_numpy(), feat["col"].to_numpy()
    fy = feat["year"].map(yidx).to_numpy()
    out = {}
    for var, short in (("frac_burnt", "burnt"), ("frac_early", "early"), ("frac_late", "late")):
        for size, tag in ((5, "5"), (11, "11")):
            for lag in (1, 2):
                col = np.full(len(feat), np.nan, "float32")
                for t in np.unique(fy):
                    if t - lag < 0:
                        continue
                    sel = fy == t
                    col[sel] = boxmean(grids[var][t - lag], size)[fr[sel], fc[sel]]
                out[f"nbr{tag}_{short}_lag{lag}"] = col
    return pd.DataFrame(out, index=feat.index)


def _monthly(feat):
    mp = PROC / "weather_monthly.parquet"
    assert mp.exists(), "run scripts/build_weather_monthly.py first"
    m = pd.read_parquet(mp)
    j = feat[["cell_id", "year"]].merge(m, on=["cell_id", "year"], how="left")
    return j.set_index(feat.index)


def block_monthly(feat, labels):
    """Each wet-season month on its own, rather than folded into a seasonal mean."""
    j = _monthly(feat)
    cols = [f"m{k}_{v}" for k in range(MONTHS_IN_SEASON) for v in MONTH_VARS]
    return j[cols].astype("float32")


def block_growth(feat, labels):
    """How the wet season was shaped, which is what decides how much grass it grew.

    Six hundred millimetres spread over six months grows a standing crop; the same
    falling in one week runs off. These describe the shape rather than the total.
    """
    j = _monthly(feat)
    rain = np.stack([j[f"m{k}_rain"].to_numpy() for k in range(MONTHS_IN_SEASON)])
    vpd = np.stack([j[f"m{k}_vpd"].to_numpy() for k in range(MONTHS_IN_SEASON)])
    tot = np.nansum(rain, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        even = np.where(tot > 0, 1 - np.nanstd(rain, axis=0) / (np.nanmean(rain, axis=0) + 1e-6), 0)
    out = {
        # Growth saturates: past a point more rain in a month does not grow more grass.
        "growth_index": np.nansum(np.minimum(rain, 150.0), axis=0).astype("float32"),
        "wet_months": (rain >= 50).sum(axis=0).astype("float32"),
        "rain_evenness": np.clip(even, -3, 1).astype("float32"),
        # The last two months of the wet season set how late the country stays damp.
        "late_wet_rain": (rain[4] + rain[5]).astype("float32"),
        "first_wet_rain": (rain[0] + rain[1]).astype("float32"),
        "vpd_rise": (vpd[5] - vpd[0]).astype("float32"),
    }
    return pd.DataFrame(out, index=feat.index)


def block_soi(feat, labels):
    """Southern Oscillation Index over the wet season.

    One number a year, so it can only tell the model which kind of year this was. It is
    also close to a summary of the rainfall already in the features, which is exactly why
    it is worth testing rather than assuming.
    """
    src = PROJ / "data" / "raw" / "soi.txt"
    if not src.exists():
        import urllib.request
        src.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve("https://www.cpc.ncep.noaa.gov/data/indices/soi", src)
    rows = {}
    for line in src.read_text().splitlines():
        m = re.match(r"^(\d{4})((?:\s+-?\d+\.\d+){12})\s*$", line)
        if m:
            rows[int(m.group(1))] = [float(v) for v in m.group(2).split()]
    # Wet season means November and December of the previous year plus January to April.
    val = {}
    for y in feat["year"].unique():
        a = rows.get(int(y) - 1, [np.nan] * 12)[10:12]
        b = rows.get(int(y), [np.nan] * 12)[0:4]
        val[y] = float(np.nanmean(a + b)) if (a + b) else np.nan
    return pd.DataFrame({"soi_wet": feat["year"].map(val).astype("float32")},
                        index=feat.index)


def block_ndvi(feat, labels):
    """Satellite greenness: the fuel itself, rather than the rain that grew it.

    Everything the models have been told about fuel so far is rainfall in disguise. This
    is MODIS looking at how green the ground actually went over the wet season, which is
    what will or will not carry a fire in August. April greenness is the standing crop as
    the dry season opens; the anomaly compares a cell with its own normal rather than
    with the desert; green-up is the growth itself rather than the standing total.
    """
    np_path = PROC / "ndvi.parquet"
    assert np_path.exists(), "run scripts/fetch_ndvi.py first"
    n = pd.read_parquet(np_path)
    cols = ["ndvi_apr", "ndvi_wet_mean", "ndvi_wet_max", "ndvi_wet_min",
            "ndvi_apr_anom", "ndvi_peak_anom", "ndvi_greenup"]
    j = feat[["cell_id", "year"]].merge(n[["cell_id", "year"] + cols],
                                        on=["cell_id", "year"], how="left")
    return j[cols].astype("float32").set_index(feat.index)


BLOCKS = {"longmem": block_longmem, "elevation": block_elevation,
          "spatial": block_spatial, "monthly": block_monthly,
          "growth": block_growth, "soi": block_soi, "ndvi": block_ndvi}


# ===================================================================== models
def fit_histgb(X, y, tr, va, te, seed=SEED, **kw):
    p = dict(max_iter=500, learning_rate=0.07, max_leaf_nodes=63,
             l2_regularization=1.0, min_samples_leaf=40, early_stopping=False,
             class_weight="balanced", random_state=seed)
    p.update(kw)
    m = HistGradientBoostingClassifier(**p)
    m.fit(X[tr], y[tr])
    return m.predict_proba(X[te])


def fit_lstm(X, y, tr, va, te, seq, seq2=None, units=64, seed=SEED):
    import tensorflow as tf
    from tensorflow import keras
    keras.utils.set_random_seed(seed)

    sc = StandardScaler().fit(X[tr])
    S = {k: sc.transform(X[m]).astype("f4") for k, m in (("tr", tr), ("va", va), ("te", te))}
    ins, branches = [], []

    si = keras.layers.Input(seq.shape[1:], name="history")
    ins.append(si)
    branches.append(keras.layers.Dropout(0.2)(keras.layers.LSTM(units)(si)))

    if seq2 is not None:
        wi = keras.layers.Input(seq2.shape[1:], name="weather")
        ins.append(wi)
        branches.append(keras.layers.Dropout(0.15)(keras.layers.LSTM(32)(wi)))

    ti = keras.layers.Input((X.shape[1],), name="static")
    ins.append(ti)
    branches.append(keras.layers.Dense(32, activation="relu")(ti))

    z = keras.layers.Concatenate()(branches) if len(branches) > 1 else branches[0]
    z = keras.layers.Dense(32, activation="relu")(z)
    model = keras.Model(ins, keras.layers.Dense(3, activation="softmax")(z))
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

    def pack(mask, key):
        a = [seq[mask]]
        if seq2 is not None:
            a.append(seq2[mask])
        a.append(S[key])
        return a

    cw = dict(enumerate(compute_class_weight("balanced", classes=np.arange(3), y=y[tr])))
    model.fit(pack(tr, "tr"), y[tr], validation_data=(pack(va, "va"), y[va]),
              epochs=40, batch_size=4096, verbose=0, class_weight=cw,
              callbacks=[BestMacroF1(pack(va, "va"), y[va], patience=8)])
    return model.predict(pack(te, "te"), verbose=0, batch_size=8192)


# ================================================================ the queue
ROUND_THREE = [
    ("r3-gb-ndvi", "Probe: satellite greenness on top of the round-two winner",
     ["base", "longmem", "elevation", "ndvi"], "histgb"),
    ("r3-gb-ndvi-growth", "Probe: greenness and wet-season shape together",
     ["base", "longmem", "elevation", "ndvi", "growth"], "histgb"),
    ("r3-lstm-ndvi", "The incumbent architecture on the winning features", ["winner"], "lstm"),
    ("r3-lstm-seq-ndvi", "Winning features plus the monthly weather branch", ["winner"], "lstm-seq"),
    ("r3-ensemble", "Average the best tree and the best network", ["winner"], "ensemble"),
]

ROUND_TWO = [
    ("r2-gb-longmem", "Probe: long memory and elevation", ["base", "longmem", "elevation"], "histgb"),
    ("r2-gb-spatial", "Probe: add district-scale neighbourhood", ["base", "longmem", "elevation", "spatial"], "histgb"),
    ("r2-gb-growth", "Probe: add wet-season shape", ["base", "longmem", "elevation", "growth"], "histgb"),
    ("r2-gb-soi", "Probe: add the Southern Oscillation Index", ["base", "longmem", "elevation", "growth", "soi"], "histgb"),
    ("r2-lstm", "The incumbent architecture on the winning features", ["winner"], "lstm"),
    ("r2-lstm-seq", "Winning features plus a monthly weather branch", ["winner"], "lstm-seq"),
    ("r2-ensemble", "Average the best tree and the best network", ["winner"], "ensemble"),
]

EXPERIMENTS = [
    ("gb-base", "Gradient boosting on the current features", ["base"], "histgb"),
    ("gb-longmem", "Add 15, 20 and 25 year fire frequency", ["base", "longmem", "elevation"], "histgb"),
    ("gb-spatial", "Add district-scale neighbourhood at two scales", ["base", "longmem", "elevation", "spatial"], "histgb"),
    ("gb-growth", "Add wet-season shape and growth index", ["base", "longmem", "elevation", "spatial", "growth"], "histgb"),
    ("gb-soi", "Add the Southern Oscillation Index", ["base", "longmem", "elevation", "spatial", "growth", "soi"], "histgb"),
    ("gb-monthly", "Give the trees the months as separate columns", ["base", "longmem", "elevation", "spatial", "growth", "monthly"], "histgb"),
    ("lstm-base", "The incumbent architecture, for a like-for-like line", ["base"], "lstm"),
    ("lstm-best-static", "Incumbent architecture on whatever features won above", ["winner"], "lstm"),
    ("lstm-weather-seq", "Second recurrent branch reading the wet season month by month", ["winner"], "lstm-seq"),
    ("ensemble", "Average the two strongest models that disagree most", ["winner"], "ensemble"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--round2", action="store_true")
    ap.add_argument("--round3", action="store_true")
    ap.add_argument("only", nargs="*")
    args = ap.parse_args()
    if args.list:
        for n, d, b, m in EXPERIMENTS:
            print(f"  {n:<20} {d}")
        return

    feat = pd.read_parquet(PROC / "cell_year_features_weather.parquet").reset_index(drop=True)
    labels = pd.read_parquet(INTERIM / "cell_year_labels.parquet")
    base_cols = FIRE_FEATURES + [c for c in feat.columns if c.startswith("wx_")]

    built, cols_of = {}, {"base": base_cols}
    for name, fn in BLOCKS.items():
        t0 = time.time()
        df = fn(feat, labels)
        built[name] = df
        cols_of[name] = list(df.columns)
        feat = pd.concat([feat, df], axis=1)
        log(f"built {name:<10} {len(df.columns):>3} columns in {time.time()-t0:>5.0f}s")

    y = feat["label"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    tr = feat["year"].isin(TRAIN_YEARS).to_numpy()
    va = feat["year"].isin(VAL_YEARS).to_numpy()
    te = feat["year"].isin(TEST_YEARS).to_numpy()
    seq = build_sequences(labels, feat)

    mcols = cols_of["monthly"]
    wseq = feat[mcols].to_numpy("float32").reshape(len(feat), MONTHS_IN_SEASON, len(MONTH_VARS))
    wm = np.nanmean(wseq.reshape(-1, len(MONTH_VARS)), axis=0)
    ws = np.nanstd(wseq.reshape(-1, len(MONTH_VARS)), axis=0) + 1e-6
    wseq = np.nan_to_num((wseq - wm) / ws, nan=0.0).astype("float32")

    ledger = json.loads(LEDGER.read_text()) if LEDGER.exists() else {"runs": []}
    best = {"name": "LSTM (balanced), previous best", "f1_macro": INCUMBENT, "blocks": ["base"]}
    # Feature selection is judged on its own terms, against the best feature set found so
    # far rather than against the incumbent model. Otherwise a block that clearly helps
    # the probe is discarded because a different architecture still scores higher, and
    # every later run quietly falls back to the base columns.
    probe_best, winner_blocks = 0.0, ["base"]
    probs = {}

    pool = ROUND_THREE if args.round3 else (ROUND_TWO if args.round2 else EXPERIMENTS)
    queue = [e for e in pool if not args.only or e[0] in args.only]
    for name, desc, blocks, kind in queue:
        blocks = winner_blocks if blocks == ["winner"] else blocks
        cols = [c for b in blocks for c in cols_of[b]]
        X = feat[cols].to_numpy("float32")
        log(f"\n{name}  ({len(cols)} columns)  {desc}")

        t0 = time.time()
        if kind == "histgb":
            p = fit_histgb(X, y, tr, va, te)
        elif kind == "lstm":
            p = fit_lstm(X, y, tr, va, te, seq)
        elif kind == "lstm-seq":
            p = fit_lstm(X, y, tr, va, te, seq, seq2=wseq)
        elif kind == "ensemble":
            # One entry per architecture, and the pair must actually disagree.
            # Averaging two fits of the same model is not an ensemble.
            kinds = {}
            for k, v in probs.items():
                fam = "lstm" if k.startswith("lstm") else "gb"
                f = f1_score(y[te], v.argmax(1), average="macro", zero_division=0)
                if fam not in kinds or f > kinds[fam][0]:
                    kinds[fam] = (f, k, v)
            picked = [(k, v) for _, k, v in kinds.values()]
            p = sum(v for _, v in picked) / len(picked)
            a, b = [v.argmax(1) for _, v in picked[:2]] if len(picked) > 1 else (None, None)
            if a is not None:
                log(f"  the two disagree on {(a != b).mean()*100:.1f}% of test rows")
            desc += " (" + " + ".join(k for k, _ in picked) + ")"
        secs = time.time() - t0
        probs[name] = p

        pred = p.argmax(1)
        f1 = float(f1_score(y[te], pred, average="macro", zero_division=0))
        rec = {
            "name": name, "description": desc, "blocks": blocks, "model": kind,
            "n_features": len(cols), "seconds": round(secs, 1),
            "accuracy": round(float(accuracy_score(y[te], pred)), 4),
            "f1_macro": round(f1, 4),
            "f1_per_class": {c: round(float(v), 4) for c, v in zip(
                CLASSES, f1_score(y[te], pred, average=None, labels=[0, 1, 2], zero_division=0))},
            "confusion": confusion_matrix(y[te], pred, labels=[0, 1, 2]).tolist(),
            "beat_incumbent": f1 > best["f1_macro"],
            "incumbent_at_run": round(best["f1_macro"], 4),
            "verdict": "KEEP" if f1 > best["f1_macro"] else "REVERT",
        }
        ledger["runs"].append(rec)
        LEDGER.parent.mkdir(exist_ok=True)
        LEDGER.write_text(json.dumps(ledger, indent=1))

        delta = f1 - best["f1_macro"]
        log(f"  macroF1 {f1:.4f}  ({delta:+.4f} vs {best['name']})  "
            f"acc {rec['accuracy']:.3f}  late {rec['f1_per_class']['late']:.3f}  "
            f"[{rec['verdict']}]  {secs:.0f}s")
        if kind == "histgb" and f1 > probe_best:
            probe_best, winner_blocks = f1, blocks
            log(f"  feature set adopted: {' + '.join(blocks)}")
        elif kind == "histgb":
            log(f"  feature set rejected, staying with {' + '.join(winner_blocks)}")
        if f1 > best["f1_macro"]:
            best = {"name": name, "f1_macro": f1, "blocks": blocks}

    log("\n" + "=" * 78)
    log(f"{'experiment':<22}{'features':>9}{'macroF1':>10}{'vs prev':>10}{'late F1':>9}  verdict")
    log("-" * 78)
    for r in ledger["runs"][-len(queue):]:
        log(f"{r['name']:<22}{r['n_features']:>9}{r['f1_macro']:>10.4f}"
            f"{r['f1_macro']-r['incumbent_at_run']:>+10.4f}{r['f1_per_class']['late']:>9.3f}"
            f"  {r['verdict']}")
    log("=" * 78)
    log(f"\nbest overall: {best['name']}  macroF1 {best['f1_macro']:.4f}  "
        f"(started from {INCUMBENT:.4f})")


if __name__ == "__main__":
    main()
