"""A convolutional model that sees the Territory as a picture, not as a list of squares.

Every model so far reads one cell at a time. It is handed numbers describing that cell's
past and its immediate neighbours, and it has no way to notice that the cell sits on the
edge of a burn scar, or inside a corridor that funnels fire, or on the boundary between
two fire regimes. Shape is invisible to it.

This gives the whole grid to a convolutional network as a stack of images: one channel per
piece of evidence, one image per year, and a three-class label for every pixel.

The hard constraint is that there are only fifteen training years. Fifteen images is
nothing for a network, so it trains on random crops instead of whole maps, which turns
fifteen pictures into tens of thousands of examples while keeping the spatial structure
that is the entire point. Inference then runs on the full map in one pass, since the
network is fully convolutional and does not care about input size.

Pixels outside the Northern Territory are masked out of the loss rather than fed to it as
a fourth class, so the network never spends capacity learning the coastline.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pyrantis.schema import CLASSES, INTERIM, PROC, ROOT as PROJ
from scripts.build_features import to_grid
from scripts.train_models import TRAIN_YEARS, VAL_YEARS, TEST_YEARS

SEED = 42
CROP = 64
STEPS_PER_EPOCH = 220
BATCH = 16
INCUMBENT = 0.6740

# Evidence given to the network, one channel each. Everything is from before the target
# year, or from weather known by 30 April, exactly as for every other model here.
LAGGED = ["frac_burnt", "frac_early", "frac_late"]
LAGS = (1, 2, 3)
STATIC_FEATS = ["freq_5", "freq_10", "yrs_since_burnt", "lat", "lon"]
WEATHER_FEATS = ["wx_wet_rain", "wx_wet_vpd", "wx_wet_tmax", "wx_days_since_rain",
                 "wx_wet_rain_anom", "wx_dry_rain_prev"]


def log(m=""):
    print(m, flush=True)


def build_stack():
    """(year, row, col, channel) evidence, plus labels and an in-Territory mask."""
    lab = pd.read_parquet(INTERIM / "cell_year_labels.parquet")
    feat = pd.read_parquet(PROC / "cell_year_features_weather.parquet")
    years = np.sort(lab["year"].unique())
    nrow, ncol = lab["row"].max() + 1, lab["col"].max() + 1

    lagged = {c: to_grid(lab, c, nrow, ncol, years) for c in LAGGED}
    inside = ~np.isnan(lagged["frac_burnt"][0])

    y_grid = np.full((len(years), nrow, ncol), -1, dtype="int8")
    yi = np.searchsorted(years, lab["year"].to_numpy())
    code = lab["label"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    y_grid[yi, lab["row"].to_numpy(), lab["col"].to_numpy()] = code

    fy = feat["year"].to_numpy()
    fr, fc = feat["row"].to_numpy(), feat["col"].to_numpy()
    yidx = {y: i for i, y in enumerate(years)}
    fyi = np.array([yidx[y] for y in fy])

    per_year = {}
    for name in STATIC_FEATS + WEATHER_FEATS:
        g = np.full((len(years), nrow, ncol), np.nan, "float32")
        g[fyi, fr, fc] = feat[name].to_numpy()
        per_year[name] = g

    chans = []
    names = []
    for c in LAGGED:
        for lg in LAGS:
            a = np.full_like(lagged[c], np.nan)
            a[lg:] = lagged[c][:-lg]
            chans.append(a)
            names.append(f"{c}_lag{lg}")
    for name in STATIC_FEATS + WEATHER_FEATS:
        chans.append(per_year[name])
        names.append(name)

    X = np.stack(chans, axis=-1).astype("float32")
    X = np.nan_to_num(X, nan=0.0)

    # Standardised on the training years only, so the test years contribute nothing to
    # the scaling any more than they do to the weights.
    tr = np.isin(years, list(TRAIN_YEARS))
    m = X[tr].reshape(-1, X.shape[-1]).mean(0)
    s = X[tr].reshape(-1, X.shape[-1]).std(0) + 1e-6
    X = (X - m) / s

    log(f"grid {nrow} x {ncol}, {len(years)} years, {X.shape[-1]} channels")
    log(f"channels: {', '.join(names)}")
    return years, X, y_grid, inside


def crop_stream(X, y, inside, idxs, cw, seed):
    """Endless random windows, cut on demand.

    Materialising the crops up front cost eleven gigabytes and put the machine into swap,
    where it trained at six per cent CPU. Cutting them as they are needed keeps only the
    year stack in memory, and makes the supply of training windows effectively unlimited
    rather than fixed at whatever fitted.
    """
    H, W = inside.shape
    rng = np.random.default_rng(seed)

    def gen():
        while True:
            t = int(rng.choice(idxs))
            r = int(rng.integers(0, H - CROP))
            c = int(rng.integers(0, W - CROP))
            m = inside[r:r + CROP, c:c + CROP]
            if m.mean() < 0.25:
                continue          # mostly sea or interstate, nothing to learn from
            yy = y[t, r:r + CROP, c:c + CROP]
            lab = np.where(yy < 0, 0, yy).astype("int32")
            # The mask and the class weights share one per-pixel weight: pixels outside
            # the Territory weigh nothing, the rest weigh by how rare their class is.
            w = (m & (yy >= 0)).astype("float32") * cw[lab]
            yield X[t, r:r + CROP, c:c + CROP], lab, w

    return gen


def main():
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    keras.utils.set_random_seed(SEED)
    rng = np.random.default_rng(SEED)

    years, X, Y, inside = build_stack()
    yidx = {y: i for i, y in enumerate(years)}
    tr_i = [yidx[y] for y in TRAIN_YEARS if y in yidx]
    va_i = [yidx[y] for y in VAL_YEARS if y in yidx]
    te_i = [yidx[y] for y in TEST_YEARS if y in yidx]
    log(f"train {len(tr_i)} years, validate {len(va_i)}, test {len(te_i)}")

    counts = np.bincount(Y[tr_i][(Y[tr_i] >= 0) & inside[None]].astype(int), minlength=3)
    cw = (counts.sum() / (3 * np.maximum(counts, 1))).astype("float32")
    log(f"training pixels {counts.tolist()}, class weights {np.round(cw, 2).tolist()}")

    sig = (tf.TensorSpec((CROP, CROP, X.shape[-1]), tf.float32),
           tf.TensorSpec((CROP, CROP), tf.int32),
           tf.TensorSpec((CROP, CROP), tf.float32))
    ds_tr = (tf.data.Dataset.from_generator(
        crop_stream(X, Y, inside, tr_i, cw, SEED), output_signature=sig)
        .batch(BATCH).prefetch(tf.data.AUTOTUNE))
    ds_va = (tf.data.Dataset.from_generator(
        crop_stream(X, Y, inside, va_i, cw, SEED + 1), output_signature=sig)
        .batch(BATCH).take(40).cache().prefetch(tf.data.AUTOTUNE))
    log(f"streaming {CROP}x{CROP} crops, {STEPS_PER_EPOCH} steps of {BATCH} per epoch")

    def block(x, f):
        x = layers.Conv2D(f, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(f, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        return layers.Activation("relu")(x)

    # A small U-Net. Two downsamples is enough: at 0.05 degrees a 96 pixel crop already
    # spans about 500 km, and the receptive field only has to cover how far fire and
    # burning programmes actually reach.
    inp = layers.Input((None, None, X.shape[-1]))
    c1 = block(inp, 32)
    c2 = block(layers.MaxPooling2D()(c1), 64)
    c3 = block(layers.MaxPooling2D()(c2), 128)
    u2 = layers.Concatenate()([layers.UpSampling2D()(c3), c2])
    d2 = block(u2, 64)
    u1 = layers.Concatenate()([layers.UpSampling2D()(d2), c1])
    d1 = block(u1, 32)
    out = layers.Conv2D(3, 1, activation="softmax")(d1)
    model = keras.Model(inp, out)
    # Keras 3 takes a per-pixel sample_weight array directly and dropped the old
    # sample_weight_mode argument, so the mask and the class weights ride in together
    # as one weight per pixel.
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy")
    log(f"U-Net with {model.count_params():,} parameters")

    def full_predict(idxs):
        # Pad to a multiple of four so the two downsamples line up, then trim back.
        H, W = inside.shape
        ph, pw = (-H) % 4, (-W) % 4
        a = np.pad(X[idxs], ((0, 0), (0, ph), (0, pw), (0, 0)))
        p = model.predict(a, verbose=0, batch_size=1)
        return p[:, :H, :W]

    def score_years(idxs):
        p = full_predict(idxs).argmax(-1)
        t = Y[idxs]
        m = (t >= 0) & inside[None]
        return t[m], p[m]

    best = {"f1": -1.0, "epoch": 0, "weights": None}

    class Watch(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            yt, yp = score_years(va_i)
            f = f1_score(yt, yp, average="macro", zero_division=0)
            if f > best["f1"]:
                best.update(f1=f, epoch=epoch, weights=self.model.get_weights())
            log(f"  epoch {epoch + 1:>2}  val macroF1 {f:.4f}"
                f"{'   <- best' if f == best['f1'] else ''}")

    t0 = time.time()
    model.fit(ds_tr, steps_per_epoch=STEPS_PER_EPOCH, validation_data=ds_va,
              epochs=25, verbose=0, callbacks=[Watch()])
    model.set_weights(best["weights"])
    secs = time.time() - t0
    log(f"\ntrained in {secs/60:.1f} min, best epoch {best['epoch'] + 1} "
        f"at val macroF1 {best['f1']:.4f}")

    yt, yp = score_years(te_i)
    f1 = float(f1_score(yt, yp, average="macro", zero_division=0))
    per = f1_score(yt, yp, average=None, labels=[0, 1, 2], zero_division=0)
    rec = {
        "name": "cnn-unet", "model": "cnn",
        "description": "U-Net over the whole grid, trained on random crops",
        "n_features": int(X.shape[-1]), "seconds": round(secs, 1),
        "accuracy": round(float(accuracy_score(yt, yp)), 4),
        "f1_macro": round(f1, 4),
        "f1_per_class": {c: round(float(v), 4) for c, v in zip(CLASSES, per)},
        "confusion": confusion_matrix(yt, yp, labels=[0, 1, 2]).tolist(),
        "incumbent_at_run": INCUMBENT,
        "verdict": "KEEP" if f1 > INCUMBENT else "REVERT",
    }
    log(f"\ntest macroF1 {f1:.4f} ({f1 - INCUMBENT:+.4f} vs incumbent {INCUMBENT})")
    log(f"accuracy {rec['accuracy']:.3f}   late F1 {rec['f1_per_class']['late']:.3f}"
        f"   [{rec['verdict']}]")

    led = PROJ / "reports" / "experiments.json"
    d = json.loads(led.read_text()) if led.exists() else {"runs": []}
    d["runs"].append(rec)
    led.write_text(json.dumps(d, indent=1))
    if f1 > INCUMBENT:
        model.save(PROJ / "reports" / "cnn_unet.keras")
        log("saved the model, it beat the incumbent")


if __name__ == "__main__":
    main()
