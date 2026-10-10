"""Figures and numbers for the Assessment 4 report.

The Assessment 3 figures argued for the model. These ones are for reading the data
first: when the Territory burns, where, under what wet season, and how the inputs
relate to each other, followed by the results charts the report discusses. Every number
quoted in the report text is written to reports/a4/analysis.json from here, so the text
can be checked against the data rather than trusted.

The k-means section follows the Week 11 practical (scale, elbow, silhouette and
Davies-Bouldin). It describes the Territory; nothing from it is fed to the models, since
it is fitted on every year including the test years.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pyrantis.schema import CLASSES, INTERIM, PROC, RAW, ROOT as PROJ

OUT = PROJ / "reports" / "a4"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

COLOUR = {"unburnt": "#b9c2c7", "early": "#2e8b74", "late": "#c1440e"}
INK, MUTED, RULE = "#22282b", "#5f686d", "#d9dee0"
RAIN = "#2f5d8a"
NICE = {"unburnt": "No fire", "early": "Fire before August", "late": "Fire after July"}

plt.rcParams.update({
    "font.family": "Times New Roman", "font.size": 10, "figure.dpi": 150,
    "savefig.dpi": 220, "axes.edgecolor": INK, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "figure.facecolor": "white", "legend.frameon": False,
})

stats: dict = {}


def save(fig, name: str) -> None:
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", name)


def load():
    feat = pd.read_parquet(PROC / "cell_year_features_full.parquet")
    feat["label"] = feat["label"].astype(str)
    lab = pd.read_parquet(INTERIM / "cell_year_labels.parquet")
    lab["label"] = lab["label"].astype(str)
    return feat, lab


def outline(ax) -> None:
    import geopandas as gpd
    nt = gpd.read_file(RAW / "boundary" / "au_states.geojson")
    name_col = next(c for c in nt.columns if nt[c].astype(str).str.contains("Northern").any())
    nt[nt[name_col].astype(str).str.contains("Northern")].boundary.plot(
        ax=ax, color=INK, linewidth=0.6)


def raster(df: pd.DataFrame, col: str, shape) -> np.ndarray:
    a = np.full(shape, np.nan)
    a[df["row"].to_numpy(), df["col"].to_numpy()] = df[col].to_numpy()
    return a


def map_axes(ax, title: str) -> None:
    ax.set_title(title, fontsize=10.5)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_aspect("equal")


# ------------------------------------------------------------------- section 2 figures

def fig_months(lab: pd.DataFrame) -> None:
    """Which month each burnt square burnt in, the basis of the 31 July rule."""
    burnt = lab[lab["peak_month"] > 0]
    counts = burnt["peak_month"].value_counts().sort_index()
    share = counts / counts.sum() * 100
    stats["month_share"] = {int(k): round(float(v), 1) for k, v in share.items()}
    stats["share_aug_nov"] = round(float(share.loc[8:11].sum()), 1)
    stats["share_may_jul"] = round(float(share.loc[5:7].sum()), 1)

    fig, ax = plt.subplots(figsize=(7.2, 2.7))
    cols = [COLOUR["early"] if m <= 7 else COLOUR["late"] for m in share.index]
    ax.bar(share.index, share.values, color=cols, width=0.72)
    ax.axvline(7.5, color=INK, lw=0.8, ls="--")
    ax.text(7.62, share.max() * 0.96, "31 July", fontsize=9, va="top")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep",
                        "Oct", "Nov", "Dec"])
    ax.set_ylabel("share of burnt squares (%)")
    ax.set_title("Month in which each burnt square burnt, 2000–2025")
    save(fig, "f01_months")


def fig_years(feat: pd.DataFrame, lab: pd.DataFrame) -> None:
    """Fire regime by year against the Territory's wet-season rainfall."""
    share = pd.crosstab(lab["year"], lab["label"], normalize="index")[list(CLASSES)] * 100
    rain = feat.groupby("year")["wx_wet_rain"].mean()
    both = share.join(rain, how="inner")
    r_late = float(both["late"].corr(both["wx_wet_rain"]))
    r_early = float(both["early"].corr(both["wx_wet_rain"]))
    stats["year_corr_rain_late"] = round(r_late, 2)
    stats["year_corr_rain_early"] = round(r_early, 2)
    stats["year_share"] = {int(y): {c: round(float(share.loc[y, c]), 1) for c in CLASSES}
                           for y in share.index}
    stats["year_rain"] = {int(y): round(float(v), 0) for y, v in rain.items()}

    fig, ax = plt.subplots(figsize=(7.4, 3.2))
    bottom = np.zeros(len(share))
    for c in ("late", "early"):
        ax.bar(share.index, share[c], bottom=bottom, color=COLOUR[c], width=0.75,
               label=NICE[c])
        bottom += share[c].to_numpy()
    ax.set_ylabel("share of the Territory burnt (%)")
    ax.set_ylim(0, 75)
    ax2 = ax.twinx()
    ax2.plot(rain.index, rain.values, color=RAIN, lw=1.6, marker="o", ms=3,
             label="Wet-season rainfall")
    ax2.set_ylabel("mean wet-season rainfall (mm)", color=RAIN)
    ax2.tick_params(axis="y", colors=RAIN)
    ax2.spines["right"].set_visible(True); ax2.spines["right"].set_color(RAIN)
    ax2.set_ylim(0, 1100)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", ncol=3, fontsize=9)
    ax.set_title("Burnt share by season and the wet season before it")
    ax.set_xlim(1999.4, 2025.6)
    save(fig, "f02_years_rain")


def fig_maps(feat: pd.DataFrame, lab: pd.DataFrame) -> None:
    """Where fire happens, where it is late, and where the rain falls."""
    shape = (lab["row"].max() + 1, lab["col"].max() + 1)
    g = lab.groupby(["row", "col"])
    per = pd.DataFrame({
        "lon": g["lon"].first(), "lat": g["lat"].first(),
        "freq": g["label"].apply(lambda s: (s != "unburnt").mean() * 100),
        "late_share": g["label"].apply(
            lambda s: (s == "late").sum() / max((s != "unburnt").sum(), 1) * 100),
        "nburnt": g["label"].apply(lambda s: (s != "unburnt").sum()),
        "early_freq": g["label"].apply(lambda s: (s == "early").mean() * 100),
        "late_freq": g["label"].apply(lambda s: (s == "late").mean() * 100),
    }).reset_index()
    per.loc[per["nburnt"] == 0, "late_share"] = np.nan
    rain = feat.groupby(["row", "col"])["wx_wet_rain"].mean().rename("rain").reset_index()
    per = per.merge(rain, on=["row", "col"], how="left")
    per.to_parquet(OUT / "cell_summary.parquet")

    ext = [lab["lon"].min() - 0.025, lab["lon"].max() + 0.025,
           lab["lat"].min() - 0.025, lab["lat"].max() + 0.025]
    fire = LinearSegmentedColormap.from_list("fire", ["#f4efe6", "#e8b04a", "#c1440e", "#5e1c06"])
    late = LinearSegmentedColormap.from_list("late", [COLOUR["early"], "#efe7d4", COLOUR["late"]])
    wet = LinearSegmentedColormap.from_list("wet", ["#f2ede2", "#8fb4cf", RAIN, "#16304a"])

    fig, axes = plt.subplots(1, 3, figsize=(7.6, 3.6))
    specs = [("freq", fire, "Years with fire (%)", (0, 100)),
             ("late_share", late, "Fires after July (%)", (0, 100)),
             ("rain", wet, "Mean wet-season rainfall (mm)", (0, 1600))]
    for ax, (col, cmap, title, (lo, hi)) in zip(axes, specs):
        a = raster(per, col, shape)
        im = ax.imshow(a, extent=ext, origin="upper", cmap=cmap, vmin=lo, vmax=hi,
                       interpolation="nearest")
        outline(ax)
        ax.set_xlim(ext[0] - 0.2, ext[1] + 0.2); ax.set_ylim(ext[2] - 0.2, ext[3] + 0.2)
        map_axes(ax, title)
        cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.05, pad=0.03,
                          shrink=0.82)
        cb.outline.set_visible(False); cb.ax.tick_params(labelsize=8)
        cb.set_ticks(np.linspace(lo, hi, 5 if hi == 100 else 5))
    fig.subplots_adjust(wspace=0.05)
    save(fig, "f03_maps")

    top = per[per["lat"] > -14]
    south = per[per["lat"] < -20]
    stats["top_end_mean_freq"] = round(float(top["freq"].mean()), 1)
    stats["south_mean_freq"] = round(float(south["freq"].mean()), 1)
    stats["south_late_share"] = round(float(
        (lab[(lab["lat"] < -20)]["label"] == "late").sum()
        / max((lab[(lab["lat"] < -20)]["label"] != "unburnt").sum(), 1) * 100), 1)
    stats["top_late_share"] = round(float(
        (lab[(lab["lat"] > -14)]["label"] == "late").sum()
        / max((lab[(lab["lat"] > -14)]["label"] != "unburnt").sum(), 1) * 100), 1)
    stats["cells_never_burnt_pct"] = round(float((per["nburnt"] == 0).mean() * 100), 1)


def fig_bands(feat: pd.DataFrame) -> None:
    """Class mix down the Territory, beside the rain and greenness behind it."""
    bands = [-26.1, -23, -20, -17, -14, -10.9]
    names = ["23–26° S", "20–23° S", "17–20° S", "14–17° S", "11–14° S"]
    b = pd.cut(feat["lat"], bands, labels=names)
    mix = pd.crosstab(b, feat["label"], normalize="index")[list(CLASSES)] * 100
    env = feat.groupby(b, observed=True)[["wx_wet_rain", "ndvi_wet_mean"]].mean()
    stats["bands"] = {n: {**{c: round(float(mix.loc[n, c]), 1) for c in CLASSES},
                          "rain": round(float(env.loc[n, "wx_wet_rain"]), 0),
                          "ndvi": round(float(env.loc[n, "ndvi_wet_mean"]), 2)}
                      for n in names}

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.4, 2.8), width_ratios=[2.2, 1],
                                  sharey=True)
    left = np.zeros(len(mix))
    for c in CLASSES:
        ax.barh(mix.index.astype(str), mix[c], left=left, color=COLOUR[c], label=NICE[c],
                height=0.7)
        for i, v in enumerate(mix[c]):
            if v > 8:
                ax.text(left[i] + v / 2, i, f"{v:.0f}%", ha="center", va="center",
                        fontsize=8.5, color="white" if c != "unburnt" else INK)
        left += mix[c].to_numpy()
    ax.set_xlim(0, 100); ax.set_xlabel("share of cell-years (%)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.62, -0.2), ncol=3, fontsize=8.5)
    ax.set_title("Class mix by latitude band")
    ax2.barh(env.index.astype(str), env["wx_wet_rain"], color=RAIN, height=0.7)
    for i, v in enumerate(env["wx_wet_rain"]):
        ax2.text(v + 20, i, f"{v:.0f}", va="center", fontsize=8.5)
    ax2.set_xlabel("wet-season rainfall (mm)")
    ax2.set_xlim(0, 1650)
    ax2.set_title("Rain")
    save(fig, "f04_bands")


def fig_interaction(feat: pd.DataFrame) -> None:
    """Late-fire rate by wet-season anomaly and years since the square last burnt."""
    q = pd.qcut(feat["wx_wet_rain_anom"], 5, labels=False)
    ysb = feat["yrs_since_burnt"].clip(upper=8)
    t = (feat.assign(late=(feat["label"] == "late") * 100.0, q=q, ysb=ysb)
         .pivot_table(index="q", columns="ysb", values="late", aggfunc="mean"))
    n = (feat.assign(q=q, ysb=ysb).pivot_table(index="q", columns="ysb", values="year",
                                                aggfunc="size"))
    edges = feat["wx_wet_rain_anom"].quantile([0, .2, .4, .6, .8, 1]).to_numpy()
    stats["interaction"] = {
        "driest_q": {int(k): round(float(v), 1) for k, v in t.loc[0].items()},
        "wettest_q": {int(k): round(float(v), 1) for k, v in t.loc[4].items()},
        "anom_edges_mm": [round(float(e), 0) for e in edges],
    }
    ysb_mix = pd.crosstab(feat["yrs_since_burnt"].clip(upper=10), feat["label"],
                          normalize="index")[list(CLASSES)] * 100
    stats["ysb_mix"] = {int(k): {c: round(float(ysb_mix.loc[k, c]), 1) for c in CLASSES}
                        for k in ysb_mix.index}

    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    heat = LinearSegmentedColormap.from_list("h", ["#fbf7ef", "#eeb26a", "#c1440e", "#5e1c06"])
    im = ax.imshow(t.to_numpy(), cmap=heat, vmin=0, vmax=50, aspect="auto", origin="lower")
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            v = t.iat[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8.5,
                    color="white" if v > 30 else INK)
    ax.set_xticks(range(t.shape[1]))
    ax.set_xticklabels([str(int(c)) if c < 8 else "8+" for c in t.columns])
    ax.set_yticks(range(5))
    ax.set_yticklabels(["driest fifth", "2nd", "3rd", "4th", "wettest fifth"])
    ax.set_xlabel("years since the square last burnt")
    ax.set_ylabel("wet season vs the cell's normal")
    ax.set_title("Late-fire rate (%) by wet-season anomaly and time since fire")
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.outline.set_visible(False)
    save(fig, "f05_interaction")


def fig_skew(feat: pd.DataFrame) -> None:
    """Shape of four inputs, following the Week 6 skewness check."""
    cols = [("wx_wet_rain", "Wet-season rainfall (mm)"),
            ("wx_apr_rain", "April rainfall (mm)"),
            ("yrs_since_burnt", "Years since last fire"),
            ("wx_wet_vpd", "Wet-season VPD (kPa)")]
    stats["skew"] = {c: round(float(feat[c].skew()), 2) for c, _ in cols}
    stats["skew_log_wet_rain"] = round(float(np.log1p(feat["wx_wet_rain"]).skew()), 2)
    stats["skew_log_apr_rain"] = round(float(np.log1p(feat["wx_apr_rain"]).skew()), 2)
    fig, axes = plt.subplots(1, 4, figsize=(7.6, 2.2))
    for ax, (c, name) in zip(axes, cols):
        v = feat[c].to_numpy()
        hi = np.quantile(v, 0.995)
        ax.hist(np.clip(v, None, hi), bins=40, color="#7d8a90", edgecolor="white", lw=0.3)
        ax.set_title(name, fontsize=9.5)
        ax.text(0.97, 0.92, f"skew {stats['skew'][c]:+.2f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=8.5,
                bbox=dict(facecolor="white", edgecolor=RULE, boxstyle="round,pad=0.25"))
        ax.set_yticks([]); ax.spines["left"].set_visible(False)
        ax.tick_params(labelsize=8)
    save(fig, "f06_skew")


def fig_corr(feat: pd.DataFrame) -> None:
    """Correlation between a readable subset of the 50 inputs."""
    cols = {"freq_3": "Fire freq. 3 yr", "freq_10": "Fire freq. 10 yr",
            "freq_25": "Fire freq. 25 yr", "late_freq_10": "Late freq. 10 yr",
            "yrs_since_burnt": "Years since fire", "prev_burnt": "Burnt last year",
            "nbr_frac_burnt_prev": "Neighbours burnt",
            "wx_wet_rain": "Wet rain", "wx_wet_rain_anom": "Wet rain anomaly",
            "wx_wet_vpd": "Wet VPD", "wx_wet_tmax": "Wet max temp",
            "ndvi_wet_mean": "Wet greenness", "ndvi_apr": "April greenness",
            "ndvi_apr_anom": "April green anomaly", "lat": "Latitude",
            "elevation": "Elevation"}
    c = feat[list(cols)].corr()
    c.index = c.columns = list(cols.values())
    pairs = []
    keys = list(cols.values())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            pairs.append((keys[i], keys[j], float(c.iat[i, j])))
    pairs.sort(key=lambda t: -abs(t[2]))
    stats["corr_top"] = [(a, b, round(v, 2)) for a, b, v in pairs[:10]]
    stats["corr_wet_rain_vpd"] = round(float(c.loc["Wet rain", "Wet VPD"]), 2)
    stats["corr_lat_greenness"] = round(float(c.loc["Latitude", "Wet greenness"]), 2)
    stats["corr_anom_rain"] = round(float(c.loc["Wet rain anomaly", "Wet rain"]), 2)

    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    div = LinearSegmentedColormap.from_list("d", [RAIN, "#f7f4ee", COLOUR["late"]])
    im = ax.imshow(c.to_numpy(), cmap=div, vmin=-1, vmax=1)
    for i in range(len(c)):
        for j in range(len(c)):
            v = c.iat[i, j]
            ax.text(j, i, f"{v:.1f}".replace("-0.0", "0.0"), ha="center", va="center",
                    fontsize=6.8, color="white" if abs(v) > 0.65 else INK)
    ax.set_xticks(range(len(c))); ax.set_yticks(range(len(c)))
    ax.set_xticklabels(c.columns, rotation=55, ha="right", fontsize=8)
    ax.set_yticklabels(c.index, fontsize=8)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Pearson correlation between sixteen of the inputs")
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.outline.set_visible(False)
    save(fig, "f07_corr")


def fig_kmeans() -> None:
    """Week 11 practical applied to the Territory: cluster squares by fire regime."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import davies_bouldin_score, silhouette_score
    from sklearn.preprocessing import StandardScaler

    per = pd.read_parquet(OUT / "cell_summary.parquet").dropna(subset=["rain"])
    # Early and late frequency rather than "share of fires that were late": a square
    # that never burnt has no late share, and filling it with zero invents a regime.
    X = StandardScaler().fit_transform(per[["early_freq", "late_freq", "rain"]])
    rng = np.random.default_rng(42)
    sample = rng.choice(len(X), size=min(8000, len(X)), replace=False)

    ks = range(2, 10)
    inertia, sil, dbi = [], [], []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=1).fit(X)
        inertia.append(float(km.inertia_))
        sil.append(float(silhouette_score(X[sample], km.labels_[sample])))
        dbi.append(float(davies_bouldin_score(X, km.labels_)))
    k_best = 3
    km = KMeans(n_clusters=k_best, n_init=10, random_state=1).fit(X)
    per["cluster"] = km.labels_

    prof = per.groupby("cluster")[["early_freq", "late_freq", "freq", "rain", "lat"]].mean()
    order = prof.sort_values("rain").index.tolist()          # dry south first
    labels_by_rank = ["Arid south: rarely burns", "Interior savanna: burns late",
                      "Top End: burns most years, mostly early"]
    names = {cl: labels_by_rank[i] for i, cl in enumerate(order)}
    palette = ["#cfc6b4", COLOUR["late"], COLOUR["early"]]
    cmap_idx = {cl: i for i, cl in enumerate(order)}

    stats["kmeans"] = {
        "k_range": list(ks), "inertia": [round(v, 0) for v in inertia],
        "silhouette": [round(v, 3) for v in sil], "davies_bouldin": [round(v, 3) for v in dbi],
        "k": k_best,
        "silhouette_k": round(sil[list(ks).index(k_best)], 3),
        "db_k": round(dbi[list(ks).index(k_best)], 3),
        "profiles": {names[cl]: {"cells": int((per["cluster"] == cl).sum()),
                                 "freq": round(float(prof.loc[cl, "freq"]), 1),
                                 "early_freq": round(float(prof.loc[cl, "early_freq"]), 1),
                                 "late_freq": round(float(prof.loc[cl, "late_freq"]), 1),
                                 "rain": round(float(prof.loc[cl, "rain"]), 0),
                                 "lat": round(float(prof.loc[cl, "lat"]), 1)}
                     for cl in order},
    }

    fig = plt.figure(figsize=(7.6, 3.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.35], hspace=0.6, wspace=0.15)
    a1 = fig.add_subplot(gs[0, 0]); a2 = fig.add_subplot(gs[1, 0])
    a3 = fig.add_subplot(gs[:, 1])
    a1.plot(list(ks), inertia, "o-", color=INK, ms=3.5, lw=1.2)
    a1.axvline(k_best, color=COLOUR["late"], lw=0.8, ls="--")
    a1.set_title("Elbow (inertia)", fontsize=9.5); a1.tick_params(labelsize=8)
    a1.set_yticks([])
    a2.plot(list(ks), sil, "o-", color=COLOUR["early"], ms=3.5, lw=1.2, label="silhouette ↑")
    a2.plot(list(ks), dbi, "s-", color=RAIN, ms=3.5, lw=1.2, label="Davies-Bouldin ↓")
    a2.axvline(k_best, color=COLOUR["late"], lw=0.8, ls="--")
    a2.set_xlabel("number of clusters k", fontsize=9); a2.tick_params(labelsize=8)
    a2.legend(fontsize=7.5, loc="center right", bbox_to_anchor=(1.0, 0.62))
    a2.set_title("Cluster quality", fontsize=9.5)

    lab = pd.read_parquet(INTERIM / "cell_year_labels.parquet", columns=["row", "col", "lon", "lat"])
    shape = (lab["row"].max() + 1, lab["col"].max() + 1)
    ext = [lab["lon"].min() - 0.025, lab["lon"].max() + 0.025,
           lab["lat"].min() - 0.025, lab["lat"].max() + 0.025]
    per["ci"] = per["cluster"].map(cmap_idx)
    a = raster(per, "ci", shape)
    a3.imshow(a, extent=ext, origin="upper", cmap=ListedColormap(palette[:k_best]),
              vmin=-0.5, vmax=k_best - 0.5, interpolation="nearest")
    outline(a3)
    a3.set_xlim(ext[0] - 0.2, ext[1] + 0.2); a3.set_ylim(ext[2] - 0.2, ext[3] + 0.2)
    map_axes(a3, f"k-means fire regions (k = {k_best})")
    from matplotlib.patches import Patch
    a3.legend(handles=[Patch(color=palette[cmap_idx[cl]], label=names[cl]) for cl in order[::-1]],
              loc="lower left", bbox_to_anchor=(1.0, 0.0), fontsize=8)
    save(fig, "f08_kmeans")


# ------------------------------------------------------------------- results figures

def results():
    r = json.load(open(PROJ / "reports" / "results_full.json"))
    return {m["model"]: m for m in r["results"]}


SHOW = [("gradient boosting (balanced)", "Gradient boosting"), ("LSTM (balanced)", "LSTM"),
        ("multilayer ANN (balanced)", "Multilayer ANN"),
        ("random forest (balanced)", "Random forest"),
        ("decision tree (balanced)", "Decision tree"), ("naive bayes (balanced)", "Naive Bayes"),
        ("same as last year", "Same as last year"), ("majority class", "Always unburnt")]


def per_class(cm) -> dict:
    cm = np.asarray(cm, float)
    tp = np.diag(cm)
    prec = np.divide(tp, cm.sum(0), out=np.zeros_like(tp), where=cm.sum(0) > 0)
    rec = tp / cm.sum(1)
    return {"precision": prec.tolist(), "recall": rec.tolist()}


def fig_metrics() -> None:
    res = results()
    stats["metrics"] = {}
    for key, name in SHOW:
        m = res[key]
        pc = per_class(m["confusion"])
        stats["metrics"][name] = {
            "accuracy": round(m["accuracy"], 3), "precision": round(m["precision_macro"], 3),
            "recall": round(m["recall_macro"], 3), "f1": round(m["f1_macro"], 3),
            "f1_class": {c: round(v, 3) for c, v in m["f1_per_class"].items()},
            "precision_class": dict(zip(CLASSES, [round(v, 3) for v in pc["precision"]])),
            "recall_class": dict(zip(CLASSES, [round(v, 3) for v in pc["recall"]])),
            "confusion": m["confusion"],
        }
    unw = {"gradient boosting": "gradient boosting", "random forest": "random forest",
           "multilayer ANN": "multilayer ANN", "LSTM": "LSTM", "decision tree": "decision tree",
           "naive bayes": "naive bayes"}
    stats["unweighted"] = {k: {"f1": round(res[k]["f1_macro"], 3),
                               "accuracy": round(res[k]["accuracy"], 3),
                               "f1_late": round(res[k]["f1_per_class"]["late"], 3),
                               "recall_late": round(per_class(res[k]["confusion"])["recall"][2], 3)}
                           for k in unw}

    names = [n for _, n in SHOW]
    measures = [("accuracy", "Accuracy", "#3d4a50"), ("precision", "Precision", RAIN),
                ("recall", "Recall", COLOUR["early"]), ("f1", "F1", COLOUR["late"])]
    fig, ax = plt.subplots(figsize=(7.6, 3.3))
    x = np.arange(len(names)); w = 0.2
    for i, (k, lab, col) in enumerate(measures):
        vals = [stats["metrics"][n][k] for n in names]
        ax.bar(x + (i - 1.5) * w, vals, w, color=col, label=lab)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    ax.set_ylim(0, 0.85); ax.set_ylabel("score on 2023–2025 (macro)")
    ax.axvspan(5.5, 7.5, color="#f1f3f4", zorder=-1)
    ax.text(6.5, 0.8, "baselines", ha="center", fontsize=9, color=MUTED)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.45, 1.13), fontsize=9)
    ax.set_title("Accuracy, precision, recall and F1 on the held-out years", pad=24)
    save(fig, "f10_metrics")

    # per-class precision and recall for every model
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0), sharey=True)
    six = names[:6]
    for ax, k, title in ((axes[0], "precision_class", "Precision by class"),
                         (axes[1], "recall_class", "Recall by class")):
        for i, c in enumerate(CLASSES):
            vals = [stats["metrics"][n][k][c] for n in six]
            ax.barh(np.arange(len(six)) + (i - 1) * 0.26, vals, 0.26, color=COLOUR[c],
                    label=NICE[c])
        ax.set_yticks(range(len(six))); ax.set_yticklabels(six, fontsize=9)
        ax.set_xlim(0, 1); ax.set_title(title, fontsize=10)
    axes[0].invert_yaxis()                   # shared axis: inverting both would undo it
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.05, 1.22), ncol=3, fontsize=8.5)
    save(fig, "f11_class_pr")


def fig_confusion() -> None:
    m = stats["metrics"]["Gradient boosting"]
    cm = np.asarray(m["confusion"], float)
    rown = cm / cm.sum(1, keepdims=True) * 100
    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    cmap = LinearSegmentedColormap.from_list("c", ["#fbf8f2", "#e3a35d", "#9b3509"])
    ax.imshow(rown, cmap=cmap, vmin=0, vmax=100)
    lbl = ["No fire", "Before Aug", "After July"]
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{int(cm[i, j]):,}\n{rown[i, j]:.0f}%", ha="center", va="center",
                    fontsize=9, color="white" if rown[i, j] > 55 else INK)
    ax.set_xticks(range(3)); ax.set_xticklabels(lbl); ax.set_yticks(range(3))
    ax.set_yticklabels(lbl)
    ax.set_xlabel("predicted"); ax.set_ylabel("what happened")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Gradient boosting, 2023–2025 (row %)", fontsize=10.5)
    save(fig, "f12_confusion")


def fig_fit() -> None:
    rob = json.load(open(PROJ / "reports" / "robustness.json"))["models"]
    order = ["gradient boosting", "LSTM", "multilayer ANN", "random forest", "decision tree",
             "naive bayes"]
    nice = ["Gradient boosting", "LSTM", "Multilayer ANN", "Random forest", "Decision tree",
            "Naive Bayes"]
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    x = np.arange(len(order)); w = 0.26
    for i, (k, lab, col) in enumerate([("f1_train", "Training years", "#9aa5aa"),
                                        ("f1_val", "Validation years", RAIN),
                                        ("f1_test", "Test years", COLOUR["late"])]):
        ax.bar(x + (i - 1) * w, [rob[o][k] for o in order], w, color=col, label=lab)
    for i, o in enumerate(order):
        gap = rob[o]["f1_train"] - rob[o]["f1_test"]
        top = max(rob[o]["f1_train"], rob[o]["f1_val"], rob[o]["f1_test"])
        ax.text(i, top + 0.015, f"train − test {gap:+.2f}".replace("+-", "−"),
                ha="center", fontsize=8, color=MUTED)
    ax.set_xticks(x); ax.set_xticklabels(nice, fontsize=9)
    ax.set_ylim(0.4, 1.0); ax.set_ylabel("macro F1")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=8.5)
    ax.set_title("Training, validation and test F1: where each model overfits")
    save(fig, "f13_fit")


def fig_walk() -> None:
    """Walk-forward folds. Gradient boosting was not part of this run (it was added after),
    so each of the five learners that were is reported on its own, against the baseline."""
    w = json.load(open(PROJ / "reports" / "walk_forward.json"))
    d = pd.DataFrame([{"horizon": f["horizon"], "model": m, "f1": v["f1_macro"],
                       "cutoff": f["cutoff"], "target": f["target"]}
                      for f in w["folds"] for m, v in f["models"].items()])
    p = d.pivot_table(index=["cutoff", "target"], columns="model", values="f1")
    learned = ["LSTM", "multilayer ANN", "random forest", "naive bayes", "decision tree"]
    base = "same as last year"
    stats["walk"] = {"folds": int(len(p)), "features": w["features"],
                     "baseline": {"mean": round(float(p[base].mean()), 3),
                                  "sd": round(float(p[base].std()), 3)},
                     "models": {m: {"mean": round(float(p[m].mean()), 3),
                                    "sd": round(float(p[m].std()), 3),
                                    "min": round(float(p[m].min()), 3),
                                    "max": round(float(p[m].max()), 3),
                                    "wins": int((p[m] > p[base]).sum())} for m in learned}}
    g = d.groupby(["model", "horizon"])["f1"].mean().unstack(0)
    stats["walk"]["by_horizon"] = {m: {int(h): round(float(v), 3) for h, v in g[m].items()}
                                   for m in learned + [base]}

    fig, ax = plt.subplots(figsize=(6.8, 2.9))
    cols = {"LSTM": COLOUR["late"], "multilayer ANN": "#d9963b", "random forest": COLOUR["early"],
            "naive bayes": RAIN, "decision tree": "#8a7f6b"}
    nice = {"LSTM": "LSTM", "multilayer ANN": "Multilayer ANN", "random forest": "Random forest",
            "naive bayes": "Naive Bayes", "decision tree": "Decision tree"}
    for m in learned:
        ax.plot(g.index, g[m], "o-", color=cols[m], lw=1.4, ms=3, label=nice[m])
    ax.plot(g.index, g[base], "s--", color="#8c979c", lw=1.2, ms=3, label="Same as last year")
    ax.set_xlabel("years between the last training year and the year predicted")
    ax.set_ylabel("macro F1"); ax.set_ylim(0.45, 0.7); ax.set_xticks(g.index)
    ax.legend(loc="lower left", fontsize=8, ncol=3)
    ax.set_title(f"Walk-forward tests ({len(p)} folds per model): skill against horizon")
    save(fig, "f14_walk")


def fig_importance() -> None:
    rob = json.load(open(PROJ / "reports" / "robustness.json"))
    gb = rob["permutation"]["gradient boosting"][:12]
    nice = {"ndvi_wet_mean": "Wet-season greenness", "ndvi_apr": "April greenness",
            "lat": "Latitude", "wx_wet_rain": "Wet-season rainfall",
            "wx_wet_vpd": "Wet-season VPD", "nbr_frac_late_prev": "Neighbours late fire last yr",
            "nbr_frac_burnt_prev": "Neighbours burnt last yr", "lon": "Longitude",
            "nbr_frac_early_prev": "Neighbours early fire last yr",
            "yrs_since_burnt": "Years since fire", "prev_frac_burnt": "Burnt fraction last yr",
            "wx_wet_rain_anom": "Wet rain anomaly", "ndvi_wet_min": "Wet greenness, lowest",
            "elevation": "Elevation", "wx_wet_tmax": "Wet-season max temp",
            "ndvi_greenup": "Green-up speed", "wx_wet_radn": "Wet-season radiation",
            "freq_25": "Fire freq. 25 yr", "late_freq_25": "Late freq. 25 yr",
            "wx_dry_vpd_prev": "Last dry season VPD", "ndvi_wet_max": "Wet greenness, peak",
            "wx_wet_rain_ratio": "Wet rain ratio", "prev_burnt": "Burnt last yr",
            "freq_10": "Fire freq. 10 yr", "ndvi_peak_anom": "Peak greenness anomaly",
            "ndvi_apr_anom": "April greenness anomaly", "wx_apr_vpd": "April VPD",
            "wx_dry_tmax_prev": "Last dry season max temp", "late_freq_10": "Late freq. 10 yr",
            "wx_wet_rain_prev": "Previous wet rain", "wx_dry_rain_prev": "Last dry rain",
            "freq_20": "Fire freq. 20 yr", "late_freq_20": "Late freq. 20 yr",
            "prev_frac_late": "Late fraction last yr", "prev_frac_early": "Early fraction last yr",
            "freq_15": "Fire freq. 15 yr", "late_freq_15": "Late freq. 15 yr"}
    group = lambda f: ("Greenness" if f.startswith("ndvi") else "Weather" if f.startswith("wx")
                       else "Place" if f in ("lat", "lon", "elevation") else "Fire history")
    gcol = {"Greenness": COLOUR["early"], "Weather": RAIN, "Place": "#8a7f6b",
            "Fire history": COLOUR["late"]}
    stats["perm_gb_top"] = [(d["feature"], round(d["mean"], 4)) for d in gb]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    y = np.arange(len(gb))[::-1]
    ax.barh(y, [d["mean"] for d in gb], xerr=[d["sd"] for d in gb],
            color=[gcol[group(d["feature"])] for d in gb], height=0.68,
            error_kw=dict(lw=0.7, capsize=2, ecolor=INK))
    ax.set_yticks(y); ax.set_yticklabels([nice.get(d["feature"], d["feature"]) for d in gb],
                                         fontsize=9)
    ax.set_xlabel("fall in macro F1 when the input is shuffled")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=v, label=k) for k, v in gcol.items()], fontsize=8.5,
              loc="lower right")
    ax.set_title("Permutation importance, gradient boosting (top 12)")
    save(fig, "f15_importance")


def fig_forecast() -> None:
    z = np.load(PROJ / "reports" / "forecast_2026.npz")
    fc = json.load(open(PROJ / "reports" / "forecast_2026.json"))
    lab = pd.read_parquet(INTERIM / "cell_year_labels.parquet", columns=["row", "col", "lon", "lat"])
    shape = (lab["row"].max() + 1, lab["col"].max() + 1)
    ext = [lab["lon"].min() - 0.025, lab["lon"].max() + 0.025,
           lab["lat"].min() - 0.025, lab["lat"].max() + 0.025]
    a = np.full(shape, np.nan)
    a[z["row"], z["col"]] = z["pred"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.6), width_ratios=[1.25, 1])
    ax.imshow(a, extent=ext, origin="upper", interpolation="nearest",
              cmap=ListedColormap([COLOUR[c] for c in CLASSES]), vmin=-0.5, vmax=2.5)
    outline(ax)
    ax.set_xlim(ext[0] - 0.2, ext[1] + 0.2); ax.set_ylim(ext[2] - 0.2, ext[3] + 0.2)
    map_axes(ax, "Forecast class for every square, 2026")
    cats = ["early", "late"]
    x = np.arange(2); w = 0.36
    ax2.bar(x - w / 2, [fc["long_run_share"][c] for c in cats], w, color="#c9d0d3",
            label="2000–2025 average")
    ax2.bar(x + w / 2, [fc["share"][c] for c in cats], w,
            color=[COLOUR[c] for c in cats], label="2026 forecast")
    for i, c in enumerate(cats):
        ax2.text(i - w / 2, fc["long_run_share"][c] + 0.8, f"{fc['long_run_share'][c]:.1f}%",
                 ha="center", fontsize=9)
        ax2.text(i + w / 2, fc["share"][c] + 0.8, f"{fc['share'][c]:.1f}%", ha="center",
                 fontsize=9, weight="bold")
    ax2.set_xticks(x); ax2.set_xticklabels(["Before August", "After July"])
    ax2.set_ylabel("share of the Territory (%)"); ax2.set_ylim(0, 45)
    from matplotlib.patches import Patch
    ax2.legend(handles=[Patch(color="#c9d0d3", label="2000–2025 average"),
                        Patch(color=INK, label="2026 forecast (coloured)")],
               fontsize=8.5, loc="upper left")
    ax2.set_title("Against the long-run average", fontsize=10.5)
    ax.legend(handles=[Patch(color=COLOUR[c], label=NICE[c]) for c in CLASSES],
              loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=3, fontsize=8)
    save(fig, "f16_forecast")


def main() -> None:
    feat, lab = load()
    stats["rows_model"] = int(len(feat))
    stats["rows_labels"] = int(len(lab))
    stats["cells"] = int(lab["cell_id"].nunique())
    stats["class_share_model"] = {c: round(float((feat["label"] == c).mean() * 100), 1)
                                  for c in CLASSES}
    stats["class_share_labels"] = {c: round(float((lab["label"] == c).mean() * 100), 1)
                                   for c in CLASSES}
    for name, yrs in (("train", range(2005, 2020)), ("val", range(2020, 2023)),
                      ("test", range(2023, 2026))):
        sub = feat[feat["year"].isin(yrs)]
        stats[f"rows_{name}"] = int(len(sub))
        stats[f"share_{name}"] = {c: round(float((sub["label"] == c).mean() * 100), 1)
                                  for c in CLASSES}
    stats["missing_values"] = int(feat.isna().sum().sum())

    fig_months(lab)
    fig_years(feat, lab)
    fig_maps(feat, lab)
    fig_bands(feat)
    fig_interaction(feat)
    fig_skew(feat)
    fig_corr(feat)
    fig_kmeans()
    fig_metrics()
    fig_confusion()
    fig_fit()
    fig_walk()
    fig_importance()
    fig_forecast()

    json.dump(stats, open(OUT / "analysis.json", "w"), indent=1, default=str)
    print("wrote", (OUT / "analysis.json").relative_to(PROJ))


if __name__ == "__main__":
    main()
