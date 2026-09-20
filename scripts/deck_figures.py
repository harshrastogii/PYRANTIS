"""Figures drawn for a projected slide, not for a page.

Everything is sized for a room: type large enough to read from the back, chrome stripped
to what carries meaning, and one idea per figure. They are drawn light because each one
sits in a pale card on a dark slide.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pyrantis.schema import CLASSES, INTERIM, PROC, ROOT as PROJ

OUT = PROJ / "reports" / "deck"
INK = "#14202b"
MUTED = "#5d6f80"
COL = {"unburnt": "#93a4b1", "early": "#15805f", "late": "#c8441a"}
ACCENT = "#c8441a"
COOL = "#1f6f9b"

plt.rcParams.update({
    "figure.dpi": 160, "savefig.dpi": 220, "font.size": 15,
    "font.family": "DejaVu Sans",
    "axes.edgecolor": "#c9d3db", "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {name}")


def fig_fire_years(lab):
    share = (lab.groupby(["year", "label"], observed=True).size()
             .unstack(fill_value=0).pipe(lambda d: d.div(d.sum(1), axis=0)) * 100)
    fig, ax = plt.subplots(figsize=(11, 3.9))
    bottom = np.zeros(len(share))
    for c in CLASSES:
        ax.bar(share.index, share[c], bottom=bottom, color=COL[c], width=.8,
               edgecolor="white", linewidth=.6, label=c)
        bottom += share[c].to_numpy()
    ax.set_ylim(0, 100); ax.set_ylabel("share of the Territory (%)")
    ax.legend(frameon=False, ncol=3, loc="lower left", bbox_to_anchor=(0, 1.01), fontsize=14)
    ax.margins(x=.01)
    save(fig, "fire_years.png")


def fig_models(res):
    bal = sorted([r for r in res["results"] if "(balanced)" in r["model"]],
                 key=lambda r: r["f1_macro"])
    base = {r["model"]: r for r in res["results"]
            if r["model"] in ("same as last year", "majority class")}
    names = [r["model"].replace(" (balanced)", "") for r in bal]
    vals = [r["f1_macro"] for r in bal]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    cols = [ACCENT if v == max(vals) else "#b9c6d1" for v in vals]
    ax.barh(names, vals, color=cols, height=.66)
    for i, v in enumerate(vals):
        ax.text(v + .008, i, f"{v:.3f}", va="center", fontsize=14,
                fontweight="bold" if v == max(vals) else "normal")
    b = base["same as last year"]["f1_macro"]
    ax.axvline(b, color=INK, lw=1.4, ls="--")
    ax.text(b, len(vals) - .35, "  copying last year", fontsize=13, color=INK, va="center")
    ax.set_xlim(0, max(vals) + .09); ax.set_xlabel("balanced F1 on years the model never saw")
    save(fig, "models.png")


def fig_confusion(res):
    best = max((r for r in res["results"] if "(balanced)" in r["model"]),
               key=lambda r: r["f1_macro"])
    m = np.array(best["confusion"], float)
    m = m / m.sum(1, keepdims=True) * 100
    lab = ["no fire", "before Aug", "after Jul"]
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    ax.imshow(m, cmap="Oranges", vmin=0, vmax=100)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{m[i, j]:.0f}", ha="center", va="center", fontsize=19,
                    fontweight="bold" if i == j else "normal",
                    color="white" if m[i, j] > 55 else INK)
    ax.set_xticks(range(3), lab, fontsize=13)
    ax.set_yticks(range(3), lab, fontsize=13)
    ax.set_xlabel("what the model said"); ax.set_ylabel("what happened")
    for s in ax.spines.values():
        s.set_visible(False)
    save(fig, "confusion.png")


def fig_walk(walk):
    byh = defaultdict(list)
    for f in walk["folds"]:
        best = max(v["f1_macro"] for k, v in f["models"].items()
                   if k not in ("same as last year", "majority class"))
        base = f["models"]["same as last year"]["f1_macro"]
        byh[f["horizon"]].append((best, base))
    hs = sorted(byh)
    best = [np.mean([b for b, _ in byh[h]]) for h in hs]
    basel = [np.mean([x for _, x in byh[h]]) for h in hs]
    fig, ax = plt.subplots(figsize=(10, 4.3))
    ax.plot(hs, best, "-o", color=ACCENT, lw=3, ms=8, label="our model")
    ax.plot(hs, basel, "-o", color="#b9c6d1", lw=2.4, ms=6, label="copying last year")
    ax.fill_between(hs, basel, best, color=ACCENT, alpha=.08)
    ax.set_xticks(hs); ax.set_ylim(0, .8)
    ax.set_xlabel("years ahead of the last year it learned from")
    ax.set_ylabel("balanced F1")
    ax.legend(frameon=False, fontsize=14, loc="lower left")
    save(fig, "walkforward.png")


def fig_experiments(exp):
    runs = [r for r in exp["runs"] if r.get("f1_macro")]
    fig, ax = plt.subplots(figsize=(11, 3.9))
    x = np.arange(len(runs))
    vals = [r["f1_macro"] for r in runs]
    keep = [r.get("verdict") == "KEEP" for r in runs]
    ax.bar(x, vals, color=[ACCENT if k else "#d5dee5" for k in keep], width=.7)
    ax.axhline(0.674, color=INK, lw=1.4, ls="--")
    ax.text(len(runs) - .5, .681, "where we started", ha="right", fontsize=13, color=INK)
    k = keep.index(True)
    ax.annotate("the one we kept:\nsatellite greenness",
                xy=(k, vals[k]), xytext=(k - 6.2, .40), fontsize=14, color=ACCENT,
                fontweight="bold", linespacing=1.4,
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=2))
    ax.set_ylim(0, .8); ax.set_xticks([])
    ax.set_ylabel("balanced F1"); ax.set_xlabel(f"{len(runs)} experiments, in the order we ran them")
    save(fig, "experiments.png")


def fig_suppression(rec):
    s = rec["suppression"]
    regions = [k for k in s if k != "all"]
    names, early, alone = [], [], []
    for k in ["all"] + regions:
        names.append("Whole Territory" if k == "all" else k.split(" (")[0])
        early.append(s[k]["after_early"]); alone.append(s[k]["after_unburnt"])
    y = np.arange(len(names)); h = .36
    fig, ax = plt.subplots(figsize=(10, 4.4))
    ax.barh(y + h / 2, early, h, color=COL["early"], label="burned early last year")
    ax.barh(y - h / 2, alone, h, color="#b9c6d1", label="left alone last year")
    for i, (e, a) in enumerate(zip(early, alone)):
        ax.text(e + .6, i + h / 2, f"{e}%", va="center", fontsize=13)
        ax.text(a + .6, i - h / 2, f"{a}%", va="center", fontsize=13)
    ax.set_yticks(y, names, fontsize=14)
    ax.invert_yaxis()
    ax.set_xlabel("chance of a fire after July, next year (%)")
    ax.set_xlim(0, max(max(early), max(alone)) + 7)
    ax.legend(frameon=False, fontsize=13.5, loc="lower right")
    save(fig, "suppression.png")


def fig_months(lab):
    b = lab[lab["label"] != "unburnt"]
    v = (b["peak_month"][b["peak_month"] > 0].value_counts(normalize=True).sort_index() * 100)
    names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    cols = [COL["late"] if m >= 8 else COL["early"] if m >= 4 else COL["unburnt"]
            for m in v.index]
    ax.bar([names[m - 1] for m in v.index], v.values, color=cols, width=.72)
    for i, (m, val) in enumerate(zip(v.index, v.values)):
        ax.text(i, val + .4, f"{val:.0f}", ha="center", fontsize=13)
    ax.set_ylabel("share of all fires (%)"); ax.set_ylim(0, max(v.values) + 3)
    save(fig, "months.png")


def fig_ndvi():
    nd = pd.read_parquet(PROC / "ndvi.parquet")
    apr = nd.groupby("year")["ndvi_apr"].mean()
    fig, ax = plt.subplots(figsize=(10, 3.9))
    cols = [ACCENT if y == apr.idxmax() else "#b9c6d1" for y in apr.index]
    ax.bar(apr.index, apr.values, color=cols, width=.74)
    top = apr.idxmax()
    ax.annotate(f"{top}: the most grass\nin the whole record",
                xy=(top, apr.max()), xytext=(top - 9, apr.max() + .045),
                fontsize=14, color=ACCENT, fontweight="bold", linespacing=1.4,
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=2))
    ax.set_ylabel("April greenness"); ax.set_ylim(0, apr.max() + .12)
    ax.margins(x=.01)
    save(fig, "ndvi.png")


def fig_reburn(rec):
    rb = rec["reburn"]["all"]
    fig, ax = plt.subplots(figsize=(10, 4.1))
    x = np.arange(1, len(rb["curve"]) + 1)
    ax.bar(x, rb["curve"], color=COL["late"], width=.66)
    ax.axhline(rb["base"], color=INK, lw=1.6, ls="--")
    ax.text(len(x) + .1, rb["base"], f"  {rb['base']}% in any year", va="center",
            fontsize=13.5, color=INK)
    ax.set_xticks(x); ax.set_ylim(0, 70)
    ax.set_xlabel("years after a fire"); ax.set_ylabel("chance of burning again (%)")
    save(fig, "reburn.png")


def main():
    lab = pd.read_parquet(INTERIM / "cell_year_labels.parquet")
    res = json.load(open(PROJ / "reports" / "results_full.json"))
    rec = json.load(open(PROJ / "reports" / "recurrence.json"))
    walk = json.load(open(PROJ / "reports" / "walk_forward.json"))
    exp = json.load(open(PROJ / "reports" / "experiments.json"))
    print("writing deck figures")
    fig_fire_years(lab); fig_models(res); fig_confusion(res)
    fig_walk(walk); fig_experiments(exp); fig_suppression(rec)
    fig_months(lab); fig_ndvi(); fig_reburn(rec)


if __name__ == "__main__":
    main()
