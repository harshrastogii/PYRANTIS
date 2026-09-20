"""Render the presentation's charts in the product's own colour language.

The deck reuses the three class colours the web product uses -- slate for
unburnt, green for early, ember for late -- so a reader who has seen the map
already knows what every bar means before the speaker says a word. Charts are
drawn here rather than as native PowerPoint charts because the palette, the
baseline markers and the annotation placement all matter more than being
editable inside PowerPoint afterwards.
"""

from __future__ import annotations

import json
import pathlib
import collections
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = pathlib.Path(__file__).resolve().parents[1] / "reports"
OUT = R / "deck"
OUT.mkdir(parents=True, exist_ok=True)

INK      = "#12303C"
BLUE     = "#245F80"
BLUE_L   = "#4A90B4"
EMBER    = "#C8552F"
EMBER_L  = "#E0835C"
GREEN    = "#2E7D5B"
SLATE    = "#9AA8AE"
MUTED    = "#6B7C85"
PAPER    = "#F7F5F2"
RULE     = "#DCD8D2"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Calibri", "Helvetica Neue", "Arial", "DejaVu Sans"],
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": INK,
    "axes.edgecolor": RULE,
    "figure.facecolor": PAPER,
    "axes.facecolor": PAPER,
    "savefig.facecolor": PAPER,
})

j = lambda n: json.loads((R / n).read_text())
DPI = 240


def _finish(fig, name):
    fig.savefig(OUT / name, dpi=DPI, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    print(f"  {name}")


def _bare(ax, grid_x=True):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(RULE)
    ax.tick_params(length=0)
    if grid_x:
        ax.xaxis.grid(True, color=RULE, lw=0.8)
        ax.set_axisbelow(True)


def models():
    """Held-out scores, with the two baselines pushed back in slate."""
    res = j("results_full.json")["results"]
    keep = ["gradient boosting (balanced)", "LSTM (balanced)", "multilayer ANN (balanced)",
            "random forest (balanced)", "decision tree (balanced)", "naive bayes (balanced)",
            "same as last year", "majority class"]
    label = {"gradient boosting (balanced)": "gradient boosting", "LSTM (balanced)": "LSTM",
             "multilayer ANN (balanced)": "multilayer ANN", "random forest (balanced)": "random forest",
             "decision tree (balanced)": "decision tree", "naive bayes (balanced)": "naive bayes",
             "same as last year": "same as last year", "majority class": "majority class"}
    rows = [(label[k], r["f1_macro"]) for k in keep
            for r in res if r["model"] == k]
    rows.sort(key=lambda t: t[1])
    names = [n for n, _ in rows]
    vals = [v for _, v in rows]
    cols = [SLATE if n in ("same as last year", "majority class") else BLUE for n in names]
    cols[names.index("gradient boosting")] = EMBER

    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    y = np.arange(len(names))
    ax.barh(y, vals, color=cols, height=0.62)
    for i, v in enumerate(vals):
        ax.text(v + 0.008, i, f"{v:.3f}", va="center", ha="left",
                fontsize=12, color=INK, fontweight="bold" if cols[i] == EMBER else "normal")
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=12.5)
    ax.set_xlim(0, 0.80); ax.set_xlabel("balanced F1 on 2023-2025, never seen in training", fontsize=11)
    _bare(ax)
    _finish(fig, "deck_models.png")


def layers():
    """What each new source of data actually bought."""
    steps = [("fire history\nalone", 0.6394, 22), ("+ weather", 0.6739, 36), ("+ greenness", 0.6842, 50)]
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    x = np.arange(3)
    vals = [s[1] for s in steps]
    ax.bar(x, vals, color=[BLUE_L, BLUE, EMBER], width=0.52)
    base = 0.5063
    # The dashed baseline stops short of the left edge, leaving a gutter for its
    # own caption rather than running the caption across the first bar.
    ax.hlines(base, -0.34, 2.34, color=SLATE, lw=1.4, ls=(0, (4, 3)))
    ax.text(-0.92, base, "same as\nlast year\n0.506", ha="left", va="center",
            fontsize=10.5, color=MUTED, linespacing=1.35)
    for i, (lab, v, nf) in enumerate(steps):
        ax.text(i, v + 0.004, f"{v:.3f}", ha="center", va="bottom", fontsize=15,
                fontweight="bold", color=INK)
        ax.text(i, 0.545, f"{nf} features", ha="center", va="bottom", fontsize=10.5, color="white")
    for i in (1, 2):
        d = vals[i] - vals[i - 1]
        ax.annotate("", xy=(i, vals[i]), xytext=(i - 1, vals[i - 1]),
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.9, ls=":"))
        ax.text(i - 0.5, max(vals[i], vals[i - 1]) + 0.016, f"+{d:.3f}",
                ha="center", fontsize=11.5, color=EMBER, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([s[0] for s in steps], fontsize=12.5)
    ax.set_xlim(-0.98, 2.44)
    ax.set_ylim(0.49, 0.72); ax.set_yticks([])
    ax.set_ylabel("balanced F1", fontsize=11)
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(RULE); ax.tick_params(length=0)
    _finish(fig, "deck_layers.png")


def walk():
    """Skill against how far ahead the model was asked to guess."""
    w = j("walk_forward.json")
    by, base = collections.defaultdict(list), collections.defaultdict(list)
    for f in w["folds"]:
        real = [v["f1_macro"] for k, v in f["models"].items()
                if k not in ("same as last year", "majority class")]
        by[f["horizon"]].append(max(real))
        base[f["horizon"]].append(f["models"]["same as last year"]["f1_macro"])
    hs = sorted(by)
    mv = [statistics.mean(by[h]) for h in hs]
    bv = [statistics.mean(base[h]) for h in hs]

    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    ax.plot(hs, mv, color=EMBER, lw=2.6, marker="o", ms=7, zorder=3, label="the model")
    ax.plot(hs, bv, color=SLATE, lw=2.0, ls=(0, (4, 3)), marker="o", ms=5, label="same as last year")
    ax.fill_between(hs, bv, mv, color=EMBER, alpha=0.07, zorder=1)
    ax.set_xticks(hs); ax.set_xticklabels([str(h) for h in hs], fontsize=11.5)
    ax.set_xlabel("years ahead of the last year it was allowed to learn from", fontsize=11)
    ax.set_ylabel("balanced F1", fontsize=11)
    ax.set_ylim(0.44, 0.72)
    ax.legend(frameon=False, fontsize=11.5, loc="upper right", ncol=2)
    ax.yaxis.grid(True, color=RULE, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(RULE); ax.tick_params(length=0)
    ax.annotate(f"{mv[0]:.3f}", (hs[0], mv[0]), textcoords="offset points", xytext=(2, 12),
                fontsize=12, color=EMBER, fontweight="bold")
    ax.annotate(f"{mv[-1]:.3f}", (hs[-1], mv[-1]), textcoords="offset points", xytext=(-6, -20),
                fontsize=12, color=EMBER, fontweight="bold")
    _finish(fig, "deck_walk.png")


def experiments():
    """The ledger: 23 runs against a fixed benchmark, one kept."""
    runs = j("experiments.json")["runs"]
    bench = [r for r in runs if r["name"] == "gb-base"][0]["f1_macro"]
    rs = sorted(runs, key=lambda r: r["f1_macro"])
    names = [r["name"] for r in rs]
    d = [r["f1_macro"] - bench for r in rs]
    cols = [EMBER if r["name"] == "r3-gb-ndvi" else (SLATE if r["f1_macro"] < bench else BLUE_L)
            for r in rs]

    fig, ax = plt.subplots(figsize=(9.8, 5.2))
    y = np.arange(len(names))
    ax.barh(y, d, color=cols, height=0.66)
    ax.axvline(0, color=INK, lw=1.4)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=9.6)
    ax.set_xlabel("change in balanced F1 against the fixed benchmark", fontsize=11)
    i = names.index("r3-gb-ndvi")
    ax.text(d[i] + 0.0015, i, "  kept", va="center", fontsize=11.5, color=EMBER, fontweight="bold")
    k = names.index("cnn-unet")
    ax.text(d[k] + 0.0018, k, "the CNN", ha="left", va="center", fontsize=11, color="white")
    b = names.index("gb-base")
    ax.text(0.0012, b, "the benchmark", va="center", ha="left", fontsize=10.5, color=MUTED)
    _bare(ax)
    _finish(fig, "deck_experiments.png")


def forecast():
    """2026 against the twenty-six year average."""
    fc = j("forecast_2026.json")
    s, lr = fc["share"], fc["long_run_share"]
    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    order = [("late", "fire after July", EMBER), ("early", "fire before August", GREEN),
             ("unburnt", "no fire", SLATE)]
    left_a = left_b = 0.0
    for key, lab, col in order:
        ax.barh(1, s[key], left=left_a, color=col, height=0.52)
        ax.text(left_a + s[key] / 2, 1, f"{s[key]:.1f}%", ha="center", va="center",
                color="white", fontsize=13.5, fontweight="bold")
        left_a += s[key]
        ax.barh(0, lr[key], left=left_b, color=col, height=0.52, alpha=0.42)
        ax.text(left_b + lr[key] / 2, 0, f"{lr[key]:.1f}%", ha="center", va="center",
                color="white", fontsize=12.5)
        left_b += lr[key]
    ax.set_yticks([1, 0]); ax.set_yticklabels(["2026 forecast", "2000-2025 average"], fontsize=13)
    ax.set_xlim(0, 100); ax.set_xticks([])
    for s_ in ("top", "right", "left", "bottom"): ax.spines[s_].set_visible(False)
    ax.tick_params(length=0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in order]
    ax.legend(handles, [l for _, l, _ in order], frameon=False, fontsize=11.5,
              loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=3)
    _finish(fig, "deck_forecast.png")


def classbalance():
    """Why a model that ignores the rare answers still looks accurate."""
    fig, ax = plt.subplots(figsize=(7.6, 2.5))
    share = [69.0, 14.1, 16.9]
    labs = ["no fire", "fire before August", "fire after July"]
    cols = [SLATE, GREEN, EMBER]
    left = 0.0
    for v, l, c in zip(share, labs, cols):
        ax.barh(0, v, left=left, color=c, height=0.5)
        ax.text(left + v / 2, 0, f"{v:.1f}%", ha="center", va="center", color="white",
                fontsize=14, fontweight="bold")
        left += v
    ax.set_xlim(0, 100); ax.set_ylim(-0.62, 0.36)
    ax.axis("off")
    # The two rare classes are too narrow to sit a caption under, so the names go
    # in a legend rather than colliding above the bar.
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in cols]
    ax.legend(handles, labs, frameon=False, fontsize=12, loc="upper center",
              bbox_to_anchor=(0.5, 0.16), ncol=3, handlelength=1.1, handleheight=1.1)
    _finish(fig, "deck_balance.png")


def suppression():
    """Early burning, Territory-wide against the Top End."""
    rec = j("recurrence.json")["suppression"]
    groups = [("Whole Territory", rec["all"]), ("Top End (north of 16S)", rec["Top End (north of 16S)"])]
    fig, ax = plt.subplots(figsize=(8.2, 3.9))
    x = np.arange(2); w = 0.33
    early = [g[1]["after_early"] for g in groups]
    alone = [g[1]["after_unburnt"] for g in groups]
    ax.bar(x - w / 2, early, w, color=GREEN, label="burned early the year before")
    ax.bar(x + w / 2, alone, w, color=SLATE, label="left alone the year before")
    for i in range(2):
        ax.text(i - w / 2, early[i] + 0.7, f"{early[i]:.1f}%", ha="center", fontsize=13,
                fontweight="bold", color=INK)
        ax.text(i + w / 2, alone[i] + 0.7, f"{alone[i]:.1f}%", ha="center", fontsize=13,
                fontweight="bold", color=INK)
    ax.set_xticks(x); ax.set_xticklabels([g[0] for g in groups], fontsize=12.5)
    ax.set_ylabel("chance of a fire after July, next year", fontsize=11)
    ax.set_ylim(0, 38)
    ax.legend(frameon=False, fontsize=11.5, loc="upper left")
    ax.yaxis.grid(True, color=RULE, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(RULE); ax.tick_params(length=0)
    _finish(fig, "deck_suppression.png")


def importance():
    """What the model leans on, coloured by where the number came from."""
    fi = j("feature_importance_full.json")
    items = fi if isinstance(fi, list) else list(fi.items())
    norm = []
    for it in items:
        if isinstance(it, dict):
            norm.append((it.get("feature"), it.get("importance")))
        else:
            norm.append((it[0], it[1]))
    norm = [(f, v) for f, v in norm if v is not None]
    norm.sort(key=lambda t: -t[1])
    top = norm[:11][::-1]
    pretty = {"wx_wet_vpd": "wet-season dryness of the air", "ndvi_wet_mean": "average greenness, wet season",
              "wx_wet_rain": "wet-season rainfall", "nbr_frac_early_prev": "neighbours burned early last year",
              "ndvi_wet_min": "lowest greenness, wet season", "wx_wet_rain_prev": "last wet season's rainfall",
              "ndvi_apr": "greenness at the end of April", "lat": "latitude", "lon": "longitude",
              "wx_wet_radn": "wet-season sunlight", "wx_dry_rain_prev": "last dry season's rainfall",
              "wx_dry_tmax_prev": "last dry season's heat"}

    def src(f):
        if f.startswith("ndvi"): return GREEN
        if f.startswith("wx"): return EMBER
        return BLUE

    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    y = np.arange(len(top))
    ax.barh(y, [v for _, v in top], color=[src(f) for f, _ in top], height=0.64)
    ax.set_yticks(y); ax.set_yticklabels([pretty.get(f, f) for f, _ in top], fontsize=11.8)
    ax.set_xlabel("how much the model leans on it", fontsize=11)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (EMBER, GREEN, BLUE)]
    ax.legend(handles, ["weather", "greenness", "fire history and place"],
              frameon=False, fontsize=11.5, loc="lower right")
    _bare(ax)
    _finish(fig, "deck_importance.png")


if __name__ == "__main__":
    print("rendering deck figures")
    models(); layers(); walk(); experiments()
    forecast(); classbalance(); suppression(); importance()
    print("done ->", OUT)
