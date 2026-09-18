"""Build the figures that carry the argument of the report.

Four of them, in the order the story is told: what the Territory's fire years actually
look like, which models beat which baselines, where each model's errors go, and which
inputs the forest leaned on.
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pyrantis.schema import CLASSES, FIGURES, INTERIM, ROOT as PROJ

# Unburnt reads as quiet ground, early as the green flush of a managed burn, late as the
# hot end of the season. Chosen to stay distinguishable in greyscale print.
COLOUR = {"unburnt": "#b9c2c7", "early": "#2e8b74", "late": "#c1440e"}
INK = "#22282b"

plt.rcParams.update({
    "figure.dpi": 140, "savefig.dpi": 200, "font.size": 9,
    "axes.edgecolor": INK, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False,
    "axes.spines.right": False, "figure.facecolor": "white",
})


def load(tag: str):
    p = PROJ / "reports" / f"results{tag}.json"
    return json.load(open(p)) if p.exists() else None


def fig_fire_years() -> None:
    """The interannual signal the models have to cope with."""
    df = pd.read_parquet(INTERIM / "cell_year_labels.parquet")
    share = (df.groupby(["year", "label"], observed=True).size()
             .unstack(fill_value=0).pipe(lambda d: d.div(d.sum(1), axis=0)) * 100)

    fig, ax = plt.subplots(figsize=(9.5, 3.6))
    bottom = np.zeros(len(share))
    for c in CLASSES:
        ax.bar(share.index, share[c], bottom=bottom, color=COLOUR[c],
               label=c, width=0.78, edgecolor="white", linewidth=0.4)
        bottom += share[c].to_numpy()

    ax.set_ylim(0, 100)
    ax.set_ylabel("share of NT cells (%)")
    ax.set_title("Fire regime across the Northern Territory, 2000–2025", loc="left",
                 fontsize=11, weight="bold", pad=44)
    ax.legend(frameon=False, ncol=3, loc="lower left", bbox_to_anchor=(0, 1.005))
    # The bars fill the axes from 0 to 100, so the note lives above them rather than
    # over the very years it is drawing attention away from.
    ax.annotate("2011: two thirds of the Territory burns,\nfire reaching the arid centre "
                "after the 2010–11 La Niña",
                xy=(2011, 100), xytext=(0.995, 1.02), textcoords="axes fraction",
                fontsize=7.5, ha="right", va="bottom", linespacing=1.4,
                arrowprops=dict(arrowstyle="-", lw=0.7, color=INK,
                                connectionstyle="angle,angleA=0,angleB=90,rad=3"))
    ax.margins(x=0.01)
    fig.tight_layout()
    fig.savefig(FIGURES / "fire_years.png", bbox_inches="tight")
    plt.close(fig)


def fig_models(runs: dict) -> None:
    """Macro-F1 for every model, against the two baselines."""
    tag, res = next(iter(runs.items()))
    rows = [r for r in res["results"] if r["model"] not in ("majority class", "same as last year")]
    base = {r["model"]: r for r in res["results"]
            if r["model"] in ("majority class", "same as last year")}

    names = sorted({r["model"].replace(" (balanced)", "") for r in rows})
    plain = {n: 0.0 for n in names}
    bal = {n: 0.0 for n in names}
    for r in rows:
        (bal if "(balanced)" in r["model"] else plain)[
            r["model"].replace(" (balanced)", "")] = r["f1_macro"]
    order = sorted(names, key=lambda n: -max(plain[n], bal[n]))

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2),
                             gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    y = np.arange(len(order))
    ax.barh(y - 0.19, [plain[n] for n in order], 0.38, color="#c8d2d6",
            label="as trained", edgecolor="white")
    ax.barh(y + 0.19, [bal[n] for n in order], 0.38, color="#2e8b74",
            label="class-weighted", edgecolor="white")
    for k, n in enumerate(order):
        ax.text(bal[n] + 0.008, k + 0.19, f"{bal[n]:.3f}", va="center", fontsize=7.5)
    # The baselines go in the legend rather than as text beside the lines, which
    # collided with the axis label at every figure size worth using.
    for name, style in (("same as last year", "-"), ("majority class", ":")):
        ax.axvline(base[name]["f1_macro"], ls=style, lw=1.1, color=INK, alpha=0.7,
                   label=f"baseline: {name}")
    ax.set_yticks(y, order)
    ax.invert_yaxis()
    ax.set_xlabel("macro-averaged F1 on 2023–2025")
    ax.set_xlim(0, max(max(bal.values()), 0.7) + 0.1)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="lower left",
              bbox_to_anchor=(0, 1.0), columnspacing=1.4)
    ax.set_title("No model beats a decision tree by much", loc="left",
                 fontsize=11, weight="bold", pad=34)

    ax = axes[1]
    best = max(rows, key=lambda r: r["f1_macro"])
    w = 0.38
    x = np.arange(len(CLASSES))
    worst = next(r for r in rows if r["model"] == best["model"].replace(" (balanced)", ""))
    ax.bar(x - w / 2, [worst["f1_per_class"][c] for c in CLASSES], w,
           color="#c8d2d6", label="as trained", edgecolor="white")
    ax.bar(x + w / 2, [best["f1_per_class"][c] for c in CLASSES], w,
           color=[COLOUR[c] for c in CLASSES], edgecolor="white")
    ax.set_xticks(x, CLASSES)
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1)
    # The weighted bars take one colour per class, so the legend needs proxy handles
    # rather than a swatch borrowed from whichever class happened to be drawn first.
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="#c8d2d6", label="as trained"),
                       Patch(facecolor="#7a8b91", label="class-weighted")],
              frameon=False, fontsize=8, ncol=2, loc="lower left",
              bbox_to_anchor=(0, 1.0))
    ax.set_title(f"Where the gain comes from: {best['model'].replace(' (balanced)','')}",
                 loc="left", fontsize=11, weight="bold", pad=24)
    fig.tight_layout()
    fig.savefig(FIGURES / "model_comparison.png", bbox_inches="tight")
    plt.close(fig)


def fig_confusion(runs: dict) -> None:
    tag, res = next(iter(runs.items()))
    picks = [r for r in res["results"] if "(balanced)" in r["model"]]
    picks = sorted(picks, key=lambda r: -r["f1_macro"])[:4]

    fig, axes = plt.subplots(1, len(picks), figsize=(3.1 * len(picks), 3.3))
    for ax, r in zip(np.atleast_1d(axes), picks):
        m = np.array(r["confusion"], dtype=float)
        m = m / m.sum(1, keepdims=True) * 100
        ax.imshow(m, cmap="BuGn", vmin=0, vmax=100)
        for i in range(len(CLASSES)):
            for j in range(len(CLASSES)):
                ax.text(j, i, f"{m[i, j]:.0f}", ha="center", va="center", fontsize=8.5,
                        color="white" if m[i, j] > 55 else INK)
        ax.set_xticks(range(len(CLASSES)), CLASSES, fontsize=7.5)
        ax.set_yticks(range(len(CLASSES)), CLASSES, fontsize=7.5)
        ax.set_title(r["model"].replace(" (balanced)", ""), fontsize=9.5, weight="bold")
        ax.set_xlabel("predicted", fontsize=8)
    np.atleast_1d(axes)[0].set_ylabel("actual", fontsize=8)
    fig.suptitle("Row-normalised confusion, 2023–2025 (%)", x=0.02, ha="left",
                 fontsize=11, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIGURES / "confusion.png", bbox_inches="tight")
    plt.close(fig)


def fig_importance(tag: str) -> None:
    p = PROJ / "reports" / f"feature_importance{tag}.json"
    if not p.exists():
        return
    imp = json.load(open(p))
    top = list(imp.items())[:15][::-1]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    cols = ["#c1440e" if k.startswith("wx_") else "#2e8b74" for k, _ in top]
    ax.barh([k for k, _ in top], [v for _, v in top], color=cols, edgecolor="white")
    ax.set_xlabel("random forest importance")
    ax.set_title("What the forest leans on", loc="left", fontsize=11, weight="bold", pad=10)
    if any(k.startswith("wx_") for k in imp):
        ax.text(0.98, 0.04, "orange = weather", transform=ax.transAxes, ha="right",
                fontsize=8, color="#c1440e")
    fig.tight_layout()
    fig.savefig(FIGURES / f"importance{tag}.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    runs = {t: r for t in ("_weather", "_firehistory") if (r := load(t))}
    assert runs, "no results files; run scripts/train_models.py first"
    tag = next(iter(runs))

    fig_fire_years()
    fig_models(runs)
    fig_confusion(runs)
    fig_importance(tag)
    for f in sorted(FIGURES.glob("*.png")):
        print(f"  {f.relative_to(PROJ)}  {f.stat().st_size / 1000:.0f} kB")


if __name__ == "__main__":
    main()
