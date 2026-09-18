"""Measure how fire repeats itself on the same ground.

Four questions, each answered against the base rate rather than in isolation, because
"62% of burnt cells burn again within three years" means nothing until you know what
share of all cells do.

  1. After a fire, when does the next one come?
  2. Can the same ground burn twice inside one season, early and then late?
  3. Does this year's regime predict next year's, and how strongly?
  4. Does burning early actually suppress late-season fire the following year?

The fourth is the one with money attached. Early dry-season burning programmes across
northern Australia exist on the premise that a cool, patchy fire before August removes
the fuel that would otherwise carry a damaging one later. This measures whether the
record agrees.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pyrantis.schema import CLASSES, INTERIM, MIN_BURNT_FRACTION, ROOT as PROJ

MAX_LAG = 10
# The Top End savanna and the arid south burn on entirely different logics, so every
# headline is also reported split. The boundary follows the usual rainfall gradient.
REGIONS = {"Top End (north of 16S)": (-16.0, 0.0),
           "Centre (16S to 20S)": (-20.0, -16.0),
           "Arid south (below 20S)": (-90.0, -20.0)}


def dense(df: pd.DataFrame, col: str, years, nrow, ncol, dtype="float32"):
    g = np.full((len(years), nrow, ncol), np.nan, dtype=dtype)
    yi = df["year"].map({y: i for i, y in enumerate(years)}).to_numpy()
    g[yi, df["row"].to_numpy(), df["col"].to_numpy()] = df[col].to_numpy()
    return g


def main() -> None:
    lab = pd.read_parquet(INTERIM / "cell_year_labels.parquet")
    years = np.sort(lab["year"].unique())
    nrow, ncol = lab["row"].max() + 1, lab["col"].max() + 1

    l = lab["label"].to_numpy()
    burnt = dense(lab.assign(v=(l != "unburnt").astype("f4")), "v", years, nrow, ncol)
    early = dense(lab.assign(v=(l == "early").astype("f4")), "v", years, nrow, ncol)
    late = dense(lab.assign(v=(l == "late").astype("f4")), "v", years, nrow, ncol)
    fe = dense(lab, "frac_early", years, nrow, ncol)
    fl = dense(lab, "frac_late", years, nrow, ncol)
    latg = dense(lab, "lat", years, nrow, ncol)[0]

    inside = ~np.isnan(burnt[0])
    out = {"years": [int(years[0]), int(years[-1])],
           "base_rate": float(np.nanmean(burnt) * 100)}

    def region_mask(name):
        lo, hi = REGIONS[name]
        return inside & (latg > lo) & (latg <= hi)

    # ---------------------------------------------------------- 1. reburn curve
    # P(burns again exactly k years later | burnt this year), against the plain
    # probability that a cell burns in any given year.
    def reburn(mask):
        base = float(np.nanmean(np.where(mask, burnt, np.nan)) * 100)
        curve = []
        for k in range(1, MAX_LAG + 1):
            a, b = burnt[:-k], burnt[k:]
            sel = (a == 1) & mask
            curve.append(round(float(b[sel].mean() * 100), 1) if sel.sum() else None)
        return {"base": round(base, 1), "curve": curve}

    out["reburn"] = {"all": reburn(inside)}
    for r in REGIONS:
        out["reburn"][r] = reburn(region_mask(r))

    # ------------------------------------------------- 2. twice in one season
    # Both an early and a late burn inside the same cell-year, each covering a real
    # share of the cell rather than a stray pixel.
    twice = (fe >= MIN_BURNT_FRACTION) & (fl >= MIN_BURNT_FRACTION) & inside
    anyburn = (burnt == 1) & inside
    out["same_season"] = {
        "share_of_all_cells": round(float(twice.sum() / inside.sum() / len(years) * 100), 2),
        "share_of_burnt_cells": round(float(twice.sum() / anyburn.sum() * 100), 2),
        "by_year": {str(int(y)): round(float(twice[i].sum() / inside.sum() * 100), 2)
                    for i, y in enumerate(years)},
    }

    # -------------------------------------------------- 3. year-to-year transitions
    # P(next year's class | this year's class): a plain 3x3 read of persistence.
    stacks = {"unburnt": (burnt == 0), "early": (early == 1), "late": (late == 1)}
    trans, counts = {}, {}
    for frm, m0 in stacks.items():
        sel = m0[:-1] & inside
        n = int(sel.sum())
        counts[frm] = n
        row = {}
        for to, m1 in stacks.items():
            row[to] = round(float(m1[1:][sel].mean() * 100), 1) if n else None
        trans[frm] = row
    out["transition"] = trans
    out["transition_counts"] = counts

    # ------------------------------------- 4. does early burning suppress late fire?
    # Compared against the two honest alternatives: ground that burnt late last year,
    # and ground that did not burn at all.
    def late_next(prev_mask, mask):
        sel = prev_mask[:-1] & mask
        return (round(float(late[1:][sel].mean() * 100), 1), int(sel.sum())) if sel.sum() \
            else (None, 0)

    def suppression(mask):
        e, ne = late_next(early == 1, mask)
        la, nl = late_next(late == 1, mask)
        u, nu = late_next(burnt == 0, mask)
        return {"after_early": e, "after_late": la, "after_unburnt": u,
                "n_early": ne, "n_late": nl, "n_unburnt": nu,
                "difference": None if (e is None or u is None) else round(u - e, 1)}

    out["suppression"] = {"all": suppression(inside)}
    for r in REGIONS:
        out["suppression"][r] = suppression(region_mask(r))

    # ------------------------------------------------- 5. interval between fires
    # Gaps between one fire and the next on the same ground, over the whole record.
    gaps = []
    b = np.nan_to_num(burnt, nan=0.0) > 0
    rr, cc = np.where(inside)
    for r_, c_ in zip(rr[::7], cc[::7]):          # every seventh cell: same shape, faster
        yrs = np.where(b[:, r_, c_])[0]
        if len(yrs) > 1:
            gaps.extend(np.diff(yrs).tolist())
    gaps = np.array(gaps)
    out["interval"] = {
        "median": float(np.median(gaps)), "mean": round(float(gaps.mean()), 2),
        "n": int(len(gaps)),
        "histogram": {str(k): int((gaps == k).sum()) for k in range(1, 11)},
        "share_within_3": round(float((gaps <= 3).mean() * 100), 1),
    }

    (PROJ / "reports").mkdir(exist_ok=True)
    json.dump(out, open(PROJ / "reports" / "recurrence.json", "w"), indent=2)

    # ------------------------------------------------------------------ report
    R = out["reburn"]["all"]
    print(f"base rate: a cell burns in {R['base']}% of years\n")
    print("after a fire, chance of burning again k years later")
    for k, v in enumerate(R["curve"], 1):
        bar = "#" * int(round(v / 2))
        print(f"  +{k:<2} {v:5.1f}%  {bar}")
    print(f"\nsame ground burning twice in one season: "
          f"{out['same_season']['share_of_burnt_cells']}% of burnt cell-years")
    print(f"median gap between fires: {out['interval']['median']:.0f} years; "
          f"{out['interval']['share_within_3']}% of gaps are 3 years or less")

    print("\nnext year's regime, given this year's (%)")
    print(f"  {'from':<10}" + "".join(f"{c:>10}" for c in CLASSES))
    for frm in CLASSES:
        print(f"  {frm:<10}" + "".join(f"{trans[frm][c]:>10.1f}" for c in CLASSES))

    print("\nchance of a LATE fire next year, by what happened this year")
    for name, s in out["suppression"].items():
        print(f"  {name:<24} after early {s['after_early']:>5}%   "
              f"after unburnt {s['after_unburnt']:>5}%   "
              f"after late {s['after_late']:>5}%   diff {s['difference']:+.1f}")


if __name__ == "__main__":
    main()
