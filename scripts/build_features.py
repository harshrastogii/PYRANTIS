"""Derive fire-history features for each cell-year from the label table.

Every feature here answers a question that could have been asked on 1 January of the
target year: how long since this place last burnt, how often it has burnt, whether its
neighbours burnt last year. Nothing reads the target year itself or any year after it.
That is the whole discipline of this file -- a fire-history feature that peeked at the
current year would predict the label almost perfectly and mean nothing.

The mechanics: the cell-year table is inflated back into dense (year, row, col) grids,
walked forward one year at a time carrying only past state, then flattened back to a
table. The grid form is what makes neighbour features a shift instead of a spatial join.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pyrantis.schema import CLASSES, MIN_HISTORY_YEARS, PROC, INTERIM

LABELS = INTERIM / "cell_year_labels.parquet"
OUT = PROC / "cell_year_features.parquet"

# Windows over which fire frequency is counted, in years.
WINDOWS = (3, 5, 10)


def to_grid(df: pd.DataFrame, col: str, nrow: int, ncol: int, years: np.ndarray) -> np.ndarray:
    """Inflate one column of the long table into a dense (year, row, col) array.

    Cells outside the NT never appear in the table and stay NaN, which keeps them out
    of every sum and mean that follows.
    """
    g = np.full((len(years), nrow, ncol), np.nan, dtype="float32")
    yi = np.searchsorted(years, df["year"].to_numpy())
    g[yi, df["row"].to_numpy(), df["col"].to_numpy()] = df[col].to_numpy()
    return g


def neighbour_mean(a: np.ndarray) -> np.ndarray:
    """Mean of the eight surrounding cells, ignoring cells outside the Territory.

    Edge cells and coastal cells simply average over fewer neighbours rather than
    being padded with zeros, which would invent unburnt land in the Arafura Sea.
    """
    filled = np.nan_to_num(a, nan=0.0)
    present = (~np.isnan(a)).astype("float32")
    tot = np.zeros_like(filled)
    cnt = np.zeros_like(present)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            tot += np.roll(np.roll(filled, dr, axis=0), dc, axis=1)
            cnt += np.roll(np.roll(present, dr, axis=0), dc, axis=1)
    # np.roll wraps around the edges; blank the wrapped row/column so the top of the
    # Territory does not borrow neighbours from the bottom of it.
    for arr in (tot, cnt):
        arr[0, :] = arr[-1, :] = np.nan
        arr[:, 0] = arr[:, -1] = np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, tot / cnt, np.nan)


def main() -> None:
    df = pd.read_parquet(LABELS)
    years = np.sort(df["year"].unique())
    nrow, ncol = df["row"].max() + 1, df["col"].max() + 1
    print(f"read {len(df):,} cell-years, grid {nrow} x {ncol}, {years[0]}-{years[-1]}")

    frac_burnt = to_grid(df, "frac_burnt", nrow, ncol, years)
    frac_early = to_grid(df, "frac_early", nrow, ncol, years)
    frac_late = to_grid(df, "frac_late", nrow, ncol, years)

    lab = df["label"].to_numpy()
    burnt = to_grid(df.assign(v=(lab != "unburnt").astype("f4")), "v", nrow, ncol, years)
    early = to_grid(df.assign(v=(lab == "early").astype("f4")), "v", nrow, ncol, years)
    late = to_grid(df.assign(v=(lab == "late").astype("f4")), "v", nrow, ncol, years)

    inside = ~np.isnan(frac_burnt[0])
    rows = []

    for t, year in enumerate(years):
        if t < MIN_HISTORY_YEARS:
            continue  # too little history for the windows to mean anything

        # Everything below slices [:t] -- strictly years before the target year.
        past_burnt = burnt[:t]
        past_late = late[:t]

        f = {}
        f["hist_years"] = float(t)

        for w in WINDOWS:
            win = past_burnt[max(0, t - w):t]
            f[f"freq_{w}"] = np.nansum(win, axis=0) / win.shape[0]
            winl = past_late[max(0, t - w):t]
            f[f"late_freq_{w}"] = np.nansum(winl, axis=0) / winl.shape[0]

        # Time since the last fire of each kind. argmax on the reversed stack finds the
        # most recent year; where a cell has never burnt, argmax returns 0 on an
        # all-zero column, so the never-burnt case is handled explicitly.
        for name, stack in (("burnt", past_burnt), ("late", past_late)):
            ever = np.nansum(stack, axis=0) > 0
            rev = np.nan_to_num(stack[::-1], nan=0.0) > 0
            since = rev.argmax(axis=0) + 1.0
            # Cells that never burnt get "longer ago than the record goes", plus a flag
            # so the models can tell that apart from a genuine long interval.
            f[f"yrs_since_{name}"] = np.where(ever, since, float(t) + 1.0)
            f[f"never_{name}"] = (~ever).astype("f4")

        # Last year's state, the single strongest fire-history signal: a place that
        # burnt last year has little fuel left to carry a fire this year.
        f["prev_burnt"] = burnt[t - 1]
        f["prev_early"] = early[t - 1]
        f["prev_late"] = late[t - 1]
        f["prev_frac_burnt"] = frac_burnt[t - 1]
        f["prev_frac_early"] = frac_early[t - 1]
        f["prev_frac_late"] = frac_late[t - 1]
        f["prev2_burnt"] = burnt[t - 2]

        # What the surrounding country did last year. Fire spreads across cells, and
        # burning programmes are run over whole districts, not single 5 km squares.
        f["nbr_frac_burnt_prev"] = neighbour_mean(frac_burnt[t - 1])
        f["nbr_frac_early_prev"] = neighbour_mean(frac_early[t - 1])
        f["nbr_frac_late_prev"] = neighbour_mean(frac_late[t - 1])

        rr, cc = np.where(inside)
        out = pd.DataFrame({"row": rr, "col": cc, "year": year})
        for k, v in f.items():
            out[k] = v[rr, cc] if isinstance(v, np.ndarray) else v
        rows.append(out)

    feat = pd.concat(rows, ignore_index=True)

    keep = ["cell_id", "row", "col", "year", "lon", "lat", "frac_burnt", "label"]
    feat = feat.merge(df[keep], on=["row", "col", "year"], how="inner")

    # A cell on the coastal or state edge has no complete neighbourhood; rather than
    # fabricate one, drop those cell-years and say so.
    before = len(feat)
    feat = feat.dropna(subset=["nbr_frac_burnt_prev"]).reset_index(drop=True)
    print(f"dropped {before - len(feat):,} edge cell-years with no neighbourhood")

    PROC.mkdir(parents=True, exist_ok=True)
    feat.to_parquet(OUT, index=False)

    print(f"\nwrote {OUT.relative_to(Path(__file__).resolve().parents[1])}")
    print(f"  {len(feat):,} rows, {feat['cell_id'].nunique():,} cells, "
          f"{feat['year'].min()}-{feat['year'].max()}")
    print(f"  {sum(c not in keep for c in feat.columns)} features")
    print("\nclass balance on the modelling years:")
    print((feat["label"].value_counts(normalize=True) * 100).round(1).to_string())


if __name__ == "__main__":
    main()
