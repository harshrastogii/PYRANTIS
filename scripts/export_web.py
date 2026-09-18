"""Pack the results into the single data blob the atlas page reads.

The atlas draws real cells, not a decorative map: each year becomes one string with a
character per grid position, which the page paints straight onto a canvas. At 0.05 degrees
the Territory is a 301 by 181 grid, so a year is about 54,000 characters and the whole
record fits comfortably inside the page.

Predictions are included only for the held-out years. Showing the model's output over
years it trained on would flatter it, and the atlas would be advertising rather than
reporting.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pyrantis.schema import CELL_DEG, CLASSES, INTERIM, ROOT as PROJ

TEMPLATE = PROJ / "web" / "template.html"
OUT = PROJ / "web" / "index.html"
OUTSIDE = "."                      # grid positions beyond the Territory


def grid_strings(df: pd.DataFrame, col: str, r0: int, r1: int, c0: int, c1: int) -> dict:
    """One string a year, row-major, a character per grid position."""
    nr, nc = r1 - r0 + 1, c1 - c0 + 1
    out = {}
    for year, g in df.groupby("year", observed=True):
        buf = np.full((nr, nc), OUTSIDE, dtype="<U1")
        buf[g["row"].to_numpy() - r0, g["col"].to_numpy() - c0] = g[col].to_numpy()
        out[str(int(year))] = "".join(buf.ravel())
    return out


def main() -> None:
    lab = pd.read_parquet(INTERIM / "cell_year_labels.parquet")
    code = {c: str(i) for i, c in enumerate(CLASSES)}
    lab["ch"] = lab["label"].map(code)

    r0, r1 = int(lab["row"].min()), int(lab["row"].max())
    c0, c1 = int(lab["col"].min()), int(lab["col"].max())
    lat_top = float(lab.loc[lab["row"] == r0, "lat"].iloc[0])
    lon_left = float(lab.loc[lab["col"] == c0, "lon"].iloc[0])

    observed = grid_strings(lab, "ch", r0, r1, c0, c1)

    # Annual class shares, the figures the atlas quotes beside the map.
    annual = {}
    for year, g in lab.groupby("year", observed=True):
        v = g["label"].value_counts(normalize=True) * 100
        annual[str(int(year))] = {c: round(float(v.get(c, 0.0)), 1) for c in CLASSES}

    res = json.load(open(PROJ / "reports" / "results_weather.json"))
    res_fh = json.load(open(PROJ / "reports" / "results_firehistory.json"))

    best = max((r for r in res["results"] if "(balanced)" in r["model"]),
               key=lambda r: r["f1_macro"])

    npz = np.load(PROJ / "reports" / "test_predictions_weather.npz")
    key = best["model"].replace(" ", "_")
    pred = pd.DataFrame({
        "cell_id": npz["cell_id"], "year": npz["year"],
        "ch": np.asarray([str(v) for v in npz[key]]),
    })
    geo = lab[["cell_id", "row", "col"]].drop_duplicates("cell_id")
    pred = pred.merge(geo, on="cell_id", how="left")
    predicted = grid_strings(pred, "ch", r0, r1, c0, c1)

    # Per-year agreement on the held-out years, so the atlas can say how each test year
    # went rather than only quoting one number for all three.
    truth = lab.set_index(["cell_id", "year"])["ch"]
    pred["truth"] = truth.reindex(
        pd.MultiIndex.from_arrays([pred["cell_id"], pred["year"]])).to_numpy()
    per_year = {str(int(y)): round(float((g["ch"] == g["truth"]).mean() * 100), 1)
                for y, g in pred.groupby("year", observed=True)}

    imp = list(json.load(open(PROJ / "reports" / "feature_importance_weather.json")).items())

    data = {
        "grid": {"rows": r1 - r0 + 1, "cols": c1 - c0 + 1, "cell": CELL_DEG,
                 "latTop": lat_top, "lonLeft": lon_left, "outside": OUTSIDE},
        "classes": list(CLASSES),
        "years": sorted(int(y) for y in observed),
        "observed": observed,
        "predicted": predicted,
        "predModel": best["model"].replace(" (balanced)", ""),
        "annual": annual,
        "accuracyByYear": per_year,
        "split": res["split"],
        "models": res["results"],
        "modelsNoWeather": res_fh["results"],
        "importance": [[k, round(v, 4)] for k, v in imp[:12]],
        "nCells": int(lab["cell_id"].nunique()),
        "nRows": int(len(lab)),
    }

    blob = json.dumps(data, separators=(",", ":"))
    html = TEMPLATE.read_text()
    assert "__ATLAS_DATA__" in html, "template is missing the data placeholder"
    OUT.write_text(html.replace("__ATLAS_DATA__", blob))

    print(f"wrote {OUT.relative_to(PROJ)}  {OUT.stat().st_size / 1e6:.2f} MB")
    print(f"  grid {data['grid']['rows']} x {data['grid']['cols']}, "
          f"{len(observed)} observed years, {len(predicted)} predicted years")
    print(f"  best model: {best['model']}  macro-F1 {best['f1_macro']:.3f}")
    print(f"  accuracy by held-out year: {per_year}")


if __name__ == "__main__":
    main()
