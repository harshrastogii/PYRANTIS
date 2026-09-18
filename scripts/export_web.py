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
SITE = PROJ / "_site"
OUT = SITE / "index.html"
ASSETS = PROJ / "web" / "assets"
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
    recur = json.load(open(PROJ / "reports" / "recurrence.json"))
    places = json.load(open(PROJ / "reports" / "places.json"))

    # When the Territory burns, as a share of every burnt cell-year. The raster carried
    # this all along; only the early/late split was being kept.
    burnt = lab[lab["label"] != "unburnt"]
    mc = burnt["peak_month"][burnt["peak_month"] > 0].value_counts(normalize=True) * 100
    months = {str(m): round(float(mc.get(m, 0.0)), 1) for m in range(1, 13)}

    wf_path = PROJ / "reports" / "walk_forward.json"
    walk = json.load(open(wf_path)) if wf_path.exists() else None

    # The forecast year: a grid like any other, plus the model's confidence per cell so
    # the page can show where it is sure and where it is guessing.
    fpath = sorted((PROJ / "reports").glob("forecast_*.npz"))
    fc_grid, fc_meta, fc_conf = {}, {}, {}
    if fpath:
        fz = np.load(fpath[-1])
        fyear = int(fpath[-1].stem.split("_")[1])
        fdf = pd.DataFrame({"row": fz["row"], "col": fz["col"],
                            "ch": [str(v) for v in fz["pred"]], "year": fyear})
        fc_grid = grid_strings(fdf, "ch", r0, r1, c0, c1)
        conf = fz["prob"].max(axis=1)
        # Confidence banded into four steps: a continuous ramp would be unreadable at
        # one pixel per cell.
        band = np.clip((conf - 0.34) / (1.0 - 0.34) * 4, 0, 3.999).astype(int)
        cdf = pd.DataFrame({"row": fz["row"], "col": fz["col"],
                            "ch": [str(v) for v in band], "year": fyear})
        fc_conf = grid_strings(cdf, "ch", r0, r1, c0, c1)
        fc_meta = json.load(open(PROJ / "reports" / f"forecast_{fyear}.json"))

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
        "recurrence": recur,
        "places": places["places"],
        "months": months,
        "walk": walk,
        "forecast": {"grid": fc_grid, "confidence": fc_conf, **fc_meta},
    }

    blob = json.dumps(data, separators=(",", ":"))
    html = TEMPLATE.read_text()
    assert "__ATLAS_DATA__" in html, "template is missing the data placeholder"
    # MapLibre's stylesheet is inlined rather than linked: the page then has no external
    # CSS dependency and renders the same wherever it is hosted.
    css = (PROJ / "web" / "vendor" / "maplibre-gl.css").read_text()
    html = html.replace("__MAPLIBRE_CSS__", css)
    SITE.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html.replace("__ATLAS_DATA__", blob))

    # Assets travel with the build so the committed _site is self-contained and
    # Cloudflare needs no build step.
    import shutil
    if ASSETS.exists():
        shutil.copytree(ASSETS, SITE / "assets", dirs_exist_ok=True)

    print(f"wrote {OUT.relative_to(PROJ)}  {OUT.stat().st_size / 1e6:.2f} MB")
    print(f"  grid {data['grid']['rows']} x {data['grid']['cols']}, "
          f"{len(observed)} observed years, {len(predicted)} predicted years")
    print(f"  best model: {best['model']}  macro-F1 {best['f1_macro']:.3f}")
    print(f"  accuracy by held-out year: {per_year}")


if __name__ == "__main__":
    main()
