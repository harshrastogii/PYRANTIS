"""Attach real Northern Territory places to the grid, so a person can find themselves.

A map of 46,000 anonymous squares asks the reader to know where they are on an outline.
Naming the towns turns the same data into something you can look up: pick Katherine, read
what the record says about Katherine.

Each town is matched to the cell containing it, and the cell's record is turned into
sentences rather than columns: how often it has burned, when in the year it usually burns,
how long since the last one, and what the model expects next.
"""

from __future__ import annotations

import calendar
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pyrantis.schema import CELL_DEG, CLASSES, INTERIM, ROOT as PROJ

# Towns, communities and landmarks across the Territory, spread so that every region has
# something recognisable in it. Coordinates are the settlement centres.
# The towns a reader is most likely to look for. They keep their label when the map gets
# crowded; the smaller communities give way first.
MAJOR = {"Darwin", "Palmerston", "Katherine", "Tennant Creek", "Alice Springs",
         "Nhulunbuy", "Jabiru", "Yulara", "Borroloola", "Wadeye", "Maningrida"}

PLACES = [
    ("Darwin", -12.4634, 130.8456), ("Palmerston", -12.4861, 130.9833),
    ("Humpty Doo", -12.5833, 131.1333), ("Batchelor", -13.0500, 131.0333),
    ("Adelaide River", -13.2394, 131.1058), ("Pine Creek", -13.8236, 131.8342),
    ("Katherine", -14.4650, 132.2635), ("Mataranka", -14.9250, 133.0667),
    ("Larrimah", -15.5744, 133.2136), ("Daly Waters", -16.2539, 133.3722),
    ("Elliott", -17.5500, 133.5333), ("Tennant Creek", -19.6483, 134.1911),
    ("Barrow Creek", -21.5333, 133.8833), ("Ti Tree", -22.1333, 133.4167),
    ("Alice Springs", -23.6980, 133.8807), ("Hermannsburg", -23.9436, 132.7856),
    ("Santa Teresa", -24.1333, 134.3667), ("Yulara", -25.2406, 130.9889),
    ("Kings Canyon", -24.2567, 131.5772), ("Kintore", -23.2667, 129.3833),
    ("Kaltukatjara", -24.8667, 129.0833), ("Yuendumu", -22.2556, 131.8017),
    ("Papunya", -23.2167, 131.9000), ("Lajamanu", -18.3333, 130.6333),
    ("Kalkarindji", -17.4333, 130.8167), ("Timber Creek", -15.6567, 130.4783),
    ("Wadeye", -14.2333, 129.5167), ("Jabiru", -12.6714, 132.8369),
    ("Gunbalanya", -12.3250, 133.0500), ("Maningrida", -12.0500, 134.2333),
    ("Ramingining", -12.3667, 134.9000), ("Milingimbi", -12.1000, 134.9167),
    ("Galiwinku", -12.0250, 135.5667), ("Nhulunbuy", -12.1833, 136.7833),
    ("Yirrkala", -12.2500, 136.8833), ("Numbulwar", -14.2667, 135.7333),
    ("Ngukurr", -14.7333, 134.7333), ("Alyangula", -13.8500, 136.4167),
    ("Borroloola", -16.0714, 136.3064), ("Wurrumiyanga", -11.7614, 130.6231),
    ("Pirlangimpi", -11.4000, 130.4167),
]


def _mode(g: pd.DataFrame, label: str) -> int:
    m = g.loc[(g["label"] == label) & (g["peak_month"] > 0), "peak_month"]
    return int(m.mode().iloc[0]) if len(m) else 0


def sentence(rec: dict) -> str:
    """One plain sentence describing how this place burns."""
    n, yrs = rec["times_burnt"], rec["years_on_record"]
    if n == 0:
        return "No fire has been mapped here since 2000."
    if n >= yrs * 0.75:
        how = "burns almost every year"
    elif n >= yrs * 0.45:
        how = "burns in most years"
    elif n >= yrs * 0.2:
        how = "burns every few years"
    else:
        how = "burns rarely"
    month = calendar.month_name[rec["usual_month"]] if rec["usual_month"] else None
    tail = f", usually around {month}" if month else ""
    return f"This area {how}{tail}. It has burned in {n} of the last {yrs} years."


def main() -> None:
    lab = pd.read_parquet(INTERIM / "cell_year_labels.parquet")
    years = np.sort(lab["year"].unique())
    latest = int(years[-1])

    fpath = sorted((PROJ / "reports").glob("forecast_*.npz"))
    fc = None
    if fpath:
        fz = np.load(fpath[-1])
        fyear = int(fpath[-1].stem.split("_")[1])
        fc = pd.DataFrame({"cell_id": fz["cell_id"], "pred": fz["pred"],
                           "conf": fz["prob"].max(axis=1)}).set_index("cell_id")

    geo = lab[["cell_id", "row", "col", "lon", "lat"]].drop_duplicates("cell_id")
    by_cell = {c: g for c, g in lab.groupby("cell_id", observed=True)}

    out = []
    for name, lat, lon in PLACES:
        d = (geo["lon"] - lon) ** 2 + (geo["lat"] - lat) ** 2
        near = geo.loc[d.idxmin()]
        if float(np.sqrt(d.min())) > CELL_DEG * 1.5:
            print(f"  skipped {name}: no cell within range")
            continue

        g = by_cell[near["cell_id"]].sort_values("year")
        burnt = g[g["label"] != "unburnt"]
        months = burnt["peak_month"][burnt["peak_month"] > 0]

        rec = {
            "name": name, "lat": round(lat, 4), "lon": round(lon, 4),
            "cell_id": int(near["cell_id"]), "row": int(near["row"]), "col": int(near["col"]),
            "major": name in MAJOR,
            "years_on_record": int(len(g)),
            "times_burnt": int(len(burnt)),
            "times_late": int((g["label"] == "late").sum()),
            "usual_month": int(months.mode().iloc[0]) if len(months) else 0,
            # Split by season, so a forecast of "early" can name the month this place
            # usually burns in the early window rather than its overall busiest month.
            "usual_month_early": _mode(g, "early"),
            "usual_month_late": _mode(g, "late"),
            "last_burnt": int(burnt["year"].max()) if len(burnt) else None,
            "history": "".join(str(CLASSES.index(v)) for v in g["label"]),
        }
        if fc is not None and near["cell_id"] in fc.index:
            r = fc.loc[near["cell_id"]]
            rec["forecast_year"] = fyear
            rec["forecast"] = CLASSES[int(r["pred"])]
            rec["confidence"] = round(float(r["conf"]), 3)
        rec["summary"] = sentence(rec)
        out.append(rec)

    out.sort(key=lambda r: r["name"])
    json.dump({"latest_year": latest, "places": out},
              open(PROJ / "reports" / "places.json", "w"), indent=1)

    print(f"{len(out)} places matched to cells\n")
    print(f"{'place':<16}{'burnt':>7}{'usual':>11}{'last':>7}   forecast")
    for r in out[:14]:
        m = calendar.month_abbr[r["usual_month"]] if r["usual_month"] else "-"
        print(f"{r['name']:<16}{r['times_burnt']:>3}/{r['years_on_record']:<3}{m:>11}"
              f"{str(r['last_burnt'] or '-'):>7}   {r.get('forecast','-')}"
              f" ({r.get('confidence','-')})")


if __name__ == "__main__":
    main()
