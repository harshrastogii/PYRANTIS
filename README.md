# PYRANTIS

Classifying Northern Territory fire regimes from satellite burnt-area mapping and
ground weather observations.

Built for **PRT565 Machine Learning, Artificial Intelligence and Algorithms**,
Assessment 3, Charles Darwin University. Group 85 — Harsh Rastogi (386401),
Saira Zafar, Tharushi Wimalachandra.

## The question

Every year, most of the Northern Territory's savanna burns. *When* it burns is what
matters: fires before the end of July are cooler, patchier and are the basis of
savanna fire management across northern Australia, while fires after July are hotter,
larger and far more damaging.

PYRANTIS classifies each 0.05° cell of the Territory, for each year from 2000 to 2025,
into one of three fire regimes:

| Class | Meaning | Share of cell-years |
|---|---|---|
| `unburnt` | no fire detected | 67.5% |
| `early` | burnt on or before 31 July | 14.0% |
| `late` | burnt after 31 July | 18.5% |

## Data

**NAFI** — North Australia and Rangelands Fire Information, Charles Darwin University.
250 m burnt-area mapping, 2000–2025. Each pixel carries the month it was detected as
burnt, which is what makes the early/late split possible. Used with permission.

**Bureau of Meteorology** — daily ground observations at NT stations, used as fire
weather predictors. Used under a research licence.

The 31 July cut-off is not our invention: it is the northern-Australian savanna
burning convention, and NAFI publishes its own "fire frequency after July 31" layer on
the same basis.

## Why burnt area, not hotspots

Satellite hotspots are thermal detections at the moment of overpass. They record a
heat signature, not burnt ground: a fire between overpasses is missed, and a hot
surface can trigger a false detection. NAFI's product is mapped burnt area, derived by
comparing successive images and validated by aerial transects across northern
Australia. Since the label is "did this area burn, and when", burnt area is the
correct source and hotspots would add label noise we could not quantify.

Bureau of Meteorology stations record ground observations — temperature, humidity,
wind, rainfall. They do not record heat signatures, and are used here only as weather
predictors.

## Known limitation

NAFI assigns each burnt pixel the month covering the largest part of the interval in
which it was detected. Near the July/August boundary that attribution is approximate,
which is a real limitation of the early/late split and is stated rather than hidden.

## Layout

```
pyrantis/     shared constants and feature code
scripts/      download and preprocessing, kept out of the notebook
notebooks/    the analysis
data/         raw, interim and processed (not committed; scripts rebuild it)
reports/      figures
tests/        pipeline tests
```

## Reproducing

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python scripts/download_nafi.py      # 26 GeoTIFFs, about 40 MB
.venv/bin/python scripts/build_grid.py         # cell-by-year labels
```
