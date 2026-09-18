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
.venv/bin/python scripts/build_features.py     # fire-history features

# SILO asks for a contact address with each request
SILO_CONTACT=you@example.com .venv/bin/python scripts/download_weather.py
.venv/bin/python scripts/build_weather_features.py
.venv/bin/python scripts/train_models.py --weather
.venv/bin/python scripts/make_figures.py
```

The raw data is not committed. NAFI is about 40 MB and SILO about 800 MB, and both
scripts are resumable, so a rebuild picks up wherever it stopped.

## Results

Classified on held-out years 2023-2025, having trained on 2005-2019. Macro-averaged F1,
every model class-weighted:

| Model | Accuracy | Macro-F1 | F1 on late-season fire |
|---|---|---|---|
| LSTM | 0.712 | **0.674** | 0.534 |
| Multilayer ANN | 0.676 | 0.651 | 0.523 |
| Decision tree | 0.644 | 0.600 | 0.439 |
| Random forest | 0.700 | 0.598 | 0.304 |
| Naive Bayes | 0.623 | 0.586 | 0.382 |
| *baseline:* same as last year | 0.567 | 0.506 | 0.241 |
| *baseline:* majority class | 0.578 | 0.244 | 0.000 |

Late-season fire is the hard class and the one that matters, since those are the fires
management exists to prevent. Two things move it. Weighting the classes stops the models
buying accuracy by never calling it: unweighted, the LSTM scores 0.072 on late fire.
Weather then lifts it further, and only then does the sequence model earn its place --
on fire history alone a single decision tree matched the LSTM exactly.

The strongest predictors are not fire history at all. The forest ranks wet-season vapour
pressure deficit and wet-season rainfall above every fire-history feature, which is the
mechanism the problem implies: a wet growing season builds the grass, and how thirsty the
air has been decides how early it cures enough to carry a fire.

## Two rules the code is built around

**Nothing may see the year it is predicting.** Fire-history features come from the same
table as the label, so an off-by-one would let a model read the answer. `tests/` checks
this two ways: by recomputing sampled features from the raw table, and by scrambling a
year's labels and confirming not one feature for that year moves.

**Weather stops at 30 April.** The fire season runs May to October. Using August weather
to decide whether a cell burnt late would describe the fire rather than predict it, and
the model could never be run in advance. Everything it uses is known by the end of the
wet season, so it could genuinely be run in May.

The test years are split by time, never at random. Cells are correlated with themselves
across years and with their neighbours across space, so a random split would report an
accuracy that would not survive contact with a real season.
