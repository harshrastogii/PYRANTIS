# PYRANTIS

Classifying Northern Territory fire regimes from satellite burnt-area mapping, gridded
weather and satellite greenness, and forecasting the 2026 season.

Built for **PRT565 Machine Learning, Artificial Intelligence and Algorithms**,
Assessment 3, Charles Darwin University (Darwin, Danala campus). Group 85: Harsh
Rastogi (386401), Saira Zafar (407193), Tharushi Wimalachandra (387594).

- Live site: https://pyrantis.harshlabs.workers.dev
- Recorded presentation: https://youtu.be/c_0yCsdZX2U

## The question

Every year, most of the Northern Territory's savanna burns. *When* it burns is what
matters: fires before the end of July are cooler, patchier and are the basis of
savanna fire management across northern Australia, while fires after July are hotter,
larger and far more damaging.

PYRANTIS classifies each 0.05° cell of the Territory (about 5 km across), for each year
from 2000 to 2025, into one of three fire regimes. The labels cover 46,445 cells and
1,207,570 cell-years:

| Class | Meaning | Share of cell-years |
|---|---|---|
| `unburnt` | no fire detected | 67.5% |
| `early` | burnt on or before 31 July | 14.0% |
| `late` | burnt after 31 July | 18.5% |

## Data

**NAFI** (North Australia and Rangelands Fire Information, Charles Darwin University).
250 m burnt-area mapping for 2000 to 2025. Each pixel carries the month it was detected
as burnt, which is what makes the early/late split possible. Used with permission.

**SILO Data Drill** (Queensland Government). Daily rainfall, maximum temperature, vapour
pressure deficit and solar radiation, interpolated from Bureau of Meteorology station
observations onto the same 0.05° grid. CC BY 4.0.

**MODIS MOD13Q1** (NASA), through Google Earth Engine. Sixteen-day vegetation greenness
at 250 m, averaged onto the same grid. Greenness over the wet season measures how much
grass is standing when the dry season starts.

The 31 July cut-off is not our invention: it is the northern-Australian savanna burning
convention, and NAFI publishes its own "fire frequency after July 31" layer on the same
basis.

## Why burnt area, not hotspots

Satellite hotspots are thermal detections at the moment of overpass. They record a heat
signature, not burnt ground: a fire between overpasses is missed, and a hot surface can
trigger a false detection. NAFI's product is mapped burnt area, derived by comparing
successive images and validated by aerial transects across northern Australia. Since the
label is "did this area burn, and when", burnt area is the correct source, and hotspots
would add label noise we could not measure.

## Features

The models use 50 features per cell-year, all built from information available by
30 April of the year being predicted:

| Group | Count | What it holds |
|---|---|---|
| Fire history | 26 | years since the last fire and last late fire, fire frequency over 3 to 25 years, last year's state, and the eight neighbouring cells |
| Weather | 14 | wet-season rainfall, heat, vapour pressure deficit and radiation, the dry season before it, and each cell's departure from its own normal |
| Greenness | 7 | mean, peak and lowest wet-season greenness, April greenness, green-up speed and two anomalies |
| Place | 3 | longitude, latitude and elevation |

Two kinds of input are left out on purpose: anything measured in the target year, which
would hand the model the answer, and each cell's long-term average greenness, which acts
as an identifier for the cell rather than evidence about the year.

## Results

Trained on 2005 to 2019, tuned on 2020 to 2022, and scored once on 2023 to 2025. Every
model is class-weighted. Precision, recall and F1 are macro-averaged, so each class
counts equally:

| Model | Accuracy | Precision | Recall | F1 | F1 on late fire | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|
| Gradient boosting | 0.725 | 0.679 | 0.690 | **0.684** | 0.553 | 0.862 | 0.732 |
| LSTM | 0.692 | 0.679 | 0.687 | 0.673 | 0.561 | 0.854 | 0.724 |
| Multilayer ANN | 0.678 | 0.670 | 0.695 | 0.667 | 0.556 | 0.856 | 0.713 |
| Random forest | 0.727 | 0.688 | 0.651 | 0.660 | 0.468 | 0.859 | 0.730 |
| Decision tree | 0.628 | 0.598 | 0.632 | 0.604 | 0.495 | 0.793 | 0.633 |
| Naive Bayes | 0.644 | 0.592 | 0.616 | 0.602 | 0.398 | 0.797 | 0.637 |
| *baseline:* same as last year | 0.567 | 0.511 | 0.504 | 0.506 | 0.241 | | |
| *baseline:* always unburnt | 0.578 | 0.193 | 0.333 | 0.244 | 0.000 | | |

ROC-AUC and PR-AUC come from `scripts/robustness.py`, a separate run of the same models.
Neural network scores move by a few thousandths between runs, and the LSTM scored 0.670
in that run.

Late fire is the hard class and the one that matters, since those are the fires
management exists to prevent. Class weighting stops the models buying accuracy by never
calling it: gradient boosting scores 0.420 on late fire unweighted and 0.553 weighted.

Each source of data earned its place. Fire history alone reaches 0.639 macro F1, adding
weather reaches 0.674, and adding greenness, longer fire memory and elevation reaches
0.684. Twenty-three experiments against a fixed benchmark decided which extra features
to keep; one was kept. A U-Net that read the grid as an image scored 0.588 and was
dropped.

The strongest predictors are not fire history. The random forest ranks wet-season vapour
pressure deficit, wet-season greenness and wet-season rainfall highest, and permutation
importance on the held-out years agrees on 7 of the top 10. A wet growing season builds
the grass, and how thirsty the air has been decides how early it cures enough to carry a
fire.

## Checks on the results

- **Walk-forward testing.** The record is cut at every year from 2014 to 2024 and the
  model is scored on every later year: 66 tests. It averages 0.640 macro F1 (standard
  deviation 0.033) and beats the same-as-last-year baseline in all 66. Skill barely
  fades with distance: 0.635 one year ahead, 0.604 eleven years ahead.
- **Tuning.** Every model was tuned with a randomised or grid search scored on the
  validation years only. Gradient boosting stayed at 0.684; the decision tree gained
  most, from 0.604 to 0.632.
- **Overfitting.** Gradient boosting scores 0.821 on its training years and 0.684 on the
  test years. The random forest memorises far more, 0.924 against 0.660. The neural
  networks stop training at their best validation epoch.
- **Why not random cross-validation.** Repeated stratified 5-fold cross-validation on the
  training years scores 0.772, against 0.677 for the same model on later years. A random
  split puts a cell's neighbours and its own nearby years into training, which is why
  every score here comes from years the model had not seen.

## Forecast for 2026

Trained on every labelled year through 2025, the model expects 51% of the Territory to
burn in 2026 (14.5% before August, 36.4% after July), against a 2000 to 2025 average of
33%, with a mean confidence of 0.733. The wet season behind it was the second wettest in
22 years. No burnt-area record exists yet for 2026, so the forecast can only be checked
against NAFI's mapping in 2027.

## Two rules the code is built around

**Nothing may see the year it is predicting.** Fire-history features come from the same
table as the label, so an off-by-one would let a model read the answer. `tests/` checks
this two ways: by recomputing sampled features from the raw table, and by scrambling a
year's labels and confirming not one feature for that year moves.

**Everything stops at 30 April.** The fire season runs May to October. Using August
weather to decide whether a cell burnt late would describe the fire rather than predict
it, and the model could never be run in advance. Everything it uses is known by the end
of the wet season, so it can be run in May.

The test years are split by time, never at random. Cells are correlated with themselves
across years and with their neighbours across space, so a random split reports an
accuracy that would not survive a real season.

## Known limitations

- NAFI assigns each burnt pixel the month covering the largest part of the interval in
  which it was detected. Near the July/August boundary that attribution is approximate.
- Late fire is still the weakest class, at 0.553 F1.
- The settings and experiments were all judged on the same three validation years, and
  every score is measured against NAFI's own mapping rather than an independent record.
- Everything is fitted to the Northern Territory at 0.05°; transfer to other regions is
  untested.

## Layout

```
pyrantis/     shared constants
scripts/      download, preprocessing, models, checks, forecast and site export
tests/        leakage tests
data/         raw, interim and processed (not committed; scripts rebuild it)
reports/      results as JSON, figures, and the submitted report and presentation
web/          the page template and its assets
_site/        the built page Cloudflare serves (committed)
env/          pinned Python requirements
```

The submitted files are `reports/PRT565 A3 Report Group85.pdf` and
`reports/PYRANTIS Presentation.pptx`, with the presentation also as a PDF and the
speaking script in `reports/PYRANTIS_presentation_script.pdf`.

## Reproducing

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r env/requirements.txt

.venv/bin/python scripts/download_nafi.py            # 26 GeoTIFFs, about 40 MB
.venv/bin/python scripts/build_grid.py               # cell-by-year labels
.venv/bin/python scripts/build_features.py           # fire-history features

# SILO asks for a contact address with each request
SILO_CONTACT=you@example.com .venv/bin/python scripts/download_weather.py
.venv/bin/python scripts/build_weather_features.py
.venv/bin/python scripts/build_weather_monthly.py

# needs an Earth Engine account; set PROJECT in the script to your own project
.venv/bin/python scripts/fetch_ndvi.py
.venv/bin/python scripts/build_full_features.py

.venv/bin/python scripts/train_models.py             # fire history only
.venv/bin/python scripts/train_models.py --weather   # plus weather
.venv/bin/python scripts/train_models.py --full      # the final 50 features
.venv/bin/python scripts/walk_forward.py
.venv/bin/python scripts/experiments.py
.venv/bin/python scripts/robustness.py               # tuning, overfitting, AUC, importance
.venv/bin/python scripts/forecast.py

.venv/bin/python scripts/build_places.py
.venv/bin/python scripts/analyse_recurrence.py
.venv/bin/python scripts/export_web.py               # writes _site/
.venv/bin/python -m pytest tests/
```

The raw data is not committed. NAFI is about 40 MB and SILO about 800 MB, and both
download scripts are resumable, so a rebuild picks up wherever it stopped.

## Deploying

The site is served from `_site/`, which is committed, so Cloudflare Workers Builds needs
no build step: `wrangler.toml` declares the assets directory and the deploy command is
`npx wrangler deploy`. If a push does not start a build, use **Retry build** on the latest
build in the Cloudflare dashboard.

Python requirements live under `env/` rather than at the repository root. Cloudflare
detects a root `requirements.txt` and pip-installs the whole analysis stack, TensorFlow
included, before every deploy of a static page that needs none of it.
