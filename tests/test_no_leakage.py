"""Check that no fire-history feature can see the year it is predicting.

This is the test the project rests on. Every fire-history feature is derived from the
same table that holds the label, so a single off-by-one in a slice would let a model
read next year's fire from this year's features and score beautifully on nothing.

Two independent checks, because they fail in different ways:

1. Recompute a sample of features from scratch, filtering the raw table to years
   strictly before the target. Catches an off-by-one, a wrong window, a wrong sign.
2. Scramble a year's labels and rebuild. Any feature for that year that moves was
   reading the year it is supposed to be predicting. Catches leakage even if the
   recomputation repeats the same mistake.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pyrantis.schema import INTERIM, PROC

RNG = np.random.default_rng(0)


def load():
    return pd.read_parquet(INTERIM / "cell_year_labels.parquet"), pd.read_parquet(
        PROC / "cell_year_features.parquet"
    )


def test_features_recomputed_from_prior_years_only():
    labels, feat = load()
    hist = {
        (r.cell_id, r.year): (r.label, r.frac_burnt)
        for r in labels.itertuples()
    }

    sample = feat.sample(300, random_state=1)
    checked = 0

    for r in sample.itertuples():
        past = [
            hist[(r.cell_id, y)]
            for y in range(labels.year.min(), r.year)      # strictly before
            if (r.cell_id, y) in hist
        ]
        assert len(past) == r.hist_years, "history length disagrees"

        for w in (3, 5, 10):
            win = past[-w:]
            want = sum(lab != "unburnt" for lab, _ in win) / len(win)
            got = getattr(r, f"freq_{w}")
            assert abs(want - got) < 1e-5, f"freq_{w}: want {want}, got {got}"

            want_late = sum(lab == "late" for lab, _ in win) / len(win)
            assert abs(want_late - getattr(r, f"late_freq_{w}")) < 1e-5

        # years since last fire, counting back from the year before the target
        burnt_yrs = [i for i, (lab, _) in enumerate(past) if lab != "unburnt"]
        if burnt_yrs:
            want = len(past) - burnt_yrs[-1]
            assert r.never_burnt == 0
        else:
            want = len(past) + 1
            assert r.never_burnt == 1
        assert abs(want - r.yrs_since_burnt) < 1e-5, "yrs_since_burnt"

        prev_lab, prev_frac = past[-1]
        assert r.prev_burnt == float(prev_lab != "unburnt")
        assert r.prev_late == float(prev_lab == "late")
        assert abs(r.prev_frac_burnt - prev_frac) < 1e-5
        checked += 1

    assert checked == 300
    print(f"recomputed {checked} sampled cell-years from the raw table: all agree")


def test_scrambling_a_years_labels_does_not_move_its_features():
    """The adversarial check: if a feature for year Y changes when year Y's own
    labels are shuffled, that feature is reading the target year."""
    import importlib

    labels, feat = load()
    target = int(labels.year.max())

    baseline = feat[feat.year == target].sort_values("cell_id").reset_index(drop=True)

    # Shuffle the target year's labels and burnt fractions across cells, leaving every
    # other year untouched, then rebuild the features from the scrambled table.
    scrambled = labels.copy()
    mask = scrambled.year == target
    perm = RNG.permutation(mask.sum())
    for c in ("label", "frac_burnt", "frac_early", "frac_late"):
        scrambled.loc[mask, c] = scrambled.loc[mask, c].to_numpy()[perm]

    orig = INTERIM / "cell_year_labels.parquet"
    backup = INTERIM / "_labels_backup.parquet"
    out = PROC / "cell_year_features.parquet"
    out_backup = PROC / "_features_backup.parquet"
    orig.rename(backup)
    out.rename(out_backup)
    try:
        scrambled.to_parquet(orig, index=False)
        bf = importlib.import_module("scripts.build_features")
        bf.main()
        after = pd.read_parquet(out)
        after = after[after.year == target].sort_values("cell_id").reset_index(drop=True)
    finally:
        orig.unlink(missing_ok=True)
        out.unlink(missing_ok=True)
        backup.rename(orig)
        out_backup.rename(out)

    assert (baseline.cell_id.to_numpy() == after.cell_id.to_numpy()).all()

    feature_cols = [
        c for c in baseline.columns
        if c not in ("cell_id", "row", "col", "year", "lon", "lat", "frac_burnt", "label")
    ]
    moved = [
        c for c in feature_cols
        if not np.allclose(baseline[c].to_numpy(), after[c].to_numpy(), equal_nan=True)
    ]
    assert not moved, f"these features change when the target year changes: {moved}"

    # The check is only meaningful if the shuffle actually changed the target year.
    assert not (baseline.label.to_numpy() == after.label.to_numpy()).all()
    print(f"scrambled {target}: all {len(feature_cols)} features unchanged")


if __name__ == "__main__":
    test_features_recomputed_from_prior_years_only()
    test_scrambling_a_years_labels_does_not_move_its_features()
    print("\nno leakage detected")
