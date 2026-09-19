"""Assemble the final feature table: fire history, weather, and satellite greenness.

The experiment search settled on one combination out of twenty-three tried: the original
fire-history and weather columns, plus fire frequency over longer windows, plus elevation,
plus MODIS greenness. This builds that table once so every model trains on exactly the
same thing and the comparison in the report is fair.

Two blocks that looked promising are deliberately absent. District-scale neighbourhood
features made the model worse, and the wet-season shape features became redundant the
moment real greenness was available, which is what you would expect when a measurement
replaces the proxy that was standing in for it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pyrantis.schema import INTERIM, PROC
from scripts.experiments import block_longmem, block_elevation, block_ndvi

OUT = PROC / "cell_year_features_full.parquet"


def main() -> None:
    feat = pd.read_parquet(PROC / "cell_year_features_weather.parquet").reset_index(drop=True)
    labels = pd.read_parquet(INTERIM / "cell_year_labels.parquet")
    print(f"starting from {len(feat):,} rows, {feat.shape[1]} columns", flush=True)

    for name, fn in (("longmem", block_longmem), ("elevation", block_elevation),
                     ("ndvi", block_ndvi)):
        df = fn(feat, labels)
        feat = pd.concat([feat, df], axis=1)
        print(f"  + {name:<10} {len(df.columns):>2} columns", flush=True)

    missing = feat[[c for c in feat.columns if c.startswith("ndvi_")]].isna().mean()
    if missing.max() > 0:
        print(f"  greenness missing for {missing.max()*100:.2f}% of rows at worst")

    PROC.mkdir(parents=True, exist_ok=True)
    feat.to_parquet(OUT, index=False)
    print(f"\nwrote {OUT.relative_to(ROOT)}  {len(feat):,} rows, {feat.shape[1]} columns")


if __name__ == "__main__":
    main()
