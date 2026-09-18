"""Constants shared by every stage of the PYRANTIS pipeline.

Anything that changes the meaning of a row lives here, so a reader can check the
project's assumptions in one place rather than inferring them from code.
"""
from __future__ import annotations

# ---------------------------------------------------------------- study area
STATE = "Northern Territory"
# NAFI rasters are GDA94 geographic (EPSG:4283); the state boundary is WGS84
# (EPSG:4326). The two differ by well under a NAFI pixel, so no datum shift is applied.
RASTER_CRS = "EPSG:4283"
BOUNDARY_CRS = "EPSG:4326"

# ------------------------------------------------------------------- raster
NAFI_PIXEL_DEG = 0.0025          # 250 m at the equator, as published by NAFI
UNBURNT_VALUE = 0                # NAFI encodes month 1-12; 0 means no fire detected

# --------------------------------------------------------------------- grid
# Analysis cells are square in degrees, not in metres: 20 x 20 NAFI pixels.
# Across the NT's latitude range that is about 5.1 km east-west and 5.55 km
# north-south, so cell area varies by roughly a tenth between the Top End and the
# southern border. Every feature we derive is a per-cell rate or fraction, so the
# variation does not bias the label, but it is stated here rather than hidden.
BLOCK = 20
CELL_DEG = NAFI_PIXEL_DEG * BLOCK       # 0.05 degrees

# ------------------------------------------------------------------- labels
# The early/late split follows the northern-Australian savanna burning convention and
# NAFI's own published product, which reports "fire frequency after July 31".
LATE_SEASON_FIRST_MONTH = 8             # August onwards is late dry season
CLASSES = ("unburnt", "early", "late")
CLASS_TO_INT = {c: i for i, c in enumerate(CLASSES)}

# A cell is only labelled when enough of it lies inside the NT and inside NAFI's
# mapped extent. Coastal and border cells that are mostly sea or outside coverage
# would otherwise be labelled "unburnt" on the strength of a handful of pixels.
MIN_VALID_FRACTION = 0.50

# A cell counts as burnt when a real share of it burnt, not a stray pixel. NAFI's
# mapping is 250 m, so one or two pixels in a 400-pixel cell is within noise.
MIN_BURNT_FRACTION = 0.05

FIRST_YEAR, LAST_YEAR = 2000, 2025
