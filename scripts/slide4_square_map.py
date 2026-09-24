"""Render the 'what one square means' card for slide 4.

The card is sized for the space the slide leaves free: right of the title, above
the presenter's webcam bubble. It shows the whole Territory as the model sees it
in 2025, one colour per 5 km square, and blows up a small window near a named
town so the individual squares are visible. The data are the real 2025 labels,
not an illustration.
"""
import json, math, pathlib, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle, ConnectionPatch

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "deck" / "slide4_what_a_square_is.png"
YEAR, WIN = 2025, 10          # a 10 x 10 window is 50 km on a side

DF = pathlib.Path("/Applications/Microsoft PowerPoint.app/Contents/Resources/DFonts")
for f in ("Franklin Gothic Book.ttf", "Franklin Gothic Medium.ttf"):
    font_manager.fontManager.addfont(str(DF / f))
BOOK, MED = "Franklin Gothic Book", "Franklin Gothic Medium"

# the redesigned deck's palette
TINT, INK, BODY, MUTED = "#F4F1EB", "#16130F", "#3A342E", "#787066"
RED = "#C8341B"
CLS = {"unburnt": "#8E9AA1", "early": "#1E7A50", "late": "#C8341B"}
CODE = {"unburnt": 0, "early": 1, "late": 2}

df = pd.read_parquet(ROOT / "data/interim/cell_year_labels.parquet")
d = df[df.year == YEAR]
nr, nc = int(df.row.max()) + 1, int(df.col.max()) + 1
grid = np.full((nr, nc), -1, int)
grid[d.row.values, d.col.values] = d.label.map(CODE).values

# choose the town whose surrounding window shows the most even mix of all three answers
places = json.load(open(ROOT / "reports/places.json"))["places"]
best = None
for p in places:
    r0, c0 = p["row"] - WIN // 2, p["col"] - WIN // 2
    if r0 < 0 or c0 < 0 or r0 + WIN > nr or c0 + WIN > nc:
        continue
    w = grid[r0:r0 + WIN, c0:c0 + WIN]
    if (w < 0).any():
        continue
    share = np.bincount(w.ravel(), minlength=3) / w.size
    if share.min() < 0.12:
        continue
    ent = -(share * np.log(share)).sum()
    if best is None or ent > best[0]:
        best = (ent, p, r0, c0, share)
if best is None:
    sys.exit("no window with all three classes")
_, town, r0, c0, share = best
print(f"zoom near {town['name']}: rows {r0}-{r0+WIN}, cols {c0}-{c0+WIN}, "
      f"shares unburnt/early/late = {share.round(2)}")

rgb = np.ones((nr, nc, 4))
rgb[..., 3] = 0
for k, v in CODE.items():
    col = matplotlib.colors.to_rgba(CLS[k])
    rgb[grid == v] = col

# ---- figure: 4.8 x 4.1 inches, the slot on the slide
W_IN, H_IN = 4.8, 4.1
fig = plt.figure(figsize=(W_IN, H_IN), dpi=300)
fig.patch.set_facecolor(TINT)

# whole Territory, left
axL = fig.add_axes([0.055, 0.25, 0.36, 0.60])
axL.imshow(rgb, interpolation="nearest", aspect="equal")
axL.set_axis_off()
axL.add_patch(Rectangle((c0 - 0.5, r0 - 0.5), WIN, WIN, fill=False, ec=INK, lw=1.1))

# the zoomed window, right: each square drawn on its own with a hairline gap
axR = fig.add_axes([0.50, 0.25, 0.45, 0.60])
axR.set_xlim(0, WIN); axR.set_ylim(WIN, 0); axR.set_aspect("equal"); axR.set_axis_off()
gap = 0.07
for i in range(WIN):
    for j in range(WIN):
        v = grid[r0 + i, c0 + j]
        colr = [c for k, c in CLS.items() if CODE[k] == v][0]
        axR.add_patch(Rectangle((j + gap / 2, i + gap / 2), 1 - gap, 1 - gap,
                                fc=colr, ec="none"))
# mark the town's own square
ti, tj = town["row"] - r0, town["col"] - c0
axR.add_patch(Rectangle((tj + gap / 2, ti + gap / 2), 1 - gap, 1 - gap,
                        fill=False, ec="white", lw=2.0))
axR.add_patch(Rectangle((tj - 0.02, ti - 0.02), 1.04, 1.04, fill=False, ec=INK, lw=0.9))

# connector lines from the box on the map to the zoom panel
for (ya, yb) in ((r0 - 0.5, 0), (r0 + WIN - 0.5, WIN)):
    fig.add_artist(ConnectionPatch(xyA=(c0 + WIN - 0.5, ya), coordsA=axL.transData,
                                   xyB=(0, yb), coordsB=axR.transData,
                                   color=INK, lw=0.6, alpha=0.55))

# labels
fig.text(0.055, 0.925, "■", color=RED, fontsize=7.5, va="center", family=BOOK)
fig.text(0.085, 0.925, f"ONE SQUARE, ONE ANSWER   ·   {YEAR}", color=INK,
         fontsize=8.2, va="center", family=MED)
fig.text(0.055, 0.215, "The whole Territory", color=MUTED, fontsize=7.6, family=BOOK, va="top")
fig.text(0.50, 0.215, f"Around {town['name']}, 50 km across", color=MUTED, fontsize=7.6,
         family=BOOK, va="top")
fig.text(0.50, 0.165, f"Outlined: the square {town['name']} sits in", color=MUTED,
         fontsize=7.6, family=BOOK, va="top")
fig.text(0.055, 0.05, "Each square is about 5 km on a side. There are 46,445 of them.",
         color=BODY, fontsize=7.8, family=BOOK, va="bottom")

fig.savefig(OUT, dpi=300, facecolor=TINT)
print("wrote", OUT.relative_to(ROOT), f"{W_IN} x {H_IN} in")
