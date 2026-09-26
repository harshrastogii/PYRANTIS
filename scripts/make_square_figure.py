"""Draw the 'what one square is' graphic for slide 4.

The middle panel is a real 20 x 20 NAFI block, not a drawing: cell row 69,
column 65 in 2019, just west of Katherine, which the pipeline labels late
because 41.8% of it burned after July against 17.8% before August. Reading the
actual pixels keeps the picture honest, and it is the same block the parquet
was built from -- the fractions printed by build_grid agree to four decimals.
"""

from __future__ import annotations

import pathlib
import sys
import zipfile

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window, from_bounds
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle, FancyArrow

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from pyrantis.schema import BLOCK, RAW                       # noqa: E402
from build_grid import nt_geometry                           # noqa: E402

OUT = ROOT / "reports" / "slide04_what_is_a_square.png"
YEAR, R, C = 2019, 69, 65

INK   = "#16130F"
BODY  = "#3A342E"
MUTED = "#787066"
DIM   = "#A79D91"
RED   = "#C8341B"
GREEN = "#1E7A50"
SLATE = "#8E9AA1"
LAND  = "#E7E1D7"
RULE  = "#DCD5CA"

for p in ["/System/Library/Fonts/Supplemental/Georgia.ttf",
          "/System/Library/Fonts/Supplemental/Georgia Bold.ttf",
          "/Applications/Microsoft PowerPoint.app/Contents/Resources/DFonts/Franklin Gothic Book.ttf",
          "/Applications/Microsoft PowerPoint.app/Contents/Resources/DFonts/Franklin Gothic Medium.ttf"]:
    if pathlib.Path(p).exists():
        font_manager.fontManager.addfont(p)
SERIF = "Georgia" if any(f.name == "Georgia" for f in font_manager.fontManager.ttflist) else "DejaVu Serif"
SANS = ("Franklin Gothic Book"
        if any(f.name == "Franklin Gothic Book" for f in font_manager.fontManager.ttflist)
        else "DejaVu Sans")


def read_block():
    z = RAW / "nafi" / f"firescars_{YEAR}_tif.zip"
    with zipfile.ZipFile(z) as zf:
        tif = next(n for n in zf.namelist() if n.lower().endswith(".tif"))
    with rasterio.open(f"zip://{z}!/{tif}") as ds:
        minx, miny, maxx, maxy = nt_geometry().bounds
        win = from_bounds(minx, miny, maxx, maxy, ds.transform)
        co = int(np.floor(win.col_off // BLOCK * BLOCK))
        ro = int(np.floor(win.row_off // BLOCK * BLOCK))
        w = int(np.ceil(win.width / BLOCK) * BLOCK)
        h = int(np.ceil(win.height / BLOCK) * BLOCK)
        w = min(w, ds.width - co); h = min(h, ds.height - ro)
        w -= w % BLOCK; h -= h % BLOCK
        arr = ds.read(1, window=Window(co, ro, w, h))
    return arr[R * BLOCK:(R + 1) * BLOCK, C * BLOCK:(C + 1) * BLOCK]


def nt_mask():
    d = pd.read_parquet(ROOT / "data/interim/cell_year_labels.parquet",
                        columns=["row", "col", "year"])
    d = d[d.year == YEAR]
    g = np.zeros((d.row.max() + 1, d.col.max() + 1), bool)
    g[d.row.to_numpy(), d.col.to_numpy()] = True
    return g


def main():
    blk = read_block()
    inside = nt_mask()
    early = ((blk > 0) & (blk <= 7)).mean()
    late = ((blk >= 8)).mean()

    fig = plt.figure(figsize=(5.4, 2.45), dpi=300)
    fig.patch.set_alpha(0)

    # ---------------------------------------------------------- panel 1: NT
    ax = fig.add_axes([0.005, 0.20, 0.21, 0.66])
    ax.imshow(np.where(inside, 1, np.nan), cmap=matplotlib.colors.ListedColormap([LAND]),
              interpolation="nearest")
    ax.add_patch(Rectangle((C - 9, R - 9), 19, 19, fill=False, ec=RED, lw=1.5))
    ax.set_xlim(-3, inside.shape[1] + 3); ax.set_ylim(inside.shape[0] + 3, -3)
    ax.axis("off")
    ax.set_title("the Territory", fontname=SANS, fontsize=6.6, color=MUTED, pad=3)

    # -------------------------------------------- panel 2: the real 20x20 block
    ax2 = fig.add_axes([0.285, 0.20, 0.27, 0.66])
    rgb = np.empty(blk.shape + (3,))
    for cond, col in (((blk == 0), SLATE), (((blk > 0) & (blk <= 7)), GREEN),
                      ((blk >= 8), RED)):
        rgb[cond] = matplotlib.colors.to_rgb(col)
    rgb[blk == 0] = matplotlib.colors.to_rgb("#D9DDE0")
    ax2.imshow(rgb, interpolation="nearest")
    for k in range(BLOCK + 1):
        ax2.axhline(k - .5, color="white", lw=.35)
        ax2.axvline(k - .5, color="white", lw=.35)
    for sp in ax2.spines.values():
        sp.set_color(INK); sp.set_linewidth(1.1)
    ax2.set_xticks([]); ax2.set_yticks([])
    ax2.set_title("20 × 20 pixels, 250 m each", fontname=SANS, fontsize=6.6,
                  color=MUTED, pad=3)

    # ------------------------------------------------ panel 3: the one answer
    ax3 = fig.add_axes([0.625, 0.315, 0.185, 0.44])
    ax3.add_patch(Rectangle((0, 0), 1, 1, fc=RED, ec=INK, lw=1.1))
    ax3.text(.5, .5, "late", ha="center", va="center", fontname=SERIF,
             fontsize=10.5, color="white")
    ax3.set_xlim(0, 1); ax3.set_ylim(0, 1)
    ax3.set_aspect("equal", adjustable="box")
    ax3.axis("off")
    ax3.set_title("one answer", fontname=SANS, fontsize=6.6, color=MUTED, pad=3)

    for x in (0.225, 0.565):
        fig.patches.append(FancyArrow(x, 0.53, 0.045, 0, width=0.004,
                                      head_width=0.028, head_length=0.018,
                                      transform=fig.transFigure, color=DIM,
                                      length_includes_head=True, figure=fig))

    fig.text(0.845, 0.655, f"{late*100:.1f}%", fontname=SERIF, fontsize=8.6, color=RED)
    fig.text(0.845, 0.578, "after July", fontname=SANS, fontsize=6.2, color=MUTED)
    fig.text(0.845, 0.452, f"{early*100:.1f}%", fontname=SERIF, fontsize=8.6, color=GREEN)
    fig.text(0.845, 0.375, "before August", fontname=SANS, fontsize=6.2, color=MUTED)

    fig.text(0.015, 0.055,
             "One square is 0.05° across, about five kilometres, and holds 400 satellite "
             "pixels. Whichever season\nholds more of the burnt area names the year. "
             "Real block: west of Katherine, 2019.",
             fontname=SANS, fontsize=6.0, color=MUTED, linespacing=1.5)

    fig.savefig(OUT, dpi=300, transparent=True)
    print(f"wrote {OUT.relative_to(ROOT)}  ({early*100:.1f}% early, {late*100:.1f}% late)")


if __name__ == "__main__":
    main()
