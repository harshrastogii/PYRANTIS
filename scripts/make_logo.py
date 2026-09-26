"""Cut the white studio background off the logo without eating the glass inside it.

The icon is a pale glass tile on a white page, so a plain "make white transparent" pass
would punch holes through the tile itself. Instead the background is found by flooding
inward from the border: only white that connects to the edge is removed, and the pale
interior, which is enclosed by the tile's darker rim, survives.

Writes a full-size PNG for the page and the small sizes a browser asks for.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage

ROOT = Path(__file__).resolve().parent.parent
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else None
OUT = ROOT / "web" / "assets"

WHITE = 238          # brightness at or above this counts as background
SPREAD = 26          # how neutral a pixel must be; the tile's tint is bluer than this


def _emit(im: Image.Image) -> None:
    """Crop to the artwork and write the sizes a browser asks for.

    Square canvases keep their aspect: a letterform cropped tight and then squashed to a
    square favicon reads as a different mark.
    """
    im = im.crop(im.getbbox())
    w, h = im.size
    side = max(w, h)
    square = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    square.paste(im, ((side - w) // 2, (side - h) // 2))
    print(f"cropped to {w} x {h}, padded to {side} square")

    OUT.mkdir(parents=True, exist_ok=True)
    square.resize((512, 512), Image.LANCZOS).save(OUT / "pyrantis-logo.png")
    # Only the header mark is drawn from this; the favicons are their own set.
    square.resize((180, 180), Image.LANCZOS).save(OUT / "pyrantis-180.png")
    for f in sorted(OUT.glob("pyrantis*.png")):
        print(f"  {f.name}  {f.stat().st_size/1000:.0f} kB")


def main() -> None:
    assert SRC and SRC.exists(), "pass the path to the logo PNG"
    im = Image.open(SRC).convert("RGBA")
    a = np.array(im)
    rgb = a[..., :3].astype(np.int16)

    # A source that already carries transparency is left alone. Running the white cut on
    # it would replace a correct alpha channel with a solid one and turn the cut-out
    # background black.
    if (a[..., 3] == 0).any():
        print("source already has a transparent background, keeping it")
        _emit(im)
        return

    nearly_white = (rgb.min(axis=2) >= WHITE) & (np.ptp(rgb, axis=2) <= SPREAD)

    # Only the white that reaches the border is background. Anything enclosed by the
    # tile's rim is part of the artwork.
    lab, n = ndimage.label(nearly_white)
    edge = set(lab[0, :]) | set(lab[-1, :]) | set(lab[:, 0]) | set(lab[:, -1])
    edge.discard(0)
    background = np.isin(lab, list(edge))
    print(f"{n} white regions, {len(edge)} of them touch the edge, "
          f"{background.mean()*100:.1f}% of the image removed")

    alpha = np.where(background, 0, 255).astype(np.uint8)
    a[..., 3] = alpha
    im = Image.fromarray(a)

    # Soften the cut by one pixel so the tile's edge does not read as a jagged crop.
    m = Image.fromarray(alpha).filter(ImageFilter.GaussianBlur(0.7))
    im.putalpha(m)

    _emit(im)


if __name__ == "__main__":
    main()
