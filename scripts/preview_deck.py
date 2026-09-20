"""Render the built .pptx to images, and report text that does not fit.

This reads the generated file rather than the code that generated it, so a
mistake in the builder -- a box in the wrong place, a run that never got its
text -- shows up here the same way a design mistake would. It uses the Calibri
and Cambria files that ship with the installed Office, which is what makes the
overflow numbers trustworthy: the wrapping is measured with the same metrics
PowerPoint will use.
"""

from __future__ import annotations

import io
import pathlib
import sys

from PIL import Image, ImageDraw, ImageFont
from pptx import Presentation
from pptx.util import Emu

ROOT = pathlib.Path(__file__).resolve().parents[1]
DECK = ROOT / "reports" / "PYRANTIS_presentation.pptx"
OUT = ROOT / "reports" / "_preview"

DF = pathlib.Path("/Applications/Microsoft PowerPoint.app/Contents/Resources/DFonts")
FONTS = {
    ("Calibri", False): DF / "Calibri.ttf",
    ("Calibri", True): DF / "Calibrib.ttf",
    ("Cambria", False): DF / "Cambria.ttc",
    ("Cambria", True): DF / "Cambriab.ttf",
}
PXI = 120                      # pixels per inch
EMU_IN = 914400.0
_cache: dict = {}


def font(name, bold, size_pt):
    px = max(1, int(round(size_pt * PXI / 72.0)))
    k = (name, bold, px)
    if k in _cache:
        return _cache[k]
    path = FONTS.get((name, bold)) or FONTS[("Calibri", bold)]
    try:
        f = ImageFont.truetype(str(path), px)
    except Exception:
        f = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", px)
    _cache[k] = f
    return f


def emu_px(v):
    return (v or 0) / EMU_IN * PXI


def rgb(c, default=(0, 0, 0)):
    try:
        if c and c.type is not None and c.rgb is not None:
            return tuple(c.rgb)
    except Exception:
        pass
    return default


def tokens(para):
    """Split a paragraph into styled words, keeping each run's own styling."""
    out = []
    for r in para.runs:
        f = r.font
        name = f.name or "Calibri"
        size = f.size.pt if f.size else 18.0
        bold = bool(f.bold)
        col = rgb(f.color, (0x11, 0x11, 0x11))
        parts = r.text.replace("\n", " ").split(" ")
        for i, w in enumerate(parts):
            out.append({"w": w, "sp": i < len(parts) - 1, "f": name,
                        "s": size, "b": bold, "c": col})
    return out


def layout(para, width_px, draw):
    """Wrap one paragraph into lines. Returns (lines, total_height)."""
    toks = tokens(para)
    if not toks:
        return [], 0.0
    ls = para.line_spacing or 1.2
    lines, cur, curw = [], [], 0.0
    for t in toks:
        fo = font(t["f"], t["b"], t["s"])
        wl = draw.textlength(t["w"], font=fo)
        sp = draw.textlength(" ", font=fo) if t["sp"] else 0.0
        if cur and curw + wl > width_px + 0.5:
            lines.append(cur); cur, curw = [], 0.0
        cur.append((t, wl, sp)); curw += wl + sp
    if cur:
        lines.append(cur)
    h = 0.0
    for ln in lines:
        mx = max(t["s"] for t, _, _ in ln)
        h += mx * PXI / 72.0 * (ls if isinstance(ls, float) else 1.2)
    return lines, h


def draw_text(img, d, sh, report, idx):
    tf = sh.text_frame
    x, y = emu_px(sh.left), emu_px(sh.top)
    w, h = emu_px(sh.width), emu_px(sh.height)
    total = 0.0
    blocks = []
    for p in tf.paragraphs:
        lines, ph = layout(p, w, d)
        sa = (p.space_after.pt if p.space_after else 0) * PXI / 72.0
        blocks.append((p, lines, ph, sa))
        total += ph + sa
    cy = y
    for p, lines, ph, sa in blocks:
        for ln in lines:
            mx = max((t["s"] for t, _, _ in ln), default=12)
            ls = p.line_spacing or 1.2
            lh = mx * PXI / 72.0 * (ls if isinstance(ls, float) else 1.2)
            lw = sum(wl + sp for _, wl, sp in ln)
            al = str(p.alignment)
            cx = x + (w - lw) / 2 if "CENTER" in al else (x + w - lw if "RIGHT" in al else x)
            for t, wl, sp in ln:
                fo = font(t["f"], t["b"], t["s"])
                d.text((cx, cy + lh - mx * PXI / 72.0 * 0.92), t["w"], font=fo, fill=t["c"])
                cx += wl + sp
            cy += lh
        cy += sa
    if total > h + 2:
        report.append(f"    overflow  slide {idx}  by {(total - h) / PXI:.2f}in  "
                      f"'{sh.text_frame.text[:46].strip()}'")


def rounded(d, box, r, fill):
    try:
        d.rounded_rectangle(box, radius=r, fill=fill)
    except Exception:
        d.rectangle(box, fill=fill)


def render():
    prs = Presentation(DECK)
    SW, SH = emu_px(prs.slide_width), emu_px(prs.slide_height)
    OUT.mkdir(parents=True, exist_ok=True)
    report = []
    for i, s in enumerate(prs.slides, 1):
        bg = (0xF7, 0xF5, 0xF2)
        try:
            if s.background.fill.type is not None:
                bg = rgb(s.background.fill.fore_color, bg)
        except Exception:
            pass
        img = Image.new("RGB", (int(SW), int(SH)), bg)
        d = ImageDraw.Draw(img)
        for sh in s.shapes:
            x, y = emu_px(sh.left), emu_px(sh.top)
            w, h = emu_px(sh.width), emu_px(sh.height)
            if x < -1 or y < -1 or x + w > SW + 1 or y + h > SH + 1:
                report.append(f"    off-slide slide {i}  {sh.shape_type}  "
                              f"({x/PXI:.2f},{y/PXI:.2f}) {w/PXI:.2f}x{h/PXI:.2f}in")
            if sh.shape_type == 13 or sh.__class__.__name__ == "Picture":
                try:
                    pic = Image.open(io.BytesIO(sh.image.blob)).convert("RGB")
                    img.paste(pic.resize((max(1, int(w)), max(1, int(h))), Image.LANCZOS),
                              (int(x), int(y)))
                except Exception as e:
                    report.append(f"    picture failed slide {i}: {e}")
                continue
            if sh.has_text_frame and sh.text_frame.text.strip() == "" and sh.shape_type != 17:
                try:
                    f = rgb(sh.fill.fore_color, None)
                except Exception:
                    f = None
                if f:
                    rounded(d, [x, y, x + w, y + h], min(14, int(min(w, h) / 2.2)), f)
                continue
            try:
                f = rgb(sh.fill.fore_color, None)
                if f:
                    rounded(d, [x, y, x + w, y + h], min(14, int(min(w, h) / 2.2)), f)
            except Exception:
                pass
            if sh.has_text_frame:
                draw_text(img, d, sh, report, i)
        img.save(OUT / f"slide-{i:02d}.png")
    print(f"rendered {len(prs.slides._sldIdLst)} slides -> {OUT.relative_to(ROOT)}")
    if report:
        print(f"\n  {len(report)} issue(s):")
        for r in report:
            print(r)
    else:
        print("\n  no overflow or off-slide shapes")


if __name__ == "__main__":
    render()
