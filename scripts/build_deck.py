"""Assemble the Assessment 3 presentation.

The deck borrows the product's colour language rather than inventing a new one:
slate means no fire, green means a fire before August, ember means a fire after
July, on every slide and in every chart. The one repeated device is that
three-square key, so the audience learns the code once on the title slide and
never has to be told again.

Dark slides open each act and light slides carry the argument, which keeps the
three speakers' sections visually distinct without a single accent stripe.
"""

from __future__ import annotations

import pathlib

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt, Emu

ROOT = pathlib.Path(__file__).resolve().parents[1]
DECK = ROOT / "reports" / "deck"
SHOTS = DECK / "shots"
OUT = ROOT / "reports" / "PYRANTIS_presentation.pptx"

W, H = 13.333, 7.5

INK     = RGBColor(0x0F, 0x2A, 0x35)
INK_SOFT= RGBColor(0x17, 0x3B, 0x49)
PAPER   = RGBColor(0xF7, 0xF5, 0xF2)
WHITE   = RGBColor(0xFF, 0xFF, 0xFF)
BLUE    = RGBColor(0x24, 0x5F, 0x80)
BLUE_L  = RGBColor(0x4A, 0x90, 0xB4)
EMBER   = RGBColor(0xC8, 0x55, 0x2F)
EMBER_L = RGBColor(0xE0, 0x83, 0x5C)
GREEN   = RGBColor(0x2E, 0x7D, 0x5B)
SLATE   = RGBColor(0x9A, 0xA8, 0xAE)
MUTED   = RGBColor(0x6B, 0x7C, 0x85)
ON_DARK = RGBColor(0xB9, 0xCA, 0xD2)
RULE    = RGBColor(0xDC, 0xD8, 0xD2)
CARD    = RGBColor(0xFF, 0xFF, 0xFF)

DISPLAY = "Cambria"
BODY    = "Calibri"

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(W), Inches(H)
BLANK = prs.slide_layouts[6]


# ---------------------------------------------------------------- primitives

def slide(dark=False, notes=""):
    s = prs.slides.add_slide(BLANK)
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = INK if dark else PAPER
    if notes:
        s.notes_slide.notes_text_frame.text = notes
    return s


def hang(p, indent=0.30):
    """APA reference entries take a hanging indent, which python-pptx does not expose."""
    pPr = p._p.get_or_add_pPr()
    pPr.set("marL", str(int(indent * 914400)))
    pPr.set("indent", str(int(-indent * 914400)))


def text(s, x, y, w, h, runs, size=16, color=INK, font=BODY, bold=False,
         align=PP_ALIGN.LEFT, spacing=1.25, space_after=0, anchor=MSO_ANCHOR.TOP,
         hanging=False):
    """Place a text box. `runs` is a string, or a list of paragraphs, where each
    paragraph is a string or a list of (text, overrides) run tuples."""
    box = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    paras = [runs] if isinstance(runs, str) else runs
    # A newline inside a run is not a paragraph break in OOXML, and PowerPoint
    # renders it inconsistently -- so split those into real paragraphs here.
    expanded = []
    for para in paras:
        if isinstance(para, str) and "\n" in para:
            expanded.extend(para.split("\n"))
        else:
            expanded.append(para)
    paras = expanded
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = spacing
        p.space_after = Pt(space_after)
        if hanging:
            hang(p)
        bits = [(para, {})] if isinstance(para, str) else para
        for t, over in bits:
            r = p.add_run()
            r.text = t
            f = r.font
            f.name = over.get("font", font)
            f.size = Pt(over.get("size", size))
            f.bold = over.get("bold", bold)
            f.italic = over.get("italic", False)
            f.color.rgb = over.get("color", color)
    return box


def rect(s, x, y, w, h, fill=CARD, radius=0.035, line=None, shape=None):
    sh = s.shapes.add_shape(
        shape or (MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE),
        Inches(x), Inches(y), Inches(w), Inches(h))
    if radius:
        try:
            sh.adjustments[0] = radius
        except Exception:
            pass
    sh.fill.solid()
    sh.fill.fore_color.rgb = fill
    if line is None:
        sh.line.fill.background()
    else:
        sh.line.color.rgb = line
        sh.line.width = Pt(1)
    sh.shadow.inherit = False
    sh.text_frame.text = ""
    return sh


def key(s, x, y, size=0.17, gap=0.075, colors=(SLATE, GREEN, EMBER)):
    """The three-square class key -- the deck's one repeated mark."""
    for i, c in enumerate(colors):
        rect(s, x + i * (size + gap), y, size, size, fill=c, radius=0.18)


def picture(s, name, x, y, w=None, h=None, folder=DECK):
    p = folder / name
    if not p.exists():
        raise FileNotFoundError(p)
    kw = {}
    if w: kw["width"] = Inches(w)
    if h: kw["height"] = Inches(h)
    return s.shapes.add_picture(str(p), Inches(x), Inches(y), **kw)


def fit(name, box_w, box_h, folder=DECK):
    """Scale an image to fit a box, returning (w, h) in inches."""
    from PIL import Image
    im = Image.open(folder / name)
    r = min(box_w / im.width, box_h / im.height)
    return im.width * r, im.height * r


def shot(s, name, x, y, box_w, box_h, caption=None, frame=True):
    """Drop a product screenshot in, centred in its box, with a hairline frame."""
    w, h = fit(name + ".png", box_w, box_h, SHOTS)
    px, py = x + (box_w - w) / 2, y + (box_h - h) / 2
    if frame:
        rect(s, px - 0.035, py - 0.035, w + 0.07, h + 0.07, fill=RULE, radius=0.02)
    picture(s, name + ".png", px, py, w=w, folder=SHOTS)
    if caption:
        text(s, px, py + h + 0.14, w, 0.3, caption, size=11.5, color=MUTED,
             align=PP_ALIGN.CENTER)
    return px, py, w, h


def title(s, txt, y=0.62, dark=False, size=34, x=0.9, w=11.6):
    text(s, x, y, w, 1.0, txt, size=size, font=DISPLAY, bold=True,
         color=WHITE if dark else INK, spacing=1.05)


def kicker(s, txt, y=0.34, dark=False, x=0.9):
    text(s, x, y, 11.6, 0.3, txt.upper(), size=11, font=BODY, bold=True,
         color=EMBER_L if dark else EMBER, spacing=1.0)


def stat(s, x, y, value, label, w=3.0, color=INK, vsize=46, lsize=12.5, dark=False):
    text(s, x, y, w, 0.85, value, size=vsize, font=DISPLAY, bold=True, color=color,
         spacing=0.95)
    text(s, x, y + vsize / 72.0 * 0.95 + 0.06, w, 0.7, label, size=lsize,
         color=ON_DARK if dark else MUTED, spacing=1.2)


def footer(s, who, n, dark=False):
    text(s, 0.9, 6.92, 6.0, 0.3, who, size=10, color=ON_DARK if dark else MUTED)
    text(s, 10.5, 6.92, 2.0, 0.3, str(n), size=10, color=ON_DARK if dark else MUTED,
         align=PP_ALIGN.RIGHT)


def divider(s, act, head, who, sub):
    key(s, 0.9, 2.28)
    kicker(s, act, y=2.75, dark=True)
    text(s, 0.9, 3.05, 10.5, 1.4, head, size=44, font=DISPLAY, bold=True,
         color=WHITE, spacing=1.02)
    text(s, 0.9, 4.55, 8.6, 0.9, sub, size=15.5, color=ON_DARK, spacing=1.4)
    text(s, 0.9, 5.75, 8.0, 0.35, who, size=12.5, bold=True, color=EMBER_L)


# ------------------------------------------------------------------- slides

def s01_title():
    s = slide(dark=True, notes="Harsh opens. Name the place, the question, the promise.")
    text(s, 0.9, 0.78, 11.5, 0.3, "CHARLES DARWIN UNIVERSITY", size=11.5, bold=True,
         color=ON_DARK, spacing=1.0)
    key(s, 0.9, 1.52, size=0.2, gap=0.09)
    text(s, 0.9, 2.02, 11.5, 1.1, "PYRANTIS", size=56, font=DISPLAY, bold=True,
         color=WHITE, spacing=1.0)
    text(s, 0.9, 2.98, 9.6, 0.7,
         "Which country burns in the Northern Territory, and when",
         size=22, font=DISPLAY, color=ON_DARK, spacing=1.15)
    text(s, 0.9, 3.72, 9.8, 0.4,
         "Multi-class classification of 46,445 map squares across 26 fire years, "
         "and a forecast for 2026.",
         size=13.5, color=ON_DARK, spacing=1.4)

    rows = [("Unit", "Machine Learning, Artificial Intelligence and Algorithms"),
            ("Unit code", "PRT565"),
            ("Assessment", "Assessment 3 — Presentation"),
            ("Group", "Group 85")]
    y = 4.52
    for k, v in rows:
        text(s, 0.9, y, 1.5, 0.3, k, size=10.5, color=ON_DARK)
        text(s, 2.55, y, 5.0, 0.3, v, size=11.5, bold=True, color=WHITE)
        y += 0.44

    text(s, 8.15, 4.52, 4.3, 0.3, "Team", size=10.5, color=ON_DARK)
    members = [("Harsh Rastogi", "386401"), ("Saira Zafar", "407193"),
               ("Tharushi Wimalachandra", "386594")]
    y = 4.96
    for nm, sid in members:
        text(s, 8.15, y, 3.2, 0.3, nm, size=11.5, bold=True, color=WHITE)
        text(s, 11.6, y, 0.85, 0.3, sid, size=11, color=ON_DARK, align=PP_ALIGN.RIGHT)
        y += 0.44

    rect(s, 0.9, 6.28, 11.5, 0.62, fill=INK_SOFT, radius=0.06)
    text(s, 1.24, 6.46, 3.0, 0.3, "Recording (unlisted)", size=10.5, color=ON_DARK)
    text(s, 4.1, 6.46, 4.6, 0.3,
         "[ paste the unlisted video link here ]", size=11, color=EMBER_L)
    text(s, 9.1, 6.46, 3.0, 0.3, "pyrantis.harshlabs.workers.dev", size=10.5,
         color=ON_DARK, align=PP_ALIGN.RIGHT)


def s02_burns():
    s = slide(notes="The scale of it. Fire here is not an emergency, it is the weather.")
    kicker(s, "The setting")
    title(s, "Most of the Territory burns. The question is when.")
    text(s, 0.9, 1.75, 6.1, 1.9,
         "Across 2000 to 2025, roughly a third of the Northern Territory carried a "
         "fire in any given year. Nearly a million square-years of record say the "
         "same thing: burning is not the exception here, it is the season.",
         size=15, color=INK, spacing=1.5)
    # The chart sits below both statistics, so its caption cannot run into the
    # label under 46,445.
    w, h = fit("deck_balance.png", 11.5, 2.15)
    px = (W - w) / 2
    text(s, px, 4.02, w, 0.35, "Every square, every year, 2005 to 2025",
         size=11.5, color=MUTED, align=PP_ALIGN.CENTER)
    picture(s, "deck_balance.png", px, 4.38, w=w)
    stat(s, 7.6, 1.72, "1,207,570", "square-years of fire record", w=4.6, vsize=40)
    stat(s, 7.6, 2.86, "46,445", "map squares, five kilometres across", w=4.6, vsize=40)
    footer(s, "Harsh Rastogi", 2)


def s03_when():
    s = slide(notes="Why the month is the whole game. Early is management, late is damage.")
    kicker(s, "Why it matters")
    title(s, "A fire in June and a fire in October are different events.")
    cards = [
        (0.9, GREEN, "Before August",
         "The country is still damp from the wet. Fires are cool, they creep, and they "
         "leave a patchwork. Rangers light these deliberately to break up the fuel."),
        (6.95, EMBER, "After July",
         "The grass has cured. Fires run with the wind, burn hot enough to kill canopy, "
         "and take out fences, stock and country in a single afternoon."),
    ]
    for x, col, head, body in cards:
        rect(s, x, 1.78, 5.45, 1.72, fill=CARD, radius=0.04)
        rect(s, x + 0.42, 2.08, 0.2, 0.2, fill=col, radius=0.18)
        text(s, x + 0.78, 2.03, 4.3, 0.35, head, size=17, font=DISPLAY, bold=True, color=col)
        text(s, x + 0.42, 2.52, 4.68, 0.9, body, size=13, color=INK, spacing=1.38)
    w, h = fit("deck_suppression.png", 7.2, 2.6)
    picture(s, "deck_suppression.png", 0.9, 3.78, w=w)
    text(s, 8.5, 3.95, 3.9, 2.2,
         [[("And it compounds.", {"bold": True, "size": 15, "font": DISPLAY})],
          [("In the Top End, country burned early is far less likely to carry a late "
            "fire the next year — 20.1% against 31.9% for country left alone. "
            "Territory-wide the comparison reverses, because the arid south burns "
            "on a different clock. One number for the whole map would have hidden "
            "both facts.", {})]],
         size=12.5, color=INK, spacing=1.42, space_after=7)
    footer(s, "Harsh Rastogi", 3)


def s04_target():
    s = slide(dark=True, notes="State the target precisely. Three classes, one cut date.")
    kicker(s, "The question we set", y=1.42, dark=True)
    text(s, 0.9, 1.78, 11.2, 1.0,
         "For every square, for every year: which of three?",
         size=34, font=DISPLAY, bold=True, color=WHITE, spacing=1.08)
    rows = [
        (SLATE, "unburnt", "no fire detected that year"),
        (GREEN, "early",   "first detected on or before 31 July"),
        (EMBER, "late",    "first detected after 31 July"),
    ]
    y = 3.15
    for col, name, desc in rows:
        rect(s, 0.9, y + 0.09, 0.22, 0.22, fill=col, radius=0.18)
        text(s, 1.42, y, 2.4, 0.4, name, size=21, font=DISPLAY, bold=True, color=WHITE)
        text(s, 4.0, y + 0.05, 7.6, 0.4, desc, size=15, color=ON_DARK)
        y += 0.82
    text(s, 0.9, 5.92, 10.6, 0.6,
         "The 31 July line is the one the Territory's own fire management uses, so the "
         "classes mean something to the people who would act on them.",
         size=13, color=ON_DARK, spacing=1.4)
    footer(s, "Harsh Rastogi", 4, dark=True)


def s05_data():
    s = slide(notes="Three public sources, all open. Say the licence line.")
    kicker(s, "The data")
    title(s, "Three public sources, joined on one grid.")
    srcs = [
        (BLUE, "NAFI", "North Australian Fire Information",
         "250 m burnt-area rasters, 2000 to 2025. Each pixel carries the month it "
         "was detected — which is what makes an early/late label possible at all."),
        (EMBER, "SILO", "Queensland Government Data Drill",
         "Daily rainfall, temperature, humidity and radiation, interpolated from "
         "Bureau of Meteorology stations to the same 0.05° grid."),
        (GREEN, "MODIS", "MOD13Q1 via Google Earth Engine",
         "Sixteen-day vegetation index. How green the country got over the wet "
         "season is how much fuel is standing when the dry arrives."),
    ]
    x = 0.9
    for col, name, sub, body in srcs:
        rect(s, x, 1.78, 3.72, 2.62, fill=CARD, radius=0.04)
        rect(s, x + 0.4, 2.1, 0.2, 0.2, fill=col, radius=0.18)
        text(s, x + 0.4, 2.46, 3.0, 0.4, name, size=21, font=DISPLAY, bold=True, color=INK)
        text(s, x + 0.4, 2.86, 3.0, 0.35, sub, size=10.5, color=MUTED)
        text(s, x + 0.4, 3.28, 2.95, 1.0, body, size=12, color=INK, spacing=1.4)
        x += 4.02
    text(s, 0.9, 4.78, 11.5, 0.4,
         [[("All three are open data, used under their own licences and cited in the "
            "report. ", {"color": INK}),
           ("Nothing in this project sits behind a login.", {"bold": True, "color": INK})]],
         size=13.5, spacing=1.4)
    for i, (v, l) in enumerate([("0.05°", "grid cell, about 5 km"),
                                ("2000–2025", "years of fire record"),
                                ("50", "features per square-year"),
                                ("966,819", "rows used for modelling")]):
        stat(s, 0.9 + i * 3.0, 5.35, v, l, w=2.8, vsize=26, lsize=11.5)
    footer(s, "Harsh Rastogi", 5)


def s06_divider_build():
    s = slide(dark=True, notes="Hand to Saira.")
    divider(s, "Act two", "How we built it",
            "Saira Zafar",
            "Turning satellite scars into labels, deciding what the model is allowed "
            "to know, and proving it never saw the answer.")
    footer(s, "", 6, dark=True)


def s07_labels():
    s = slide(notes="Saira. The labelling pipeline and the two thresholds.")
    kicker(s, "Labelling")
    title(s, "From burnt pixels to one answer per square.")
    steps = [
        ("1", "Read the scar", "Each NAFI pixel holds the month it burned, 1 to 12, "
         "or zero for no fire."),
        ("2", "Pool to a square", "Every 20 × 20 block of pixels becomes one 0.05° cell, "
         "and we count what fraction burned early and what fraction burned late."),
        ("3", "Decide the class", "A square counts as burnt only if at least 5% of it "
         "burned, and is only used at all if at least half its pixels were readable. "
         "Whichever season holds more of the burnt area names the year."),
    ]
    y = 1.82
    for n, head, body in steps:
        rect(s, 0.9, y, 0.52, 0.52, fill=INK, radius=0.5)
        text(s, 0.9, y + 0.1, 0.52, 0.4, n, size=16, font=DISPLAY, bold=True,
             color=WHITE, align=PP_ALIGN.CENTER)
        text(s, 1.72, y + 0.02, 4.0, 0.4, head, size=17, font=DISPLAY, bold=True, color=INK)
        text(s, 1.72, y + 0.44, 5.4, 0.9, body, size=13, color=INK, spacing=1.42)
        y += 1.44
    rect(s, 7.5, 1.82, 4.95, 3.5, fill=CARD, radius=0.04)
    text(s, 7.92, 2.14, 4.1, 0.4, "The check that mattered", size=16, font=DISPLAY,
         bold=True, color=INK)
    text(s, 7.92, 2.62, 4.15, 2.5,
         "Labels are the one thing no metric can catch. So we tested them against fire "
         "history the Territory already knows: 2011, the year after the 2010–11 La Niña, "
         "comes out as the heaviest on record at 65.7% of the map burnt. 2020 comes out "
         "the lightest at 15.9%. Both match the published record. If the labels had been "
         "wrong, every number after this would have been wrong too.",
         size=12.5, color=INK, spacing=1.45)
    footer(s, "Saira Zafar", 7)


def s08_features():
    s = slide(notes="Saira. The 30 April cut-off is the honesty rule.")
    kicker(s, "Features")
    title(s, "Only what a forecaster could actually know on 30 April.", w=6.4)
    text(s, 0.9, 1.75, 6.3, 1.0,
         "The forecast has to be useful before the fire season starts, so every one of "
         "the 50 features is built from information available by the end of April in "
         "the year being predicted — and nothing later.",
         size=14.5, color=INK, spacing=1.5)
    fams = [
        (BLUE, "Fire history", "26 features",
         "Years since the last fire and the last late fire, how often it burnt over "
         "3, 5, 10, 15, 20 and 25 years, what it did last year, and what its "
         "neighbours did."),
        (EMBER, "Weather", "14 features",
         "Wet-season rainfall, heat, humidity deficit and sunlight, plus the dry "
         "season before it, and how far the wet sat from its own local normal."),
        (GREEN, "Greenness", "7 features",
         "How green the country got over the wet, where it peaked, how fast it "
         "greened up, and how far April sat above or below normal."),
        (SLATE, "Place", "3 features",
         "Longitude, latitude and elevation, so the model can tell the Top End from "
         "the arid south without being handed the regions."),
    ]
    y = 2.86
    for col, name, count, body in fams:
        rect(s, 0.9, y, 11.5, 0.9, fill=CARD, radius=0.05)
        rect(s, 1.28, y + 0.35, 0.2, 0.2, fill=col, radius=0.18)
        text(s, 1.66, y + 0.13, 2.5, 0.32, name, size=16, font=DISPLAY, bold=True, color=INK)
        text(s, 1.66, y + 0.5, 2.5, 0.28, count, size=11, color=MUTED)
        text(s, 4.4, y + 0.16, 7.6, 0.62, body, size=12, color=INK, spacing=1.36)
        y += 0.99
    stat(s, 7.6, 1.68, "30 April", "the cut-off, every feature, every year", w=4.8,
         vsize=40, color=EMBER)
    footer(s, "Saira Zafar", 8)


def s09_leakage():
    s = slide(notes="Saira. This is the slide that separates a real result from a lucky one.")
    kicker(s, "Leakage")
    title(s, "Two rules, because a leak would flatter every number after it.")
    text(s, 0.9, 1.72, 11.3, 0.7,
         "A feature that quietly reads the year it is meant to predict will score "
         "beautifully and mean nothing. We did not trust ourselves to avoid that by "
         "being careful, so we tested for it twice.",
         size=14.5, color=INK, spacing=1.5)
    rules = [
        (0.9, "Rule one", "Recomputation",
         "Every fire-history feature is rebuilt from years strictly before the target "
         "year, in one function that both the training path and the forecast path call. "
         "A second implementation would have drifted, and the drift would have been "
         "silent — the forecast has no label to check itself against."),
        (6.95, "Rule two", "Scrambling",
         "We shuffled one year's labels and rebuilt the features. If anything were "
         "reading the answer, the scores would have collapsed or held suspiciously "
         "steady. They moved exactly as much as chance says they should."),
    ]
    for x, tag, head, body in rules:
        rect(s, x, 2.72, 5.45, 2.6, fill=CARD, radius=0.04)
        text(s, x + 0.42, 3.02, 4.6, 0.3, tag.upper(), size=10.5, bold=True, color=EMBER)
        text(s, x + 0.42, 3.36, 4.6, 0.45, head, size=21, font=DISPLAY, bold=True, color=INK)
        text(s, x + 0.42, 3.94, 4.62, 1.3, body, size=12.5, color=INK, spacing=1.45)
    text(s, 0.9, 5.66, 11.4, 0.5,
         "Both checks are scripts in the repository, not claims in a slide.",
         size=13, color=MUTED, spacing=1.4)
    footer(s, "Saira Zafar", 9)


def s10_split():
    s = slide(notes="Saira. Random splits would let the model learn from its own future.")
    kicker(s, "Validation")
    title(s, "Split by time, never at random.")
    text(s, 0.9, 1.72, 11.3, 0.75,
         "Fire years are not independent draws. A random shuffle would put 2024 in "
         "training and 2023 in test, and the model would be reading its own future. "
         "So the record is cut in chronological order and the last three years are "
         "never touched until the end.",
         size=14.5, color=INK, spacing=1.5)
    bands = [(0.9, 6.55, BLUE, "Train", "2005 – 2019", "fifteen years"),
             (7.62, 1.75, BLUE_L, "Tune", "2020 – 2022", "three years"),
             (9.52, 2.9, EMBER, "Test", "2023 – 2025", "three years, seen once")]
    for x, w, col, name, years, note in bands:
        rect(s, x, 3.0, w, 1.28, fill=col, radius=0.04)
        text(s, x + 0.34, 3.22, w - 0.6, 0.4, name, size=19, font=DISPLAY, bold=True,
             color=WHITE)
        text(s, x + 0.34, 3.66, w - 0.6, 0.35, years, size=14, color=WHITE)
        text(s, x + 0.34, 3.98, w - 0.6, 0.3, note, size=10.5,
             color=RGBColor(0xE4, 0xEE, 0xF2))
    text(s, 0.9, 4.52, 11.5, 0.35, "2005", size=11, color=MUTED)
    text(s, 0.9, 4.52, 11.5, 0.35, "2025", size=11, color=MUTED, align=PP_ALIGN.RIGHT)
    rect(s, 0.9, 5.25, 11.5, 1.22, fill=CARD, radius=0.04)
    text(s, 1.3, 5.52, 10.8, 0.75,
         [[("And once was once. ", {"bold": True}),
           ("The test years were scored a single time, at the end, after every "
            "decision about features and models had already been made. Tuning against "
            "them would have turned the held-out years into another training set.", {})]],
         size=13, color=INK, spacing=1.45)
    footer(s, "Saira Zafar", 10)


def s11_models():
    s = slide(notes="Saira. Five required models, plus gradient boosting, plus weighting.")
    kicker(s, "The models")
    title(s, "Six models, two baselines, and one correction for imbalance.")
    left = [("Decision tree", "the readable one"),
            ("Random forest", "many trees, voting"),
            ("Naive Bayes", "the probabilistic floor"),
            ("Multilayer ANN", "a dense network"),
            ("LSTM", "reads the 26-year sequence"),
            ("Gradient boosting", "trees that correct each other")]
    y = 1.86
    for i, (name, note) in enumerate(left):
        col = EMBER if name == "Gradient boosting" else BLUE
        rect(s, 0.9, y, 0.14, 0.32, fill=col, radius=0.3)
        text(s, 1.28, y - 0.02, 3.0, 0.35, name, size=15, bold=True, color=INK)
        text(s, 4.2, y, 2.6, 0.32, note, size=12, color=MUTED)
        y += 0.56
    text(s, 0.9, 5.36, 6.2, 0.9,
         [[("Against two baselines: ", {"bold": True}),
           ("\"whatever happened last year\" and \"always say unburnt\". A model that "
            "cannot beat those has learned nothing worth having.", {})]],
         size=13, color=INK, spacing=1.45)
    rect(s, 7.5, 1.82, 4.95, 4.1, fill=CARD, radius=0.04)
    text(s, 7.94, 2.14, 4.1, 0.45, "Why class weighting", size=19, font=DISPLAY,
         bold=True, color=INK)
    text(s, 7.94, 2.72, 4.15, 1.5,
         "Late fires are the ones that matter and the rarest thing on the map. Left "
         "unweighted, every model quietly stopped predicting them — it is cheaper to "
         "be right about the common answer. Weighting the classes makes that trade "
         "expensive.",
         size=12.5, color=INK, spacing=1.45)
    text(s, 7.94, 4.42, 4.1, 0.3, "GRADIENT BOOSTING, F1 ON LATE FIRES", size=9.5,
         bold=True, color=MUTED)
    text(s, 7.94, 4.78, 2.0, 0.6, "0.420", size=30, font=DISPLAY, bold=True, color=SLATE)
    text(s, 7.94, 5.32, 2.0, 0.3, "unweighted", size=11, color=MUTED)
    text(s, 10.2, 4.78, 2.0, 0.6, "0.553", size=30, font=DISPLAY, bold=True, color=EMBER)
    text(s, 10.2, 5.32, 2.0, 0.3, "weighted", size=11, color=MUTED)
    footer(s, "Saira Zafar", 11)


def s12_divider_results():
    s = slide(dark=True, notes="Hand to Tharushi.")
    divider(s, "Act three", "Does it actually work?",
            "Tharushi Wimalachandra",
            "Scores on years the models never saw, sixty-six separate re-tests, and "
            "the twenty-three experiments behind the one we kept.")
    footer(s, "", 12, dark=True)


def s13_results():
    s = slide(notes="Tharushi. Read the gap to the baseline, not the absolute number.")
    kicker(s, "Results")
    title(s, "Gradient boosting wins, and every model beats the baselines.")
    w, h = fit("deck_models.png", 7.7, 4.8)
    picture(s, "deck_models.png", 0.72, 1.82, w=w)
    rect(s, 8.75, 1.86, 3.7, 4.72, fill=CARD, radius=0.04)
    text(s, 9.12, 2.16, 3.0, 0.4, "Read it this way", size=17, font=DISPLAY, bold=True,
         color=INK)
    text(s, 9.12, 2.66, 3.05, 3.0,
         "Balanced F1 treats all three answers as equally important, so a model cannot "
         "score well by ignoring the rare ones.\n\n"
         "The number to watch is the gap to \"same as last year\" at 0.506. That is "
         "what a fire officer already knows without us. Gradient boosting adds 0.178 "
         "on top of it.\n\n"
         "Accuracy on the same years is 0.725.",
         size=12.5, color=INK, spacing=1.45)
    footer(s, "Tharushi Wimalachandra", 13)


def s14_layers():
    s = slide(notes="Tharushi. Each layer of data earned its place or it went back.")
    kicker(s, "What the data bought")
    title(s, "Weather earned its place. Greenness earned less, and we said so.")
    w, h = fit("deck_layers.png", 7.5, 4.0)
    picture(s, "deck_layers.png", 0.8, 2.0, w=w)
    text(s, 8.6, 1.98, 3.95, 4.1,
         [[("Adding the weather was the single biggest step in the project — "
            "+0.035, from knowing how wet the wet season was.", {})],
          [("The third run adds greenness, longer fire memory and elevation together, "
            "for +0.010. Isolated in the experiment ledger, greenness on its own is "
            "worth +0.029.", {})],
          [("We kept it, because it holds up on years the model never saw and because "
            "two of the model's five strongest inputs turn out to be greenness — even "
            "though Earth Engine cost ten minutes a run against sixty seconds for "
            "everything else.", {"bold": True})]],
         size=12.8, color=INK, spacing=1.45, space_after=10)
    footer(s, "Tharushi Wimalachandra", 14)


def s15_walk():
    s = slide(notes="Tharushi. One test year could be luck. Sixty-six is a pattern.")
    kicker(s, "Walk-forward testing")
    title(s, "One good year could be luck. So we did it sixty-six times.")
    text(s, 0.9, 1.72, 11.4, 0.6,
         "We cut the record at each year from 2014 to 2024, let the model learn only "
         "what came before the cut, and scored it on every year after — 66 separate tests.",
         size=14, color=INK, spacing=1.45)
    w, h = fit("deck_walk.png", 7.9, 3.7)
    picture(s, "deck_walk.png", 0.72, 2.52, w=w)
    rect(s, 8.75, 2.55, 3.7, 3.35, fill=CARD, radius=0.04)
    text(s, 9.12, 2.85, 3.1, 0.4, "The surprise", size=17, font=DISPLAY, bold=True,
         color=INK)
    text(s, 9.12, 3.35, 3.05, 2.4,
         "Skill does not fade with distance. Guessing one year ahead scores 0.635; "
         "eleven years ahead it still scores 0.604.\n\n"
         "That is not the model being clever. It is telling us the strongest clue is "
         "the wet season just gone, which is known by the end of April however old the "
         "fire history is.",
         size=12.5, color=INK, spacing=1.45)
    footer(s, "Tharushi Wimalachandra", 15)


def s16_experiments():
    s = slide(notes="Tharushi. Name the CNN failure plainly. It is the credible slide.")
    kicker(s, "The ledger")
    title(s, "Twenty-three experiments against a fixed benchmark. One was kept.")
    w, h = fit("deck_experiments.png", 8.0, 4.45)
    picture(s, "deck_experiments.png", 0.72, 1.9, w=w)
    rect(s, 8.9, 1.94, 3.55, 4.4, fill=CARD, radius=0.04)
    text(s, 9.26, 2.24, 3.0, 0.4, "Including the one that failed",
         size=16, font=DISPLAY, bold=True, color=INK)
    text(s, 9.26, 2.86, 2.95, 2.9,
         "We built a U-Net over the whole grid to let a convolutional network read "
         "fire as a picture rather than a table of numbers.\n\n"
         "It scored 0.588 — 0.048 below the benchmark — so it went in the ledger as "
         "a reverted run and stayed out of the final model.\n\n"
         "Every run was scored the same way, against the same fixed benchmark, and "
         "recorded whether it helped or not.",
         size=12.3, color=INK, spacing=1.42)
    footer(s, "Tharushi Wimalachandra", 16)


def s17_importance():
    s = slide(notes="Tharushi. The model agrees with what fire people already say.")
    kicker(s, "What it leans on")
    title(s, "The model rediscovered what land managers already knew.")
    w, h = fit("deck_importance.png", 8.1, 4.5)
    picture(s, "deck_importance.png", 0.72, 1.94, w=w)
    text(s, 9.0, 2.1, 3.5, 3.6,
         [[("Nobody told it that a wet, hot wet season grows the grass that carries "
            "an October fire. It found that in the data.", {})],
          [("The top five inputs are the dryness of the air, average greenness, "
            "wet-season rainfall, what the neighbouring country did last year, and "
            "the lowest greenness of the wet.", {})],
          [("That it lands where fire ecology already sits is the best evidence we "
            "have that it learned the problem rather than the dataset.",
            {"bold": True})]],
         size=12.6, color=INK, spacing=1.45, space_after=10)
    footer(s, "Tharushi Wimalachandra", 17)


def s18_limits():
    s = slide(notes="Tharushi. Say these before a marker has to ask.")
    kicker(s, "Limitations")
    title(s, "What this does not do.")
    lims = [
        ("Late fires are still the hardest call", "0.553",
         "The class that matters most is the one we predict worst. It is the rarest "
         "and the most weather-driven, and a year's weather after April is exactly "
         "what we refuse to look at."),
        ("A season, not a day", "—",
         "This says which squares are likely to burn and roughly when. It does not "
         "say which Tuesday, and it is not a fire-danger rating."),
        ("2026 is unverified", "—",
         "No burnt-area record exists for a year still running. The forecast is "
         "published as a claim that can be checked in 2027, not as a result."),
        ("One Territory, one grid", "—",
         "Everything is fitted to the NT at 0.05°. Whether it transfers to the "
         "Kimberley or Cape York is an open question, not an assumption."),
    ]
    y = 1.84
    for head, num, body in lims:
        rect(s, 0.9, y, 11.5, 1.08, fill=CARD, radius=0.05)
        text(s, 1.3, y + 0.2, 4.6, 0.35, head, size=15.5, font=DISPLAY, bold=True, color=INK)
        text(s, 1.3, y + 0.6, 4.6, 0.35,
             "" if num == "—" else f"balanced F1 {num}", size=11, color=EMBER, bold=True)
        text(s, 6.4, y + 0.24, 5.6, 0.72, body, size=12.3, color=INK, spacing=1.4)
        y += 1.2
    footer(s, "Tharushi Wimalachandra", 18)


def s19_divider_2026():
    s = slide(dark=True, notes="Back to Harsh for the payoff.")
    divider(s, "Act four", "What it says about 2026",
            "Harsh Rastogi",
            "A forecast that can be checked next year, and the product that puts it "
            "in front of someone who is not a data scientist.")
    footer(s, "", 19, dark=True)


def s20_forecast():
    s = slide(notes="Harsh. This is the number. Say it slowly and say it is falsifiable.")
    kicker(s, "The forecast")
    title(s, "2026 is running heavy, and most of what is left will burn late.")
    w, h = fit("deck_forecast.png", 8.0, 3.8)
    picture(s, "deck_forecast.png", 0.75, 2.35, w=w)
    stat(s, 8.9, 1.9, "51%", "of the Territory forecast to burn in 2026, against a "
         "26-year average of 33%", w=3.6, vsize=58, color=EMBER)
    text(s, 8.9, 3.82, 3.6, 2.75,
         [[("The model reads a wet season that delivered 783 mm — 45% above the 2005 "
            "to 2019 average and the second wettest in 22 years — and the greenest "
            "April on record at 0.46.", {})],
          [("More water grew more grass. More grass is more fuel.", {"bold": True})],
          [("Mean confidence across the map is 0.733. No fire scars exist for 2026 "
            "yet, so this is a claim we can be held to in 2027.",
            {"color": MUTED, "size": 11.8})]],
         size=12.4, color=INK, spacing=1.42, space_after=9)
    footer(s, "Harsh Rastogi", 20)


def s21_product():
    s = slide(notes="Harsh. The work is only worth something if someone can use it.")
    kicker(s, "The product")
    title(s, "A model in a notebook helps nobody.")
    text(s, 0.9, 1.68, 11.4, 0.5,
         "So we shipped it: a public site that answers the question in plain English "
         "first, and shows the map, the record and the evidence underneath.",
         size=14, color=INK, spacing=1.45)
    shot(s, "outlook", 0.72, 2.32, 5.85, 3.85,
         "This year — the forecast for any town or square")
    shot(s, "quality_models", 6.85, 2.32, 5.85, 3.85,
         "How good is it — every model, scored on years it never saw")
    footer(s, "Harsh Rastogi", 21)


def s22_close():
    s = slide(dark=True, notes="Harsh closes. Land on the use, not the metric.")
    key(s, 0.9, 1.6, size=0.22, gap=0.1)
    text(s, 0.9, 2.16, 10.8, 1.4,
         "Every square, every year, and an honest number on how often it is right.",
         size=36, font=DISPLAY, bold=True, color=WHITE, spacing=1.1)
    text(s, 0.9, 3.86, 7.4, 1.5,
         "Fire in the Territory is not going to stop. Knowing in April which country "
         "is likely to carry an October fire is how a burning programme gets planned "
         "while there is still time to plan it.",
         size=15, color=ON_DARK, spacing=1.5)
    items = [("0.684", "balanced F1, years never seen"),
             ("66", "independent re-tests"),
             ("51%", "forecast to burn in 2026")]
    for i, (v, l) in enumerate(items):
        stat(s, 0.9 + i * 3.6, 5.5, v, l, w=3.3, vsize=34, color=EMBER_L, dark=True)
    text(s, 8.7, 3.82, 3.75, 1.7,
         [[("pyrantis.harshlabs.workers.dev", {"bold": True, "color": WHITE, "size": 14})],
          [("github.com/harshrastogii/PYRANTIS", {"color": ON_DARK, "size": 12.5})],
          [("Every figure in this deck is generated from the repository, "
            "not typed in.", {"color": ON_DARK, "size": 11.5})]],
         size=12.5, color=ON_DARK, spacing=1.5, space_after=7)
    footer(s, "", 22, dark=True)


REFERENCES = ['Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5–32. https://doi.org/10.1023/A:1010933404324', 'Chollet, F., & others. (2015). Keras [Computer software]. https://keras.io', 'Didan, K. (2021). MODIS/Terra vegetation indices 16-day L3 global 250 m SIN grid V061 [Data set]. NASA EOSDIS Land Processes Distributed Active Archive Center. https://doi.org/10.5067/MODIS/MOD13Q1.061', 'Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. The Annals of Statistics, 29(5), 1189–1232. https://doi.org/10.1214/aos/1013203451', 'Gorelick, N., Hancher, M., Dixon, M., Ilyushchenko, S., Thau, D., & Moore, R. (2017). Google Earth Engine: Planetary-scale geospatial analysis for everyone. Remote Sensing of Environment, 202, 18–27. https://doi.org/10.1016/j.rse.2017.06.031', 'Harris, C. R., Millman, K. J., van der Walt, S. J., Gommers, R., Virtanen, P., Cournapeau, D., … Oliphant, T. E. (2020). Array programming with NumPy. Nature, 585(7825), 357–362. https://doi.org/10.1038/s41586-020-2649-2', 'Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural Computation, 9(8), 1735–1780. https://doi.org/10.1162/neco.1997.9.8.1735', 'Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. Computing in Science & Engineering, 9(3), 90–95. https://doi.org/10.1109/MCSE.2007.55', 'Jeffrey, S. J., Carter, J. O., Moodie, K. B., & Beswick, A. R. (2001). Using spatial interpolation to construct a comprehensive archive of Australian climate data. Environmental Modelling & Software, 16(4), 309–330. https://doi.org/10.1016/S1364-8152(01)00008-1', 'North Australian Fire Information. (2026). Northern Territory fire scar mapping [Data set]. Darwin Centre for Bushfire Research, Charles Darwin University. https://firenorth.org.au', 'Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., … Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825–2830.', 'Queensland Government. (2026). SILO climate data: Data Drill [Data set]. Long Paddock. https://www.longpaddock.qld.gov.au/silo/', 'Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. In N. Navab, J. Hornegger, W. M. Wells, & A. F. Frangi (Eds.), Medical image computing and computer-assisted intervention – MICCAI 2015 (pp. 234–241). Springer. https://doi.org/10.1007/978-3-319-24574-4_28', "Russell-Smith, J., Yates, C. P., Whitehead, P. J., Smith, R., Craig, R., Allan, G. E., Thackway, R., Frakes, I., Cridland, S., Meyer, M. C. P., & Gill, A. M. (2007). Bushfires 'down under': Patterns and implications of contemporary Australian landscape burning. International Journal of Wildland Fire, 16(4), 361–377. https://doi.org/10.1071/WF07018"]


def refs_slide(part, entries, n, total):
    s = slide(notes="Reference slide. Nothing to say -- leave it up during questions.")
    kicker(s, "References")
    title(s, "References" + (f" ({part} of {total})" if total > 1 else ""))
    text(s, 0.9, 1.72, 11.5, 0.32,
         "American Psychological Association, 7th edition", size=11.5, color=MUTED)
    text(s, 0.9, 2.22, 11.5, 4.4, list(entries), size=12.2, color=INK,
         spacing=1.32, space_after=12, hanging=True)
    footer(s, "", n)


def s23_refs():
    refs_slide(1, REFERENCES[:8], 23, 2)


def s24_refs():
    refs_slide(2, REFERENCES[8:], 24, 2)


def main():
    for fn in (s01_title, s02_burns, s03_when, s04_target, s05_data,
               s06_divider_build, s07_labels, s08_features, s09_leakage, s10_split,
               s11_models, s12_divider_results, s13_results, s14_layers, s15_walk,
               s16_experiments, s17_importance, s18_limits,
               s19_divider_2026, s20_forecast, s21_product, s22_close,
               s23_refs, s24_refs):
        fn()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT)
    print(f"wrote {OUT.relative_to(ROOT)}  ({len(prs.slides.__iter__.__self__._sldIdLst)} slides)")


if __name__ == "__main__":
    main()
