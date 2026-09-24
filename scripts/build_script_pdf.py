"""Build the spoken script for the Assessment 3 presentation.

The script is written to be read aloud, so it is set in a single wide column at
a size that survives a lectern, with stage directions kept visually separate
from the words themselves -- a presenter glancing down needs to find their line
again in a fraction of a second, not parse a paragraph.

Timings assume roughly 140 words a minute. The total is held under fifteen
minutes and no single speaker goes past six.
"""

from __future__ import annotations

import pathlib

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, PageTemplate, Paragraph,
                                Spacer, Table, TableStyle, KeepTogether, PageBreak)

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "PYRANTIS_presentation_script.pdf"
DF = pathlib.Path("/Applications/Microsoft PowerPoint.app/Contents/Resources/DFonts")

INK    = colors.HexColor("#0F2A35")
BLUE   = colors.HexColor("#245F80")
EMBER  = colors.HexColor("#C8552F")
GREEN  = colors.HexColor("#2E7D5B")
SLATE  = colors.HexColor("#9AA8AE")
MUTED  = colors.HexColor("#6B7C85")
PAPER  = colors.HexColor("#F7F5F2")
RULE   = colors.HexColor("#DCD8D2")

# Fall back to the built-ins if Office is not installed on the machine building
# this; the layout is identical, only the letterforms change.
BODY, BODY_B, DISP, DISP_B = "Helvetica", "Helvetica-Bold", "Times-Roman", "Times-Bold"
try:
    pdfmetrics.registerFont(TTFont("Calibri", str(DF / "Calibri.ttf")))
    pdfmetrics.registerFont(TTFont("Calibri-B", str(DF / "Calibrib.ttf")))
    pdfmetrics.registerFont(TTFont("Calibri-I", str(DF / "Calibrii.ttf")))
    pdfmetrics.registerFont(TTFont("Cambria", str(DF / "Cambria.ttc"), subfontIndex=0))
    pdfmetrics.registerFont(TTFont("Cambria-B", str(DF / "Cambriab.ttf")))
    BODY, BODY_B, DISP, DISP_B = "Calibri", "Calibri-B", "Cambria", "Cambria-B"
    ITAL = "Calibri-I"
except Exception as e:                                     # pragma: no cover
    print(f"  (Office fonts unavailable, using built-ins: {e})")
    ITAL = "Helvetica-Oblique"

S = {
    "cover_title": ParagraphStyle("ct", fontName=DISP_B, fontSize=34, leading=38,
                                  textColor=INK, spaceAfter=6),
    "cover_sub":   ParagraphStyle("cs", fontName=DISP, fontSize=15, leading=21,
                                  textColor=BLUE, spaceAfter=18),
    "cover_meta":  ParagraphStyle("cm", fontName=BODY, fontSize=10.5, leading=17,
                                  textColor=MUTED),
    "h1":          ParagraphStyle("h1", fontName=DISP_B, fontSize=19, leading=23,
                                  textColor=INK, spaceBefore=10, spaceAfter=3),
    "h1sub":       ParagraphStyle("h1s", fontName=BODY, fontSize=10, leading=14,
                                  textColor=EMBER, spaceAfter=12),
    "slug":        ParagraphStyle("sl", fontName=BODY_B, fontSize=9.5, leading=13,
                                  textColor=BLUE, spaceAfter=2),
    "say":         ParagraphStyle("sy", fontName=BODY, fontSize=11.6, leading=17.6,
                                  textColor=INK, spaceAfter=7, alignment=TA_LEFT),
    "dir":         ParagraphStyle("dr", fontName=ITAL, fontSize=9.8, leading=14,
                                  textColor=EMBER, spaceAfter=6, leftIndent=0),
    "note":        ParagraphStyle("nt", fontName=BODY, fontSize=10, leading=15,
                                  textColor=MUTED),
    "qa_q":        ParagraphStyle("qq", fontName=BODY_B, fontSize=10.6, leading=15,
                                  textColor=INK, spaceAfter=2),
    "qa_a":        ParagraphStyle("qa", fontName=BODY, fontSize=10.6, leading=15.5,
                                  textColor=INK, spaceAfter=10),
}

HARSH = "Harsh Rastogi"
SAIRA = "Saira Zafar"
THARU = "Tharushi Wimalachandra"

# (slide, speaker, cue, stage direction, [paragraphs to say])
SCRIPT = [
 (1, HARSH, "0:00 – 0:30", "Title slide. Stand still, let the room settle, then start.", [
  "Good morning. Every year, most of the Northern Territory burns. That is not a "
  "disaster — that is the season. What decides whether a fire is a management tool or "
  "a catastrophe is not whether it happens. It is when.",
  "We are Group 85 — I am Harsh Rastogi, with Saira Zafar and Tharushi Wimalachandra. "
  "Our project is Pyrantis. For every five-kilometre square of the Territory, for every "
  "year since 2000, we classify what happened — no fire, a fire before August, or a "
  "fire after July — and then we forecast 2026."]),

 (2, HARSH, "0:30 – 1:05", "Point at the grey band first, then the two coloured ones.", [
  "Across 2000 to 2025, roughly a third of the Territory carried a fire in any given "
  "year — one-point-two million square-years of record, across forty-six thousand map "
  "squares.",
  "Look at the split. Sixty-nine per cent of those square-years saw no fire. Fourteen "
  "per cent burned before August. Seventeen per cent burned after July. Hold on to that "
  "imbalance, because the rarest answer is the one we care about most."]),

 (3, HARSH, "1:05 – 1:45", "Two cards, then the chart. Slow down on the last sentence.", [
  "Before August the country is still damp. Fires are cool and they leave a patchwork "
  "— rangers light these deliberately to break up the fuel. After July the grass has "
  "cured, and fires run with the wind, hot enough to kill canopy and take out fences, "
  "stock and country in an afternoon.",
  "And it compounds. In the Top End, country burned early carries a late fire the next "
  "year about twenty per cent of the time, against nearly thirty-two for country left "
  "alone. Territory-wide that reverses, because the arid south runs on a different "
  "clock — which is why we predict square by square, not region by region."]),

 (4, HARSH, "1:45 – 2:05", "Read the three classes off the slide, unhurried.", [
  "So the question we set is a three-class problem. For every square, for every year: "
  "unburnt, early, or late.",
  "Early means first detected on or before the thirty-first of July; late means after "
  "it. That date is not ours — it is the line the Territory\u2019s own fire management "
  "uses, so the classes mean something to the people who would act on them."]),

 (5, HARSH, "2:05 – 2:45", "Three sources, left to right. Then hand over, and step back.", [
  "NAFI gives us 250-metre burnt-area rasters back to 2000, and each pixel carries the "
  "month it was detected — which is what makes an early-or-late label possible at all. "
  "SILO gives us daily weather on the same grid. And MODIS gives us satellite "
  "greenness: how much fuel is standing when the dry arrives.",
  "All three are open data, cited in the report. Nothing here sits behind a login. That "
  "leaves fifty features across nine hundred and sixty-six thousand rows.",
  "Saira will take you through how we turned that into something a model can learn "
  "from."]),

 (6, SAIRA, "", "Divider. Say this with the slide up, then advance.", [
  "Thanks Harsh. This part is how we built it, and how we proved the model never saw "
  "the answer."]),

 (7, SAIRA, "", "Start on the numbered steps.", [
  "The first job was labelling.",
  "Each NAFI pixel holds the month it burned. We pool every twenty-by-twenty block into "
  "one cell and count what fraction burned early and what fraction late. A square counts "
  "as burnt only if five per cent of it burned, and is used at all only if half its "
  "pixels were readable. Whichever season holds more of the burnt area names the year.",
  "Labels are the one thing no metric can catch: if they are wrong, every number after "
  "them is wrong and nothing tells you. So we checked ours against fire history "
  "the Territory already knows. 2011, after the big La Ni\u00f1a, comes out heaviest on "
  "record at nearly sixty-six per cent burnt; 2020 lightest, under sixteen. Both match "
  "the published record."]),

 (8, SAIRA, "3:30 – 4:05", "Land hard on 30 April — it is the honesty rule.", [
  "A forecast is only useful if it arrives before the fire season. So every one of our "
  "fifty features is built from information available by the thirtieth of April in the "
  "year being predicted — and nothing after.",
  "Four groups. Twenty-six fire-history features: years since the last fire, how often "
  "a square burnt over three to twenty-five years, what it did last year, and what its "
  "neighbours did. Fourteen weather features from the wet season just gone. Seven "
  "greenness features from the satellite. And three that simply place the square: "
  "longitude, latitude and elevation."]),

 (9, SAIRA, "4:05 – 4:50", "The most important slide in the middle act. Do not rush it.", [
  "This is the slide I would ask you to weigh most heavily.",
  "A feature that quietly reads the year it is meant to predict will score beautifully "
  "and mean nothing. We did not trust ourselves to avoid that by being careful. We "
  "tested for it twice.",
  "Rule one, recomputation. Every fire-history feature is rebuilt from years strictly "
  "before the target year, inside one function that both the training and forecast paths "
  "call. Written twice, the two copies would have drifted — and silently, because a "
  "forecast has no label to check itself against.",
  "Rule two, scrambling. We shuffled one year\u2019s labels and rebuilt the features "
  "from scratch. If anything had been reading the answer, the scores would have "
  "collapsed. They moved by exactly as much as chance says they should.",
  "Both checks are scripts in the repository."]),

 (10, SAIRA, "4:50 – 5:20", "Point along the three bands, left to right.", [
  "Fire years are not independent draws. A random shuffle would put 2024 into training "
  "and 2023 into test, and the model would be reading its own future. So we cut the "
  "record chronologically: train on 2005 to 2019, tune on 2020 to 2022, and hold out "
  "2023 to 2025 entirely.",
  "And once was once. Those three years were scored a single time, at the end, after "
  "every decision about features and models had been made."]),

 (11, SAIRA, "5:20 – 6:00", "Read 0.420 and 0.553 off the card. Then hand over.", [
  "Finally, the models. The five the unit requires — decision tree, random forest, naive "
  "Bayes, a multilayer network, and an LSTM reading the twenty-six-year sequence — plus "
  "gradient boosting, which won. And two baselines: whatever happened last year, and "
  "always say unburnt. A model that cannot beat those has learned nothing.",
  "One correction mattered enormously. Late fires are the rarest thing on the map and "
  "the thing we care about most. Left unweighted, every model quietly stopped predicting "
  "them — it is cheaper to be right about the common answer. Weighting makes that trade "
  "expensive. For gradient boosting the score on late fires goes from 0.42 to 0.55. Same "
  "model, same data, same split.",
  "Tharushi will tell you whether any of this actually works."]),

 (12, THARU, "", "Divider. Say this with the slide up, then advance.", [
  "Thanks Saira. This part tests all of it on years the models never saw."]),

 (13, THARU, "", "Point at the two grey baseline bars, not the top bar.", [
  "These are the scores on 2023 to 2025 — years none of these models had "
  "ever seen.",
  "Gradient boosting wins, with a balanced F1 of 0.684 and accuracy of 0.725. The LSTM "
  "is just behind at 0.673, the network at 0.667, the random forest at 0.660.",
  "But the number to watch is not the top one. It is the gap. Same as last year scores "
  "0.506 — what a fire officer already knows without us — and gradient boosting adds "
  "0.178 on top. Majority class scores 0.244 — how badly a model can do while still "
  "looking accurate, by refusing to name the rare answers."]),

 (14, THARU, "6:40 – 7:15", "Be plain about the small gain. It reads as honesty.", [
  "Fire history alone gets to 0.639. Adding weather takes it to 0.674 — plus 0.035, and "
  "the single biggest step in the project. The third run adds greenness, longer fire "
  "memory and elevation together, and takes it to 0.684.",
  "Isolated in the ledger, greenness on its own is worth plus 0.029. We kept it, because "
  "that holds up on years the model never saw and two of its five strongest inputs are "
  "greenness measures — though it cost ten minutes a run against sixty seconds for "
  "everything else."]),

 (15, THARU, "7:15 – 7:55", "Trace the ember line with a finger. Stay on the last point.", [
  "One good test year could be luck. So we did it sixty-six times.",
  "We cut the record at each year from 2014 to 2024, let the model learn only what came "
  "before the cut, and scored it on every year after. Sixty-six separate tests.",
  "The surprise is that skill does not fade with distance. One year ahead it scores "
  "0.635; eleven years ahead it still scores 0.604, with the baseline well below "
  "throughout.",
  "That tells us the strongest clue is the wet season just gone, which is known by the "
  "end of April."]),

 (16, THARU, "7:55 – 8:40", "Say the CNN result without hedging. Markers reward this.", [
  "Behind that final model sit twenty-three experiments, each scored against the same "
  "fixed benchmark, each recorded as kept or reverted.",
  "Exactly one was kept — adding satellite greenness.",
  "And I want to be explicit about the one that failed. We built a U-Net — a "
  "convolutional network reading fire as a picture rather than a table of numbers, the "
  "architecture you would reach for if spatial pattern were the whole story. It scored "
  "0.588, which is 0.048 below the benchmark, so it went into the ledger as a reverted "
  "run and stayed out of the final model.",
  "We are reporting it because a ledger that only contains successes is not a ledger."]),

 (17, THARU, "8:40 – 9:10", "Read the top three bars. Land on the closing sentence.", [
  "Nobody told it that a wet, hot wet season grows the grass that carries an October "
  "fire. It found that in the data. Its strongest inputs are the dryness of the air "
  "through the wet, average greenness, wet-season rainfall, and what the neighbouring "
  "country did last year.",
  "That it lands where fire ecology already sits is the best evidence we have that it "
  "learned the problem rather than our dataset."]),

 (18, THARU, "9:10 – 9:50", "Four limits, briskly. Then hand back to Harsh.", [
  "Late fires are still our worst class, at 0.553. The one that matters most is also the "
  "rarest and the most weather-driven — and weather after April is precisely what we "
  "refuse to look at.",
  "It predicts a season, not a day, and it is not a fire-danger rating.",
  "The 2026 forecast is unverified, because no burnt-area record exists for a year still "
  "running. And it is fitted to the Territory at this grid size; whether it transfers to "
  "the Kimberley is an open question, not an assumption.",
  "Harsh will close with what it says about this year."]),

 (19, HARSH, "", "Divider. Say this with the slide up, then advance.", [
  "Thanks Tharushi. The last part is the 2026 forecast, and the site we built for it."]),

 (20, HARSH, "9:50 – 10:35", "Say fifty-one per cent slowly. It is the headline.", [
  "So, 2026.",
  "The model expects fifty-one per cent of the Territory to burn this year, against a "
  "twenty-six-year average of thirty-three. Fifteen per cent before August, thirty-six "
  "after July. A heavy year, and most of what is left will burn late.",
  "It reads that out of a wet season that delivered 783 millimetres — forty-five per cent "
  "above the 2005 to 2019 average, and the second wettest in twenty-two years — together "
  "with the greenest April on record. More water grew more grass. More grass is more "
  "fuel.",
  "Mean confidence across the map is 0.733. And because no fire scars exist for 2026 "
  "yet, this is not a result — it is a claim we can be held to in 2027."]),

 (21, HARSH, "10:35 – 11:10", "If live, click one town. If not, talk over the screenshot.", [
  "So we shipped it.",
  "This is a public site. It answers the question in plain English first, then puts the "
  "map, the twenty-six-year record and the evidence underneath. You can click any square, "
  "or pick any of forty-one named towns.",
  "Every sentence on it is computed from the model output, so it updates when the data "
  "does."]),

 (22, HARSH, "11:10 – 11:50", "Slow down on the last two sentences, then advance.", [
  "To close.",
  "Every square, every year, and an honest number on how often it is right. A balanced F1 "
  "of 0.684 on years the models never saw. Sixty-six independent re-tests behind that "
  "figure. And a forecast of fifty-one per cent for 2026 that can actually be checked.",
  "Fire in the Territory is not going to stop, and it should not. Knowing in April which "
  "country is likely to carry an October fire is how a burning programme gets planned "
  "in time.",
  "Everything — the code, the figures in this deck, and the site — is in the repository, "
  "and every source we used is referenced on the last two slides."]),

 (23, HARSH, "", "Advance to the references, say this, and leave the slide up for questions.", [
  "Thank you. We are happy to take questions."]),

 (24, HARSH, "", "No speech. Advance only if a marker asks to see the rest of the "
  "reference list.", []),
]

QA = [
 ("Why gradient boosting rather than the LSTM, when they are so close?",
  "On the held-out years gradient boosting is ahead, 0.684 to 0.673, and it trains in "
  "about ninety-five seconds against a much heavier fit for the network. It also held up "
  "under a systematic search: tuning left its test score at exactly 0.684."),
 ("How do you know there is no leakage?",
  "Two checks. Every fire-history feature is recomputed from years strictly before the "
  "target year through a single shared function, so the training and forecast paths "
  "cannot drift apart. And we scrambled one year's labels and rebuilt the features — the "
  "scores moved by exactly as much as chance predicts. Both are scripts in the repo."),
 ("Is 0.684 actually good?",
  "On its own the number means little. What matters is the comparison: the naive "
  "'same as last year' rule scores 0.506 on identical data, and always guessing unburnt "
  "scores 0.244. We are 0.178 above the rule a fire officer already has."),
 ("Why is the late class so much worse than the others?",
  "It is the rarest class, and it is the most weather-dependent — a late fire is driven "
  "by conditions in September and October, which is information we deliberately refuse "
  "to use because it would not exist at forecast time. Class weighting lifted it from "
  "0.42 to 0.55, but it remains the honest weak point."),
 ("Why did the CNN fail?",
  "Our best guess is that the signal is mostly per-cell rather than spatial: the wet "
  "season, the greenness and the local fire history explain most of it, and those are "
  "already columns in a table. The U-Net spent its capacity on spatial texture that "
  "carried comparatively little information, and at 0.588 it sat well below the "
  "benchmark."),
 ("Why 0.05 degrees, and not finer or coarser?",
  "It matches the SILO weather grid, so weather joins to fire without resampling. We "
  "also checked that the per-cell detail is real rather than noise: neighbouring cells "
  "disagree about twelve per cent of the time, and that disagreement is driven by fire "
  "history and greenness, not by weather — so coarsening would throw away genuine "
  "signal."),
 ("What would you do with more time?",
  "Three things. Bring in a seasonal climate outlook so the model has some view past "
  "April. Test whether it transfers to the Kimberley. And go back to the spatial idea "
  "with a model that keeps the tabular features and adds a spatial branch, rather than "
  "replacing them the way the U-Net did."),
 ("What is cross-validation, and why not use ordinary k-fold?",
  "Cross-validation trains on part of the data and tests on the part left out, in turn. "
  "Shuffled k-fold leaks here, because a square looks like its neighbour and like itself the "
  "next year. We measured it: repeated stratified 5-fold scored 0.772, while the same "
  "model on unseen years scored 0.677. So we used 66 walk-forward tests "
  "instead, which always train on the past and test on the future."),
 ("How do you know it is not overfitting?",
  "Three checks. Validation and test scores agree within 0.04 for every model. Gradient "
  "boosting scores 0.821 on its training years "
  "against 0.684 on the test years, and the neural "
  "networks stop training at the best validation epoch. And across 66 walk-forward tests it "
  "beat the same-as-last-year baseline every time."),
 ("Besides accuracy, what scores did you use?",
  "Macro precision, recall and F1, F1 on each class, the confusion matrix, and ROC-AUC and "
  "PR-AUC. Gradient boosting has a ROC-AUC of 0.862 "
  "and a PR-AUC on late fires of 0.558, about "
  "twice what guessing would score."),
 ("Were the settings tuned, or picked by hand?",
  "Both. They were first picked by hand, then every model was tuned with a randomised or grid "
  "search scored on the validation years only. Gradient boosting stayed at 0.684; the decision "
  "tree gained most, from 0.604 to 0.632."),
 ("Who did what?",
  "Harsh led the project: the problem framing, the data pipeline, the forecast and the "
  "web product. Saira led labelling, feature engineering and the leakage and validation "
  "design. Tharushi led model training, the experiment ledger, walk-forward testing and "
  "the results analysis. All three of us reviewed each other's work."),
]

SPEAKERS = [
 (HARSH, "Slides 1–5 and 19–24", "5 min 30 s", "Opening, the problem, the data, the "
  "2026 forecast, the product and the close."),
 (SAIRA, "Slides 6–11", "4 min 30 s", "Labelling, features, the leakage rules, the "
  "temporal split and the models."),
 (THARU, "Slides 12–18", "4 min 30 s", "Results, the data-layer gains, walk-forward "
  "testing, the experiment ledger and the limits."),
]


WPM = 140.0
SLIDE_GAP = 2.0            # seconds to change slide and draw breath
HANDOVER = 5.0             # seconds for a speaker change


def cues():
    """Walk the script once and return {slide: "m:ss - m:ss"} plus the total."""
    out, t, prev = {}, 0.0, None
    for n, who, _, _, paras in SCRIPT:
        if prev is not None and who != prev:
            t += HANDOVER
        prev = who
        words = len(" ".join(paras).split())
        dur = words / WPM * 60.0
        out[n] = (t, t + dur)
        t += dur + SLIDE_GAP
    return out, t


def clock(sec):
    return f"{int(sec // 60)}:{int(sec % 60):02d}"



def spoken_for(who):
    return sum(len(" ".join(p).split()) for _, w, _, _, p in SCRIPT if w == who) / WPM * 60.0


def header_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont(BODY, 8.5)
    canvas.setFillColor(MUTED)
    if doc.page > 1:
        canvas.drawString(20 * mm, 12 * mm, "PYRANTIS — presentation script — Group 85")
        canvas.drawRightString(A4[0] - 20 * mm, 12 * mm, str(doc.page))
    canvas.restoreState()


def three_squares(story, size=7):
    t = Table([[""] * 3], colWidths=[size + 3] * 3, rowHeights=[size])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, 0), SLATE),
        ("BACKGROUND", (1, 0), (1, 0), GREEN),
        ("BACKGROUND", (2, 0), (2, 0), EMBER),
        ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    story.append(t)


def build():
    doc = BaseDocTemplate(str(OUT), pagesize=A4,
                          leftMargin=20 * mm, rightMargin=20 * mm,
                          topMargin=18 * mm, bottomMargin=20 * mm,
                          title="PYRANTIS — presentation script",
                          author="Group 85 — Harsh Rastogi, Saira Zafar, Tharushi Wimalachandra")
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="f")
    doc.addPageTemplates([PageTemplate(id="p", frames=[frame], onPage=header_footer)])

    st = []
    _, TOTAL = cues()
    # ---- cover
    st.append(Spacer(1, 34 * mm))
    three_squares(st, 9)
    st.append(Spacer(1, 7 * mm))
    st.append(Paragraph("PYRANTIS", S["cover_title"]))
    st.append(Paragraph("Presentation script — which country burns in the Northern "
                        "Territory, and when", S["cover_sub"]))
    st.append(Spacer(1, 4 * mm))
    rows = [["Unit", "PRT565 Machine Learning, Artificial Intelligence and Algorithms"],
            ["Assessment", "Assessment 3 — presentation"],
            ["Group", "Group 85"],
            ["Presenters", "Harsh Rastogi (386401), Saira Zafar (407193), "
                           "Tharushi Wimalachandra (387594)"],
            ["Slides", "24, every one covered below. Slide 24 is the second half of the "
                       "reference list and has no speech."],
            ["Total running time", f"About {TOTAL/60:.1f} minutes including slide "
                                   f"changes, inside the 15 minute limit"],
            ["Live site", "pyrantis.harshlabs.workers.dev"]]
    t = Table(rows, colWidths=[34 * mm, doc.width - 34 * mm])
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), BODY_B), ("FONTNAME", (1, 0), (1, -1), BODY),
        ("FONTSIZE", (0, 0), (-1, -1), 10), ("TEXTCOLOR", (0, 0), (0, -1), MUTED),
        ("TEXTCOLOR", (1, 0), (1, -1), INK), ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LINEBELOW", (0, 0), (-1, -2), 0.5, RULE),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
    ]))
    st.append(t)
    st.append(Spacer(1, 10 * mm))
    st.append(Paragraph("How the time is divided", S["h1"]))
    st.append(Spacer(1, 2 * mm))
    CUE, TOTAL = cues()
    spoken = {}
    for n, who, _, _, paras in SCRIPT:
        spoken[who] = spoken.get(who, 0.0) + len(" ".join(paras).split()) / WPM * 60.0
    cell = ParagraphStyle("cell", fontName=BODY, fontSize=9.4, leading=12.6,
                          textColor=INK)
    head = ParagraphStyle("hd", fontName=BODY_B, fontSize=9.4, leading=12.6,
                          textColor=colors.white)
    name = ParagraphStyle("nm", fontName=BODY_B, fontSize=9.4, leading=12.6,
                          textColor=BLUE)
    rows = [[Paragraph(h, head) for h in ("Speaker", "Slides", "Time", "Covers")]] + [
        [Paragraph(w, name), Paragraph(sl, cell),
         Paragraph(f"{spoken[w]/60:.1f} min", cell), Paragraph(cv, cell)]
        for w, sl, _t, cv in SPEAKERS]
    t = Table(rows, colWidths=[34 * mm, 28 * mm, 17 * mm, doc.width - 79 * mm])
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, 0), BODY_B), ("FONTNAME", (0, 1), (-1, -1), BODY),
        ("FONTNAME", (0, 1), (0, -1), BODY_B),
        ("FONTSIZE", (0, 0), (-1, -1), 9.6), ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BACKGROUND", (0, 0), (-1, 0), INK),
        ("TEXTCOLOR", (0, 1), (0, -1), BLUE),
        ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 7), ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("LINEBELOW", (0, 1), (-1, -2), 0.5, RULE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [PAPER, colors.white]),
    ]))
    st.append(t)
    st.append(Spacer(1, 7 * mm))
    st.append(Paragraph(
        "Slide 1 is the cover — add the unlisted recording link before you present. "
        "Stage directions are set in orange italics and are not to be read aloud. "
        "Cues are a guide, not a stopwatch — if a section runs long, cut the third "
        "paragraph of slide 3 and the last limitation on slide 18 first.", S["note"]))
    st.append(PageBreak())

    # ---- the script
    current = None
    CUE, TOTAL = cues()
    for n, who, _cue, direction, paras in SCRIPT:
        a, b = CUE[n]
        # a slide with nothing to say gets a single time rather than "14:33 – 14:33"
        cue = clock(a) if not paras else f"{clock(a)} – {clock(b)}"
        block = []
        if who != current:
            block.append(Paragraph(who, S["h1"]))
            role = next(x for x in SPEAKERS if x[0] == who)
            mins = spoken_for(who) / 60.0
            block.append(Paragraph(f"{role[1]} · {mins:.1f} min", S["h1sub"]))
            current = who
        block.append(Paragraph(f"SLIDE {n} · {cue}", S["slug"]))
        block.append(Paragraph(direction, S["dir"]))
        for p in paras:
            block.append(Paragraph(p, S["say"]))
        block.append(Spacer(1, 3 * mm))
        st.append(KeepTogether(block) if len(paras) <= 3 else block[0])
        if len(paras) > 3:
            for b in block[1:]:
                st.append(b)

    # ---- Q&A
    st.append(PageBreak())
    st.append(Paragraph("Questions we expect", S["h1"]))
    st.append(Paragraph("Answer in one or two sentences, then stop.", S["h1sub"]))
    for q, a in QA:
        st.append(KeepTogether([Paragraph(q, S["qa_q"]), Paragraph(a, S["qa_a"])]))

    doc.build(st)
    words = sum(len(" ".join(p).split()) for _, _, _, _, p in SCRIPT)
    print(f"wrote {OUT.relative_to(ROOT)}")
    print(f"  {words:,} spoken words ≈ {words/140:.1f} min at 140 wpm")
    for who, slides, t_, _ in SPEAKERS:
        w = sum(len(" ".join(p).split()) for _, wh, _, _, p in SCRIPT if wh == who)
        print(f"  {who:24s} {w:4d} words ≈ {w/140:4.1f} min   (planned {t_})")


if __name__ == "__main__":
    build()
