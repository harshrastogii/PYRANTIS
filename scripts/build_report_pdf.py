"""Build the Assessment 3 report.

The cover page and the second page are laid out to the letter of the assessment
brief, because an incomplete cover page costs 15% and the brief names exactly
what has to be on each: unit information, group number, member information and
campus on page one; the recording link, project title, description and
motivation on page two.

Typography and colour are lifted from the redesigned deck so the report and the
slides read as one submission, and the figures are the deck's own images rather
than redrawn copies that could drift from it.
"""

from __future__ import annotations

import json
import pathlib

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, PageTemplate, Paragraph,
                                Spacer, Table, TableStyle, KeepTogether, PageBreak,
                                Image as RLImage)

ROOT = pathlib.Path(__file__).resolve().parents[1]
R = ROOT / "reports"
ASSETS = R / "report_assets"
OUT = R / "PRT565_A3_Report_Group85.pdf"

SYS = pathlib.Path("/System/Library/Fonts/Supplemental")
DF = pathlib.Path("/Applications/Microsoft PowerPoint.app/Contents/Resources/DFonts")

# palette taken from the redesigned deck
PAPER = colors.HexColor("#FCFBF9")
INK   = colors.HexColor("#16130F")
BODY_C= colors.HexColor("#3A342E")
MUTED = colors.HexColor("#787066")
DIM   = colors.HexColor("#A79D91")
RED   = colors.HexColor("#C8341B")
GREEN = colors.HexColor("#1E7A50")
BLUE  = colors.HexColor("#2E6E96")
SLATE = colors.HexColor("#8E9AA1")
TINT  = colors.HexColor("#F4F1EB")
RULE  = colors.HexColor("#DCD5CA")

SERIF, SERIF_B = "Helvetica", "Helvetica-Bold"
SANS, SANS_B, SANS_I = "Helvetica", "Helvetica-Bold", "Helvetica-Oblique"
try:
    pdfmetrics.registerFont(TTFont("Georgia", str(SYS / "Georgia.ttf")))
    pdfmetrics.registerFont(TTFont("Georgia-B", str(SYS / "Georgia Bold.ttf")))
    pdfmetrics.registerFont(TTFont("Georgia-I", str(SYS / "Georgia Italic.ttf")))
    SERIF, SERIF_B = "Georgia", "Georgia-B"
    pdfmetrics.registerFont(TTFont("FGBook", str(DF / "Franklin Gothic Book.ttf")))
    pdfmetrics.registerFont(TTFont("FGMed", str(DF / "Franklin Gothic Medium.ttf")))
    pdfmetrics.registerFont(TTFont("FGBookI", str(DF / "Franklin Gothic Book Italic.ttf")))
    SANS, SANS_B, SANS_I = "FGBook", "FGMed", "FGBookI"
except Exception as e:                                       # pragma: no cover
    print(f"  (falling back to built-in fonts: {e})")

S = {
 "cover_uni":  ParagraphStyle("cu", fontName=SANS_B, fontSize=11, leading=15,
                              textColor=MUTED, spaceAfter=2),
 "cover_title":ParagraphStyle("ct", fontName=SERIF_B, fontSize=40, leading=44,
                              textColor=INK, spaceAfter=4),
 "cover_sub":  ParagraphStyle("cs", fontName=SERIF, fontSize=15.5, leading=21,
                              textColor=BODY_C, spaceAfter=6),
 "h1":         ParagraphStyle("h1", fontName=SERIF_B, fontSize=17, leading=22,
                              textColor=INK, spaceBefore=4, spaceAfter=7),
 "kicker":     ParagraphStyle("kk", fontName=SANS_B, fontSize=8.6, leading=12,
                              textColor=RED, spaceAfter=3),
 "h2":         ParagraphStyle("h2", fontName=SERIF_B, fontSize=12.4, leading=17,
                              textColor=INK, spaceBefore=9, spaceAfter=4),
 "body":       ParagraphStyle("bd", fontName=SANS, fontSize=10.3, leading=15.6,
                              textColor=BODY_C, spaceAfter=7, alignment=TA_LEFT),
 "lead":       ParagraphStyle("ld", fontName=SANS, fontSize=11.4, leading=17.4,
                              textColor=INK, spaceAfter=8),
 "cap":        ParagraphStyle("cp", fontName=SANS_I, fontSize=8.8, leading=12.4,
                              textColor=MUTED, spaceBefore=3, spaceAfter=10),
 "ref":        ParagraphStyle("rf", fontName=SANS, fontSize=9.3, leading=13.0,
                              textColor=BODY_C, spaceAfter=6,
                              leftIndent=9*mm, firstLineIndent=-9*mm),
 "note":       ParagraphStyle("nt", fontName=SANS, fontSize=9.4, leading=13.4,
                              textColor=MUTED, spaceAfter=6),
}

j = lambda n: json.loads((R / n).read_text())
full = j("results_full.json")
fc = j("forecast_2026.json")
res = {r["model"]: r for r in full["results"]}


def fig(name, width_mm, caption):
    p = ASSETS / name
    from PIL import Image as PILImage
    im = PILImage.open(p)
    w = width_mm * mm
    h = w * im.height / im.width
    return [RLImage(str(p), width=w, height=h), Paragraph(caption, S["cap"])]


def rule(h=1.1, col=RED, width=46*mm):
    t = Table([[""]], colWidths=[width], rowHeights=[h])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), col),
                           ("LEFTPADDING", (0, 0), (-1, -1), 0),
                           ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                           ("TOPPADDING", (0, 0), (-1, -1), 0),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 0)]))
    t.hAlign = "LEFT"
    return t


def section(story, kicker, title):
    story.append(rule())
    story.append(Spacer(1, 4*mm))
    story.append(Paragraph(kicker.upper(), S["kicker"]))
    story.append(Paragraph(title, S["h1"]))


def page_bg(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(PAPER)
    canvas.rect(0, 0, A4[0], A4[1], stroke=0, fill=1)
    if doc.page > 1:
        canvas.setFont(SANS, 8.2)
        canvas.setFillColor(DIM)
        canvas.drawString(20*mm, 12*mm, "PRT565 Assessment 3   ·   Group 85   ·   PYRANTIS")
        canvas.drawRightString(A4[0] - 20*mm, 12*mm, str(doc.page))
        canvas.setStrokeColor(RULE); canvas.setLineWidth(0.5)
        canvas.line(20*mm, 16*mm, A4[0] - 20*mm, 16*mm)
    canvas.restoreState()


def kv_table(rows, doc_width, label_w=40*mm):
    lab = ParagraphStyle("lab", fontName=SANS_B, fontSize=9.2, leading=13, textColor=MUTED)
    val = ParagraphStyle("val", fontName=SANS, fontSize=10.2, leading=14.4, textColor=INK)
    data = [[Paragraph(k, lab), Paragraph(v, val)] for k, v in rows]
    t = Table(data, colWidths=[label_w, doc_width - label_w])
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("LINEBELOW", (0, 0), (-1, -2), 0.5, RULE),
    ]))
    return t


def metrics_table(doc_width):
    order = ["gradient boosting (balanced)", "LSTM (balanced)", "multilayer ANN (balanced)",
             "random forest (balanced)", "decision tree (balanced)", "naive bayes (balanced)",
             "same as last year", "majority class"]
    nice = {"gradient boosting (balanced)": "Gradient boosting", "LSTM (balanced)": "LSTM",
            "multilayer ANN (balanced)": "Multilayer ANN", "random forest (balanced)": "Random forest",
            "decision tree (balanced)": "Decision tree", "naive bayes (balanced)": "Naive Bayes",
            "same as last year": "Baseline: same as last year",
            "majority class": "Baseline: always unburnt"}
    hd = ParagraphStyle("hd", fontName=SANS_B, fontSize=8.8, leading=12, textColor=colors.white)
    nm = ParagraphStyle("nm", fontName=SANS_B, fontSize=9.2, leading=12.6, textColor=INK)
    nb = ParagraphStyle("nb", fontName=SANS, fontSize=9.2, leading=12.6, textColor=MUTED)
    vl = ParagraphStyle("vl", fontName=SANS, fontSize=9.2, leading=12.6, textColor=BODY_C)
    data = [[Paragraph(h, hd) for h in
             ("Model", "Accuracy", "Precision", "Recall", "F1", "F1 on late fires")]]
    for k in order:
        r = res[k]
        style = nb if k.startswith(("same as", "majority")) else nm
        data.append([Paragraph(nice[k], style),
                     Paragraph(f"{r['accuracy']:.3f}", vl),
                     Paragraph(f"{r['precision_macro']:.3f}", vl),
                     Paragraph(f"{r['recall_macro']:.3f}", vl),
                     Paragraph(f"{r['f1_macro']:.3f}", vl),
                     Paragraph(f"{r['f1_per_class']['late']:.3f}", vl)])
    w = doc_width
    t = Table(data, colWidths=[w*0.34, w*0.13, w*0.13, w*0.12, w*0.11, w*0.17])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), INK),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, 1), (-1, -2), 0.5, RULE),
        ("BACKGROUND", (0, 1), (-1, 1), TINT),
        ("LINEABOVE", (0, 7), (-1, 7), 0.9, RULE),
    ]))
    return t


REFERENCES = [
 "Breiman, L. (2001). Random forests. <i>Machine Learning, 45</i>(1), 5–32. "
 "https://doi.org/10.1023/A:1010933404324",
 "Chollet, F., &amp; others. (2015). <i>Keras</i> [Computer software]. https://keras.io",
 "Didan, K. (2021). <i>MODIS/Terra vegetation indices 16-day L3 global 250 m SIN grid "
 "V061</i> [Data set]. NASA EOSDIS Land Processes Distributed Active Archive Center. "
 "https://doi.org/10.5067/MODIS/MOD13Q1.061",
 "Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. "
 "<i>The Annals of Statistics, 29</i>(5), 1189–1232. https://doi.org/10.1214/aos/1013203451",
 "Gorelick, N., Hancher, M., Dixon, M., Ilyushchenko, S., Thau, D., &amp; Moore, R. (2017). "
 "Google Earth Engine: Planetary-scale geospatial analysis for everyone. <i>Remote Sensing "
 "of Environment, 202</i>, 18–27. https://doi.org/10.1016/j.rse.2017.06.031",
 "Harris, C. R., Millman, K. J., van der Walt, S. J., Gommers, R., Virtanen, P., "
 "Cournapeau, D., … Oliphant, T. E. (2020). Array programming with NumPy. <i>Nature, "
 "585</i>(7825), 357–362. https://doi.org/10.1038/s41586-020-2649-2",
 "Hochreiter, S., &amp; Schmidhuber, J. (1997). Long short-term memory. <i>Neural "
 "Computation, 9</i>(8), 1735–1780. https://doi.org/10.1162/neco.1997.9.8.1735",
 "Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. <i>Computing in Science "
 "&amp; Engineering, 9</i>(3), 90–95. https://doi.org/10.1109/MCSE.2007.55",
 "Jeffrey, S. J., Carter, J. O., Moodie, K. B., &amp; Beswick, A. R. (2001). Using spatial "
 "interpolation to construct a comprehensive archive of Australian climate data. "
 "<i>Environmental Modelling &amp; Software, 16</i>(4), 309–330. "
 "https://doi.org/10.1016/S1364-8152(01)00008-1",
 "North Australian Fire Information. (2026). <i>Northern Territory fire scar mapping</i> "
 "[Data set]. Darwin Centre for Bushfire Research, Charles Darwin University. "
 "https://firenorth.org.au",
 "Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., … "
 "Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. <i>Journal of Machine "
 "Learning Research, 12</i>, 2825–2830.",
 "Queensland Government. (2026). <i>SILO climate data: Data Drill</i> [Data set]. Long "
 "Paddock. https://www.longpaddock.qld.gov.au/silo/",
 "Ronneberger, O., Fischer, P., &amp; Brox, T. (2015). U-Net: Convolutional networks for "
 "biomedical image segmentation. In N. Navab, J. Hornegger, W. M. Wells, &amp; A. F. "
 "Frangi (Eds.), <i>Medical image computing and computer-assisted intervention – MICCAI "
 "2015</i> (pp. 234–241). Springer. https://doi.org/10.1007/978-3-319-24574-4_28",
 "Russell-Smith, J., Yates, C. P., Whitehead, P. J., Smith, R., Craig, R., Allan, G. E., "
 "Thackway, R., Frakes, I., Cridland, S., Meyer, M. C. P., &amp; Gill, A. M. (2007). "
 "Bushfires ‘down under’: Patterns and implications of contemporary Australian landscape "
 "burning. <i>International Journal of Wildland Fire, 16</i>(4), 361–377. "
 "https://doi.org/10.1071/WF07018",
]


def build():
    doc = BaseDocTemplate(str(OUT), pagesize=A4,
                          leftMargin=20*mm, rightMargin=20*mm,
                          topMargin=20*mm, bottomMargin=22*mm,
                          title="PRT565 Assessment 3 Report — Group 85 — PYRANTIS",
                          author="Harsh Rastogi, Saira Zafar, Tharushi Wimalachandra")
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="f")
    doc.addPageTemplates([PageTemplate(id="p", frames=[frame], onPage=page_bg)])
    W = doc.width
    st = []

    # ---------------------------------------------------------------- page 1
    st.append(Spacer(1, 26*mm))
    st.append(Paragraph("CHARLES DARWIN UNIVERSITY", S["cover_uni"]))
    st.append(Spacer(1, 3*mm))
    st.append(rule(1.6, RED, 54*mm))
    st.append(Spacer(1, 7*mm))
    st.append(Paragraph("PYRANTIS", S["cover_title"]))
    st.append(Paragraph("Which country burns in the Northern Territory, and when", S["cover_sub"]))
    st.append(Spacer(1, 10*mm))
    st.append(kv_table([
        ("Unit", "PRT565 Machine Learning, Artificial Intelligence and Algorithms"),
        ("Unit code", "PRT565"),
        ("Assessment", "Assessment 3 — Group Presentation (30% of the unit)"),
        ("Group number", "Group 85"),
        ("Campus", "Darwin (Danala) Campus"),
        ("Due", "27 September 2026, 22:00 Sydney time"),
    ], W))
    st.append(Spacer(1, 8*mm))
    st.append(Paragraph("GROUP MEMBERS", S["kicker"]))
    st.append(Spacer(1, 1*mm))
    nm = ParagraphStyle("m", fontName=SANS_B, fontSize=10.6, leading=15, textColor=INK)
    idl = ParagraphStyle("i", fontName=SANS, fontSize=10.2, leading=15, textColor=MUTED)
    mem = Table([[Paragraph("Harsh Rastogi", nm), Paragraph("386401", idl)],
                 [Paragraph("Saira Zafar", nm), Paragraph("407193", idl)],
                 [Paragraph("Tharushi Wimalachandra", nm), Paragraph("386594", idl)]],
                colWidths=[W*0.55, W*0.45])
    mem.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                             ("TOPPADDING", (0, 0), (-1, -1), 4),
                             ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                             ("LEFTPADDING", (0, 0), (-1, -1), 0),
                             ("LINEBELOW", (0, 0), (-1, -2), 0.5, RULE)]))
    st.append(mem)
    st.append(PageBreak())

    # ---------------------------------------------------------------- page 2
    section(st, "Recording, title and motivation", "The recorded presentation")
    link = ParagraphStyle("lk", fontName=SANS_B, fontSize=11, leading=16, textColor=RED)
    box = Table([[Paragraph("Link to the recorded presentation", S["note"])],
                 [Paragraph("[ paste the unlisted video link here before submitting ]", link)],
                 [Paragraph("The link is set so that anyone with it can view the file. "
                            "It was tested from a signed-out browser before submission.",
                            S["note"])]], colWidths=[W])
    box.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), TINT),
                             ("LEFTPADDING", (0, 0), (-1, -1), 10),
                             ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                             ("TOPPADDING", (0, 0), (-1, -1), 7),
                             ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
    st.append(box)
    st.append(Spacer(1, 7*mm))

    st.append(Paragraph("Project title", S["h2"]))
    st.append(Paragraph("PYRANTIS — multi-class classification of Northern Territory fire "
                        "regimes, and a forecast for 2026.", S["lead"]))

    st.append(Paragraph("What the project does", S["h2"]))
    st.append(Paragraph(
        "For every five-kilometre square of the Northern Territory, for every year from "
        "2000 to 2025, the project classifies what happened into one of three answers: no "
        "fire, a fire first detected on or before 31 July, or a fire first detected after "
        "it. Six classifiers are trained on the same 966,819 rows and compared on the same "
        "held-out years, and the strongest of them is then used to publish a forecast for "
        "2026 that can be checked against the satellite record in 2027.", S["body"]))
    st.append(Paragraph(
        "The models are fed 50 features built from three public sources: satellite fire "
        "scars from NAFI, daily weather from the Queensland Government's SILO archive, and "
        "MODIS vegetation greenness obtained through Google Earth Engine. Every feature is "
        "restricted to information that existed by 30 April of the year being predicted, so "
        "a forecast arrives before the fire season rather than describing it afterwards.",
        S["body"]))

    st.append(Paragraph("Why we chose this topic", S["h2"]))
    st.append(Paragraph(
        "Fire in the Northern Territory is not an emergency that arrives occasionally. "
        "Across 2000 to 2025 about a third of the Territory carried a fire in any given "
        "year, so the useful question is not whether country will burn but when. Before "
        "August the country is still damp: fires are cool, they creep, and rangers light "
        "them deliberately to break up the fuel. After July the grass has cured, and the "
        "same ignition runs with the wind, burns hot enough to kill canopy, and takes out "
        "fences, stock and country in an afternoon.", S["body"]))
    st.append(Paragraph(
        "That difference is a decision someone makes in April, which is what made the "
        "problem worth modelling. A classifier that says which squares are likely to carry "
        "a late fire, while there is still time to plan a burning programme, answers a "
        "question a land manager already asks. The topic also sits where the unit is "
        "taught, and all three data sources are open, so every number in this report can "
        "be reproduced from the scripts in the repository.", S["body"]))
    st.append(PageBreak())

    # ------------------------------------------------------------- 1. dataset
    section(st, "Section 1", "Dataset description")
    st.append(Paragraph(
        "Three archives are joined onto one grid of 0.05-degree cells, roughly five "
        "kilometres across. The grid is the SILO weather grid, so weather joins to fire "
        "without resampling either one.", S["body"]))
    st.append(kv_table([
        ("NAFI", "North Australian Fire Information, Charles Darwin University. "
                 "250-metre burnt-area rasters for 2000 to 2025. Each pixel carries the "
                 "month the fire was detected, which is what makes an early or late label "
                 "possible at all."),
        ("SILO", "Queensland Government Data Drill. Daily rainfall, maximum temperature, "
                 "vapour pressure deficit and solar radiation, interpolated from Bureau of "
                 "Meteorology station observations onto the same grid (Jeffrey et al., 2001)."),
        ("MODIS", "MOD13Q1 vegetation index at 250 m on a 16-day cycle, extracted through "
                  "Google Earth Engine (Didan, 2021; Gorelick et al., 2017). Greenness over "
                  "the wet season measures how much fuel is standing when the dry arrives."),
    ], W, label_w=26*mm))
    st.append(Spacer(1, 5*mm))
    st.append(Paragraph(
        "Labelling produced 1,207,570 cell-years across 46,445 cells. Cells on the coast "
        "and the state border have no complete neighbourhood, and the first five years "
        "carry too little history for the longest look-back windows, so the modelling table "
        "holds 966,819 rows over 2005 to 2025.", S["body"]))
    st.extend(fig("slide02_ef691008.png", W/mm - 2,
                  "Class balance across the modelling rows. The answer that matters most, "
                  "a fire after July, is also close to the rarest."))
    st.append(Paragraph(
        "Roughly seven rows in ten record no fire. An accuracy figure alone would therefore "
        "reward a model that never predicts a fire at all, which is why the comparison in "
        "Section 4 is led by macro-averaged F1 and why the models are trained with class "
        "weights.", S["body"]))

    # ------------------------------------------------ 2. analysis / features
    st.append(PageBreak())
    section(st, "Section 2", "Analysis, preprocessing and feature engineering")
    st.append(Paragraph("From burnt pixels to one answer per square", S["h2"]))
    st.append(Paragraph(
        "Each NAFI pixel holds the month it burned, from 1 to 12, or zero for no fire. "
        "Every 20 by 20 block of pixels is pooled into one cell, and the fraction that "
        "burned before August and the fraction that burned after July are counted "
        "separately. A cell counts as burnt only when at least 5% of it burned, and it "
        "enters the table at all only when at least half its pixels were readable through "
        "cloud. Whichever season holds more of the burnt area names the year.", S["body"]))
    st.append(Paragraph(
        "No metric can catch a wrong label, so the labels were checked against fire history "
        "that is already published. The pipeline puts 2011, the year after the 2010–11 La "
        "Niña, at 65.7% of the map burnt, the heaviest year on record, and 2020 at 15.9%, "
        "the lightest. Both agree with the published record (Russell-Smith et al., 2007).",
        S["body"]))

    st.append(Paragraph("What the models are allowed to know", S["h2"]))
    st.append(Paragraph(
        "Every one of the 50 features is built from information available by 30 April of "
        "the year being predicted. They fall into four groups: 26 fire-history features, 14 "
        "weather features, 7 greenness features and 3 that place the square.", S["body"]))
    st.append(kv_table([
        ("Fire history (26)", "Years since the last fire and the last late fire, how often "
                              "the square burnt over 3, 5, 10, 15, 20 and 25 years, what it "
                              "did in each of the two previous years, and what its eight "
                              "neighbours did last year."),
        ("Weather (14)", "Wet-season rainfall, heat, vapour pressure deficit and radiation, "
                         "the dry season before it, days since rain, and how far the wet sat "
                         "from that cell's own local normal."),
        ("Greenness (7)", "Mean, maximum and minimum greenness over the wet, the April "
                          "reading, how fast the country greened up, and two anomalies "
                          "against the cell's own history."),
        ("Place (3)", "Longitude, latitude and elevation, so the model can separate the Top "
                      "End from the arid south without being handed the regions."),
    ], W, label_w=34*mm))
    st.append(Spacer(1, 4*mm))

    st.append(Paragraph("Two tests against leakage", S["h2"]))
    st.append(Paragraph(
        "A feature that quietly reads the year it is meant to predict scores well and means "
        "nothing, so the pipeline was tested for it twice rather than trusted. First, every "
        "fire-history feature is rebuilt from years strictly before the target year inside a "
        "single function that both the training path and the forecast path call; written "
        "twice, the two copies would drift apart, and a forecast has no label to catch the "
        "drift. Second, one year's labels were shuffled and the features rebuilt from "
        "scratch. Had anything been reading the answer the scores would have collapsed, and "
        "they moved by as much as chance predicts.", S["body"]))

    st.append(Paragraph("Splitting by time", S["h2"]))
    st.append(Paragraph(
        "Fire years are not independent draws. A random shuffle would place 2024 in training "
        "and 2023 in test, letting a model read its own future, so the record is cut in "
        "chronological order: 2005 to 2019 for training, 2020 to 2022 for tuning, and 2023 "
        "to 2025 held out. The held-out years were scored once, at the end, after every "
        "decision about features and models had been made.", S["body"]))

    # -------------------------------------------------------- 3. the models
    st.append(PageBreak())
    section(st, "Section 3", "Algorithms and model overview")
    st.append(Paragraph(
        "Six classifiers were trained on identical rows and scored on identical held-out "
        "years. The first four are the algorithms named in the assessment brief; the last "
        "two were added to make the comparison harder to win.", S["body"]))
    st.append(kv_table([
        ("Decision tree", "A single CART tree. The readable baseline: every prediction can "
                          "be traced to a chain of thresholds."),
        ("Random forest", "500 trees on bootstrapped samples, voting. Handles the "
                          "interactions between rainfall, greenness and fire history without "
                          "being told they exist."),
        ("Naive Bayes", "Gaussian naive Bayes. Assumes the features are independent given "
                        "the class, which they are not, so it sets a probabilistic floor."),
        ("Multilayer ANN", "A dense feed-forward network trained with Keras (Chollet, 2015) "
                           "on the same 50 features."),
        ("LSTM", "A recurrent network reading each cell's 26-year sequence, with a second "
                 "input branch for the static weather and greenness features "
                 "(Hochreiter &amp; Schmidhuber, 1997)."),
        ("Gradient boosting", "Histogram-based boosted trees, each correcting the last "
                              "(Friedman, 2001). This is the model that won."),
    ], W, label_w=34*mm))
    st.append(Spacer(1, 4*mm))
    st.append(Paragraph(
        "Two baselines sit underneath all six. The first repeats whatever happened to that "
        "square last year; the second always answers unburnt. A model that cannot beat both "
        "has learned nothing worth having, and the second baseline is the one that exposes "
        "accuracy as a misleading measure here.", S["body"]))

    st.append(Paragraph("Correcting for imbalance", S["h2"]))
    st.append(Paragraph(
        "Left unweighted, every model quietly stopped predicting late fires, because being "
        "right about the common answer is cheaper. Class weights make that trade expensive. "
        "For gradient boosting the F1 on late fires rises from 0.420 to 0.553 with weighting, "
        "on the same data and the same split, while overall accuracy barely moves. Both "
        "variants of every model were trained and both are reported.", S["body"]))
    st.extend(fig("slide16_303f4460.png", W/mm - 2,
                  "Twenty-three experiments, each scored against the same fixed benchmark "
                  "and recorded as kept or reverted. One was kept."))
    st.append(Paragraph(
        "The final feature set was not assembled by intuition. Twenty-three experiments were "
        "run against a fixed benchmark and logged either way, including a U-Net that read "
        "fire as an image rather than a table. It scored 0.588, which is 0.048 below the "
        "benchmark, so it was reverted and kept out of the final model "
        "(Ronneberger et al., 2015).", S["body"]))

    # ---------------------------------------------------- 4. evaluation
    st.append(PageBreak())
    section(st, "Section 4", "Model performance evaluation")
    st.append(Paragraph(
        "All figures below come from 2023, 2024 and 2025, which no model saw during "
        "training or tuning. Precision, recall and F1 are macro-averaged, so each of the "
        "three classes counts equally and a model cannot score well by ignoring the rare "
        "ones. Balanced variants are shown, since weighting improved every model.", S["body"]))
    st.append(metrics_table(W))
    st.append(Paragraph("Table 1. Held-out performance on 2023–2025. The final column is the "
                        "class that matters most operationally.", S["cap"]))
    st.append(Paragraph(
        "Gradient boosting leads on F1 at 0.684 with accuracy of 0.725, and the LSTM follows "
        "at 0.673. The gap worth reading is not between the models but between the models "
        "and the naive rule: repeating last year scores 0.506 on identical data, so gradient "
        "boosting adds 0.178 over what a fire officer already knows without us. Always "
        "answering unburnt reaches 0.578 accuracy while scoring 0.244 on F1, which is the "
        "clearest argument for not reporting accuracy alone.", S["body"]))
    st.extend(fig("slide13_43bef23d.png", W/mm - 2,
                  "Balanced F1 on the held-out years. The two grey bars are the baselines."))
    st.append(Paragraph(
        "Every model finds late fires hardest. Gradient boosting reaches 0.553 on that "
        "class against 0.816 on unburnt, and its confusion matrix shows why: of the 35,285 "
        "squares that truly burned after July, it calls 10,478 of them no fire. Late fires "
        "are the rarest class and the most driven by weather after April, which is exactly "
        "the information the model is not allowed to see.", S["body"]))

    st.append(Paragraph("What each source of data bought", S["h2"]))
    st.extend(fig("slide14_97424825.png", W/mm - 60,
                  "Fire history alone, then weather, then greenness with longer memory and "
                  "elevation."))
    st.append(Paragraph(
        "Fire history on its own reaches 0.639. Adding weather takes it to 0.674, the single "
        "largest step in the project. The third run adds greenness together with the longer "
        "memory features and elevation for a further 0.010; isolated in the experiment "
        "ledger, greenness on its own is worth 0.029. It was kept because the gain holds on "
        "years the model never saw and because two of the model's five strongest inputs turn "
        "out to be greenness measures, though the Earth Engine extraction cost about ten "
        "minutes a run against sixty seconds for everything else.", S["body"]))

    st.append(PageBreak())
    st.append(Paragraph("Testing it sixty-six times", S["h2"]))
    st.append(Paragraph(
        "One good test year could be luck, so the record was cut at each year from 2014 to "
        "2024, the model was given only what came before the cut, and it was scored on every "
        "year after it. That is 66 separate tests.", S["body"]))
    st.extend(fig("slide15_7192015f.png", W/mm - 2,
                  "Skill against how far ahead the model was asked to guess, averaged over "
                  "66 folds."))
    st.append(Paragraph(
        "Skill does not fade with distance. Guessing one year ahead scores 0.635; guessing "
        "eleven years ahead it still scores 0.604, with the naive baseline well below it "
        "throughout. That is not the model being clever. It says the strongest clue is the "
        "wet season just gone, which is known by the end of April however old the fire "
        "history behind it is.", S["body"]))

    st.append(Paragraph("What the model leans on", S["h2"]))
    st.extend(fig("slide17_8dbe7376.png", W/mm - 2,
                  "Permutation importance, coloured by where each number came from."))
    st.append(Paragraph(
        "Nobody told the model that a wet, hot wet season grows the grass that carries an "
        "October fire. Its strongest inputs are the dryness of the air through the wet, "
        "average greenness, wet-season rainfall and what the neighbouring country did last "
        "year. That it lands close to where fire ecology already sits is the best evidence "
        "we have that it learned the problem rather than the dataset.", S["body"]))

    # ------------------------------------------------------- 5. limitations
    st.append(PageBreak())
    section(st, "Section 5", "Limitations")
    st.append(kv_table([
        ("Late fires", "At 0.553 the class that matters most is still the weakest. It is "
                       "the rarest and the most weather-driven, and weather after April is "
                       "what the 30 April rule deliberately excludes."),
        ("A season, not a day", "The model says which squares are likely to burn and roughly "
                                "when. It is not a fire-danger rating and should not be read "
                                "as one."),
        ("2026 is unverified", "No burnt-area record exists for a year still running, so the "
                               "forecast is published as a claim to be checked in 2027 rather "
                               "than as a result."),
        ("One Territory, one grid", "Everything is fitted to the Northern Territory at 0.05 "
                                    "degrees. Whether it transfers to the Kimberley or Cape "
                                    "York is an open question, not an assumption."),
    ], W, label_w=38*mm))

    # -------------------------------------------------------- 6. conclusion
    st.append(Spacer(1, 6*mm))
    section(st, "Section 6", "Conclusion")
    st.append(Paragraph(
        "Gradient boosting classified three fire regimes at 0.684 macro F1 and 0.725 "
        "accuracy on three years it had never seen, 0.178 above the naive rule of repeating "
        "last year. Sixty-six walk-forward folds show the result is not a single lucky "
        "split. Weather was the data that mattered most, greenness added a small but real "
        "amount, and a convolutional model that treated fire as an image performed worse "
        "than the tabular benchmark.", S["body"]))
    st.extend(fig("slide20_e979c197.png", W/mm - 2,
                  "The 2026 forecast against the 26-year average."))
    st.append(Paragraph(
        "Applied to 2026, the model expects 51% of the Territory to burn against a 26-year "
        "average of 33%, with 15% before August and 36% after July, at a mean confidence of "
        "0.733. It reads that from a wet season of 783 mm, 45% above the 2005 to 2019 "
        "average and the second wettest in 22 years, and the greenest April in the record. "
        "More water grew more grass, and more grass is more fuel.", S["body"]))
    st.append(Paragraph(
        "The model and its evidence are published as a public site so the forecast can be "
        "read by someone who is not a data scientist, and every sentence on that site is "
        "computed from the model output rather than typed in. The forecast can be checked "
        "against the satellite record in 2027.", S["body"]))
    st.append(KeepTogether([
        Spacer(1, 3*mm),
        kv_table([("Live site", "pyrantis.harshlabs.workers.dev"),
                  ("Source code", "github.com/harshrastogii/PYRANTIS")], W, label_w=30*mm),
    ]))

    # ---------------------------------------------------- contributions
    st.append(Spacer(1, 8*mm))
    section(st, "Section 7", "Individual contributions")
    st.append(kv_table([
        ("Harsh Rastogi (386401)", "Project lead. Problem framing and the choice of the "
                                   "three-class target, the data pipeline across all three "
                                   "sources, the 2026 forecast, and the public web product. "
                                   "Presents slides 1–5 and 19–22."),
        ("Saira Zafar (407193)", "Labelling from the NAFI rasters, feature engineering and "
                                 "the 30 April rule, the two leakage tests and the temporal "
                                 "split design. Presents slides 6–11."),
        ("Tharushi Wimalachandra (386594)", "Model training and tuning across all six "
                                            "classifiers, the experiment ledger, "
                                            "walk-forward validation and the results "
                                            "analysis. Presents slides 12–18."),
    ], W, label_w=48*mm))
    st.append(Spacer(1, 4*mm))
    st.append(Paragraph("All three members reviewed each other's work, and the presentation "
                        "runs to about 14 minutes with no member exceeding six.", S["body"]))

    # ----------------------------------------------------------- references
    st.append(Spacer(1, 8*mm))
    section(st, "References", "References")
    st.append(Paragraph("American Psychological Association, 7th edition.", S["note"]))
    st.append(Spacer(1, 2*mm))
    for r in REFERENCES:
        st.append(Paragraph(r, S["ref"]))

    doc.build(st)
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    build()
