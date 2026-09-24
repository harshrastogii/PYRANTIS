"""One-page script for the added slide, in the full script's own styles."""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import build_script_pdf as S0
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer

OUT = pathlib.Path.home() / "Downloads" / "PYRANTIS_script_new_slide_16.pdf"
S = S0.S
lines = (S0.ROOT / "reports" / "_new_slide_lines.txt").read_text().strip()
words = len(lines.split())

doc = BaseDocTemplate(str(OUT), pagesize=A4, leftMargin=20*mm, rightMargin=20*mm,
                      topMargin=20*mm, bottomMargin=20*mm,
                      title="PYRANTIS — script for the added slide 16")
doc.addPageTemplates([PageTemplate(frames=[Frame(doc.leftMargin, doc.bottomMargin,
                                                 doc.width, doc.height)])])
st = [
  Paragraph("PYRANTIS — one added slide", S["cover_title"]),
  Paragraph("Where it goes", S["h1"]),
  Paragraph("New <b>slide 16</b>, in Tharushi's part. Put it right after slide 15 "
            "(“One good year could be luck…”) and right before the experiments slide "
            "(“Twenty-three experiments…”).", S["say"]),
  Paragraph("It is <b>not</b> Tharushi's last slide, so there is no handover here. Her "
            "handover to Harsh stays on the Limitations slide. The old slides 16 to 24 "
            "become 17 to 25.", S["say"]),
  Spacer(1, 4*mm),
  Paragraph(S0.THARU, S["h1"]),
  Paragraph(f"SLIDE 16 · about {round(words / S0.WPM * 60)} seconds", S["slug"]),
  Paragraph("Point to each box in turn: top left, top right, bottom left, bottom right. "
            "Then carry straight on to the experiments slide.", S["dir"]),
  Paragraph(lines, S["say"]),
  Spacer(1, 6*mm),
  Paragraph(f"{words} words, about {round(words / S0.WPM * 60)} seconds at a normal pace. "
            "The recording runs about 14:10 now, so with this slide it comes to about 14:45, "
            "inside the 15-minute limit, and Tharushi stays well under 6 minutes.", S["note"]),
]
doc.build(st)
print("wrote", OUT)
