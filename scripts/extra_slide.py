"""Build the one extra slide for Tharushi's section, cut from the user's own deck.

Slide 15 supplies the kicker, title, footer and background; the Limitations slide
supplies the row style (rule, red tick, Georgia heading, body text). Every other slide
is dropped, so the file holds only the new slide. Figures come from robustness.json.
"""
import copy, json, pathlib
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "reports" / "PYRANTIS Presentation.pptx"
OUT = pathlib.Path.home() / "Downloads" / "PYRANTIS_new_slide_16.pptx"
R = json.load(open(ROOT / "reports" / "robustness.json"))
M, TU, CV = R["models"], R["tuning"], R["random_cv"]
gb, rf, dt = M["gradient boosting"], M["random forest"], TU["decision tree"]
NEW_NO, TOTAL = 16, 25

prs = Presentation(SRC)
base, rows = prs.slides[14], prs.slides[17]
by = lambda s, name: next(sh for sh in s.shapes if sh.name == name)

def settext(shape, text, size=None, color=None, font=None):
    tf = shape.text_frame
    p0 = tf.paragraphs[0]
    for p in tf.paragraphs[1:]:
        p._p.getparent().remove(p._p)
    runs = p0.runs
    for r in runs[1:]:
        r._r.getparent().remove(r._r)
    r = runs[0]; r.text = text
    if size: r.font.size = Pt(size)
    if color: r.font.color.rgb = RGBColor.from_string(color)
    if font: r.font.name = font

def place(el, x, y, w=None, h=None):
    xfrm = el.find(".//{http://schemas.openxmlformats.org/drawingml/2006/main}xfrm")
    off, ext = xfrm[0], xfrm[1]
    off.set("x", str(int(Inches(x)))); off.set("y", str(int(Inches(y))))
    if w is not None: ext.set("cx", str(int(Inches(w))))
    if h is not None: ext.set("cy", str(int(Inches(h))))

# ---- clear slide 15's body: chart, side card and its text
for name in ("Picture 5", "Rectangle 6", "Rectangle 7", "TextBox 8", "TextBox 9"):
    el = by(base, name)._element; el.getparent().remove(el)

settext(by(base, "TextBox 2"), "CHECKING THE RESULTS")
settext(by(base, "TextBox 3"), "We tuned every model, and checked it would hold up.")
settext(by(base, "TextBox 4"), "Four checks, all scored on years the models had not seen.")
place(by(base, "TextBox 4")._element, 0.82, 1.72)
settext(by(base, "TextBox 12"), str(NEW_NO))
prog = by(base, "Connector 11")._element
prog.find(".//{http://schemas.openxmlformats.org/drawingml/2006/main}xfrm")[1].set(
    "cx", str(int(Inches(11.69 * NEW_NO / TOTAL))))

# ---- the 2 x 2 grid, built from the Limitations slide's own pieces
T_RULE, T_TICK = by(rows, "Connector 4")._element, by(rows, "Rectangle 5")._element
T_HEAD, T_BODY = by(rows, "TextBox 6")._element, by(rows, "TextBox 8")._element
tree = base.shapes._spTree
f = lambda x: f"{x:.3f}"
CELLS = [
 ("Tuned properly", f"{f(TU['gradient boosting']['test_f1_manual'])} → {f(TU['gradient boosting']['test_f1_tuned'])}",
  f"A search over every model's settings, scored on the validation years only, left "
  f"gradient boosting where it was. The decision tree gained most: {f(dt['test_f1_manual'])} "
  f"→ {f(dt['test_f1_tuned'])}."),
 ("Checked for overfitting", f"{f(gb['f1_train'])} → {f(gb['f1_test'])}",
  f"Gradient boosting on the years it trained on, then on years it never saw. The random "
  f"forest memorises far more: {f(rf['f1_train'])} → {f(rf['f1_test'])}."),
 ("Scored beyond accuracy", f"ROC-AUC {f(gb['roc_auc_macro'])}  ·  PR-AUC {f(gb['pr_auc_macro'])}",
  f"These score the probabilities behind each answer. On late fires the PR-AUC is "
  f"{f(gb['pr_auc_late'])}, about twice what guessing would get."),
 ("Why not shuffled cross-validation", f"{CV['mean']:.3f} vs {CV['forward_test_f1']:.3f}",
  "Shuffled 5-fold cross-validation leaks between neighbouring years and overstates the "
  "score. Testing forward in time gives the real figure."),
]
X, Wc, Y = (0.82, 6.82), 5.69, (2.50, 4.45)
for i, (head, fig, body) in enumerate(CELLS):
    x, y = X[i % 2], Y[i // 2]
    rule = copy.deepcopy(T_RULE); place(rule, x, y, Wc, 0); tree.append(rule)
    tick = copy.deepcopy(T_TICK); place(tick, x, y); tree.append(tick)
    h = copy.deepcopy(T_HEAD); place(h, x, y + 0.20, Wc, 0.34); tree.append(h)
    g = copy.deepcopy(T_HEAD); place(g, x, y + 0.58, Wc, 0.48); tree.append(g)
    b = copy.deepcopy(T_BODY); place(b, x, y + 1.10, Wc, 0.70); tree.append(b)
    shapes = {s._element: s for s in base.shapes}
    settext(shapes[h], head)
    settext(shapes[g], fig, size=22, color="C8341B")
    settext(shapes[b], body)
end = copy.deepcopy(T_RULE); place(end, 0.82, 6.27, 11.69, 0); tree.append(end)

# ---- speaker notes carry the lines
base.notes_slide.notes_text_frame.text = open(ROOT / "reports" / "_new_slide_lines.txt").read()

# ---- keep only this slide
lst = prs.slides._sldIdLst
keep = base.part
for sid in list(lst):
    if prs.part.related_part(sid.rId) is not keep:
        prs.part.drop_rel(sid.rId); lst.remove(sid)
prs.save(OUT)
print("wrote", OUT, f"{OUT.stat().st_size/1e6:.1f} MB")
