"""Render the code excerpts the Assessment 4 report shows as figures.

Each excerpt is cut from the real file by line number, so a screenshot in the report is
always the code that produced the results. Rendering goes through Pygments to HTML and
then a headless Chrome screenshot, which keeps the type sharp at print size.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports" / "a4" / "figures"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

# name, file, [(first, last), ...] inclusive line ranges; a gap between ranges is shown
EXCERPTS = [
    ("c01_labels", "scripts/build_grid.py", [(102, 110), (125, 128)]),
    ("c02_features", "scripts/build_features.py", [(70, 80), (100, 109)]),
    ("c03_split_weights", "scripts/train_models.py", [(201, 204), (233, 245)]),
    ("c04_networks", "scripts/train_models.py", [(286, 294), (307, 316)]),
    ("c05_tuning", "scripts/robustness.py", [(195, 198), (219, 225)], 800),
    ("c06_scoring", "scripts/train_models.py", [(111, 123)], 860),
]

CSS = """
body{margin:0;background:#fff}
.card{font:12.5px/1.5 Menlo,Consolas,monospace;border:1px solid #d9dee0;border-radius:6px;
 overflow:hidden;width:700px}
.bar{background:#f3f5f6;border-bottom:1px solid #d9dee0;padding:6px 12px;
 font:600 12px -apple-system,Helvetica,sans-serif;color:#3b4448}
.bar span{font-weight:400;color:#6b7479;margin-left:8px}
pre{margin:0;padding:9px 12px;white-space:pre-wrap;overflow-wrap:anywhere}
.row{display:block;padding-left:4.6ch;text-indent:-4.6ch}
.ln{color:#9aa4a9;display:inline-block;width:3.2ch;text-align:right;margin-right:1.4ch;
 user-select:none}
.gap{color:#9aa4a9;padding:2px 14px;background:#fafbfb;border-top:1px dashed #e1e5e7;
 border-bottom:1px dashed #e1e5e7;font:12px Menlo,monospace}
"""


def render(name: str, rel: str, ranges, width: int = 700) -> None:
    lines = (ROOT / rel).read_text().splitlines()
    fmt = HtmlFormatter(style="friendly", nowrap=True)
    blocks, total = [], 0
    for k, (a, b) in enumerate(ranges):
        chunk = lines[a - 1:b]
        # dedent the chunk as a whole so nested code stays readable at print width
        ind = min((len(l) - len(l.lstrip()) for l in chunk if l.strip()), default=0)
        body = highlight("\n".join(l[ind:] for l in chunk), PythonLexer(), fmt).splitlines()
        numbered = "".join(f'<span class="row"><span class="ln">{a + i}</span>{h}</span>' for i, h in enumerate(body))
        if k:
            blocks.append('<div class="gap">⋯</div>')
        blocks.append(f"<pre>{numbered}</pre>")
        total += len(chunk)
    span = ", ".join(f"lines {a}–{b}" for a, b in ranges)
    css = CSS.replace("width:700px", f"width:{width}px")
    html = (f"<!doctype html><meta charset='utf-8'><style>{css}{fmt.get_style_defs('.card')}"
            f"</style><div class='card'><div class='bar'>{rel}<span>{span}</span></div>"
            + "".join(blocks) + "</div>")
    height = 60 + total * 40 + (len(ranges) - 1) * 26 + 24
    with tempfile.TemporaryDirectory() as tmp:
        page = Path(tmp) / "c.html"
        page.write_text(html)
        shot = Path(tmp) / "c.png"
        subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                        "--force-device-scale-factor=2", f"--window-size={width + 4},{height}",
                        f"--screenshot={shot}", page.as_uri()],
                       check=True, capture_output=True)
        from PIL import Image, ImageChops
        im = Image.open(shot).convert("RGB")
        bg = Image.new("RGB", im.size, (255, 255, 255))
        box = ImageChops.difference(im, bg).getbbox()
        im.crop(box).save(OUT / f"{name}.png")
    print("wrote", name)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    only = set(sys.argv[1:])
    for name, rel, ranges, *w in EXCERPTS:
        if not only or name in only:
            render(name, rel, ranges, *w)


if __name__ == "__main__":
    main()
