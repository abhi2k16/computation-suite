# Copyright (c) 2026 Abhijeet <abhijeetshandilya19@gmail.com>
# SPDX-License-Identifier: MIT
"""
build_user_guide_html.py -- render USER_GUIDE.md as the styled, single-file USER_GUIDE.html.

    pip install markdown pygments
    python tools/build_user_guide_html.py                 # USER_GUIDE.md -> USER_GUIDE.html
    python tools/build_user_guide_html.py in.md out.html

The markdown is the source of truth; edit it and rebuild. Every ``<a id="s-..."></a>`` anchor that sits right
above a heading becomes an entry in the sidebar (level-1 headings flush left, deeper ones indented).
GitHub-style alert quotes (``> [!NOTE]``, TIP, WARNING, IMPORTANT) become coloured callout boxes.
"""
__author__ = "Abhijeet <abhijeetshandilya19@gmail.com>"
import html
import re
import sys
from pathlib import Path

HEAD_CSS = r''':root{--bg:#0b1220;--panel:#131c2e;--panel2:#1b2740;--line:#26344f;--tx:#e5edf7;--dim:#94a3b8;--cy:#22d3ee;--pu:#a78bfa;--pk:#e879f9;--gr:#34d399;--am:#fbbf24;--rd:#f87171;--bl:#60a5fa}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--tx);font:15px/1.65 system-ui,-apple-system,Segoe UI,sans-serif}
#side{position:fixed;top:0;left:0;bottom:0;width:270px;overflow:auto;background:linear-gradient(180deg,#0f172a,#131c2e);border-right:1px solid var(--line);padding:14px 12px}
#side a{display:block;color:var(--tx);text-decoration:none;padding:3px 8px;border-radius:6px;font-size:13px;font-weight:600}#side a.sm{color:var(--dim);font-weight:400;padding-left:18px}#side a:hover{background:var(--panel2);color:var(--cy)}
#main{margin-left:270px;padding:26px 44px 90px;max-width:1180px}
h1{font-size:30px;background:linear-gradient(90deg,var(--cy),var(--pu),var(--pk));-webkit-background-clip:text;background-clip:text;color:transparent;margin-top:46px}
h2{border-left:5px solid var(--cy);padding-left:10px;margin-top:38px}h3{border-left:4px solid var(--pu);padding-left:10px;margin-top:28px}h4{color:var(--pk)}
a{color:var(--cy)}code{background:#0f1a30;border:1px solid var(--line);border-radius:5px;padding:1px 5px;font-size:.88em;color:#bde9ff}
pre{background:#0f1a30;border:1px solid var(--line);border-left:4px solid var(--cy);border-radius:10px;padding:12px 14px;overflow:auto}pre code{background:none;border:0;padding:0;color:inherit}
.hl{background:#0f1a30;border-radius:10px;margin:10px 0}.hl pre{border:1px solid var(--line);border-left:4px solid var(--cy);margin:0}
'''
TAIL_CSS = r'''table{border-collapse:collapse;width:100%;margin:12px 0;background:var(--panel);border-radius:10px;overflow:hidden;display:block;overflow-x:auto}th{background:linear-gradient(90deg,#1b2740,#25345a);text-align:left}
th,td{border-bottom:1px solid var(--line);padding:7px 10px;vertical-align:top}tr:hover td{background:#17223a}
details{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:6px 14px;margin:8px 0 18px}summary{cursor:pointer;color:var(--cy)}
blockquote{border-left:4px solid var(--pu);margin:12px 0;padding:4px 14px;background:var(--panel);border-radius:0 8px 8px 0;color:var(--dim)}
.callout{border-radius:10px;padding:8px 14px;margin:14px 0;border:1px solid;border-left-width:5px}.callout p{margin:4px 0}.callout .ct{font-weight:700;text-transform:uppercase;font-size:12px;letter-spacing:.06em}
.callout.note{background:#10243a;border-color:var(--bl)}.callout.note .ct{color:var(--bl)}.callout.tip{background:#0f2a22;border-color:var(--gr)}.callout.tip .ct{color:var(--gr)}
.callout.warning{background:#2e2410;border-color:var(--am)}.callout.warning .ct{color:var(--am)}.callout.important{background:#2a1530;border-color:var(--pk)}.callout.important .ct{color:var(--pk)}
div[align=center]{text-align:center;padding:10px 0}div[align=center] h1{margin-top:6px;font-size:38px}img{max-width:100%}hr{border:0;border-top:1px solid var(--line);margin:26px 0}
@media(max-width:900px){#side{display:none}#main{margin-left:0;padding:16px}}
'''

CENTER_OPEN, CENTER_CLOSE = "@@CENTER_OPEN@@", "@@CENTER_CLOSE@@"
ALERT = re.compile(r"^>\s*\[!(NOTE|TIP|WARNING|IMPORTANT|CAUTION)\]\s*$", re.I)


def _plain(text):
    """Heading text without markdown markup, for the sidebar."""
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\*\*([^*]*)\*\*", r"\1", text)
    return html.escape(text.strip(), quote=False)


def _nav(lines):
    items = []
    for i, line in enumerate(lines):
        m = re.fullmatch(r'<a id="(s-[^"]+)"></a>', line.strip())
        if not m:
            continue
        j = i + 1
        while j < len(lines) and not lines[j].strip():
            j += 1
        h = re.match(r"^(#{1,6}) (.*)$", lines[j]) if j < len(lines) else None
        if h:
            items.append((m.group(1), len(h.group(1)), _plain(h.group(2))))
    return "".join(f'<a class="{"" if lvl == 1 else "sm"}" href="#{a}">{t}</a>' for a, lvl, t in items)


def _callouts(lines, md):
    """Replace ``> [!KIND]`` quote blocks by placeholders; return the new lines and {placeholder: html}."""
    out, boxes, i = [], {}, 0
    while i < len(lines):
        m = ALERT.match(lines[i])
        if not m:
            out.append(lines[i])
            i += 1
            continue
        kind = m.group(1).lower()
        i += 1
        body = []
        while i < len(lines) and lines[i].startswith(">"):
            body.append(re.sub(r"^>\s?", "", lines[i]))
            i += 1
        inner = md.convert("\n".join(body))
        md.reset()
        key = f"@@CALLOUT{len(boxes)}@@"
        boxes[key] = f'<div class="callout {kind}"><div class="ct">{kind.title()}</div>{inner}</div>'
        out += ["", key, ""]
    return out, boxes


def build(md_text):
    import markdown
    from pygments.formatters import HtmlFormatter
    md = markdown.Markdown(extensions=["tables", "fenced_code", "codehilite"],
                           extension_configs={"codehilite": {"css_class": "hl", "guess_lang": False}})
    lines = md_text.splitlines()
    nav = _nav(lines)
    title = _plain(next((ln[2:] for ln in lines if ln.startswith("# ")), "User guide"))
    lines, boxes = _callouts(lines, md)
    text = "\n".join(lines)
    text = text.replace('<div align="center">', CENTER_OPEN).replace("</div>", CENTER_CLOSE)
    body = md.convert(text)
    body = body.replace(f"<p>{CENTER_OPEN}</p>", '<div align="center">').replace(CENTER_OPEN, '<div align="center">')
    body = body.replace(f"<p>{CENTER_CLOSE}</p>", "</div>").replace(CENTER_CLOSE, "</div>")
    for key, box in boxes.items():
        body = body.replace(f"<p>{key}</p>", box).replace(key, box)
    css = HEAD_CSS + HtmlFormatter(style="monokai").get_style_defs(".hl") + "\n" + TAIL_CSS
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{title}</title>\n'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">\n<style>\n{css}</style></head>'
            f'<body><nav id="side">{nav}</nav><div id="main">{body}</div></body></html>')


def main(argv):
    root = Path(__file__).resolve().parent.parent
    src = Path(argv[1]) if len(argv) > 1 else root / "USER_GUIDE.md"
    dst = Path(argv[2]) if len(argv) > 2 else src.with_suffix(".html")
    dst.write_text(build(src.read_text(encoding="utf-8")), encoding="utf-8", newline="\n")
    print(f"wrote {dst} ({dst.stat().st_size} bytes)")


if __name__ == "__main__":
    main(sys.argv)
