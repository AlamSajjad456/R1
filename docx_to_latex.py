"""
Simple .docx -> LaTeX converter (no external deps).

Why:
- This repo is a Windows workspace where installing extra Python packages may be inconvenient.
- .docx files are ZIPs containing WordprocessingML (XML). We extract text + basic structure.

What it supports:
- Paragraph text
- Basic heading heuristics:
  - "CHAPTER N" + next paragraph title -> \\chapter{...}
  - Numbered headings like "1.1 Title" -> \\section{...}
  - "1.1.1 Title" -> \\subsection{...}
  - "1.1.1.1 Title" -> \\subsubsection{...}
- Simple bullet/number lists (best-effort) using Word numbering properties

Usage:
  python docx_to_latex.py r1.docx r1_content.tex
"""

from __future__ import annotations

import re
import sys
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {"w": W_NS}


def _escape_latex(text: str) -> str:
    # Minimal escaping for common characters
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    out = []
    for ch in text:
        out.append(repl.get(ch, ch))
    return "".join(out)


def _get_p_style(p: ET.Element) -> Optional[str]:
    ppr = p.find("w:pPr", NS)
    if ppr is None:
        return None
    pstyle = ppr.find("w:pStyle", NS)
    if pstyle is None:
        return None
    return pstyle.attrib.get(f"{{{W_NS}}}val")


def _is_list_paragraph(p: ET.Element) -> bool:
    ppr = p.find("w:pPr", NS)
    if ppr is None:
        return False
    return ppr.find("w:numPr", NS) is not None


def _paragraph_text(p: ET.Element) -> str:
    # Concatenate all text runs; treat <w:br/> as newline.
    parts: List[str] = []
    for node in p.iter():
        tag = node.tag.rsplit("}", 1)[-1]
        if tag == "t" and node.text:
            parts.append(node.text)
        elif tag == "tab":
            parts.append("\t")
        elif tag == "br":
            parts.append("\n")
    return "".join(parts).strip()


@dataclass
class Para:
    text: str
    style: Optional[str]
    is_list: bool


def _read_docx_paragraphs(docx_path: Path) -> List[Para]:
    with zipfile.ZipFile(docx_path) as zf:
        xml = zf.read("word/document.xml")
    root = ET.fromstring(xml)
    out: List[Para] = []
    for p in root.findall(".//w:p", NS):
        text = _paragraph_text(p)
        if not text:
            continue
        out.append(Para(text=text, style=_get_p_style(p), is_list=_is_list_paragraph(p)))
    return out


_RE_CHAPTER = re.compile(r"^\s*CHAPTER\s+(\d+)\s*$", re.IGNORECASE)
_RE_NUMBERED = re.compile(r"^\s*(\d+(?:\.\d+){1,3})\s+(.+?)\s*$")


def _emit_heading_from_text(text: str) -> Optional[Tuple[str, str]]:
    """
    Return (cmd, title) where cmd is LaTeX sectioning command without backslash.
    """
    m = _RE_NUMBERED.match(text)
    if not m:
        return None
    nums = m.group(1)
    title = m.group(2).strip()
    depth = nums.count(".")  # 1 => 1.1, 2 => 1.1.1, ...
    if depth == 1:
        return ("section", title)
    if depth == 2:
        return ("subsection", title)
    if depth == 3:
        return ("subsubsection", title)
    return ("paragraph", title)


def _to_latex(paras: List[Para]) -> str:
    lines: List[str] = []
    i = 0
    in_list = False
    list_env = "itemize"

    def close_list() -> None:
        nonlocal in_list
        if in_list:
            lines.append(rf"\end{{{list_env}}}")
            lines.append("")
            in_list = False

    while i < len(paras):
        p = paras[i]
        # Chapter heuristic: "CHAPTER N" then next paragraph title.
        m = _RE_CHAPTER.match(p.text)
        if m:
            close_list()
            title = None
            if i + 1 < len(paras):
                title = paras[i + 1].text.strip()
            chap_title = title or f"Chapter {m.group(1)}"
            lines.append(rf"\chapter{{{_escape_latex(chap_title)}}}")
            lines.append("")
            i += 2 if title else 1
            continue

        # Headings from numbering like 1.1, 1.1.1
        heading = _emit_heading_from_text(p.text)
        if heading is not None:
            close_list()
            cmd, title = heading
            lines.append(rf"\{cmd}{{{_escape_latex(title)}}}")
            lines.append("")
            i += 1
            continue

        # Lists
        if p.is_list:
            if not in_list:
                # Best-effort: treat as bullet list
                list_env = "itemize"
                lines.append(rf"\begin{{{list_env}}}")
                in_list = True
            lines.append(rf"\item {_escape_latex(p.text)}")
            i += 1
            continue
        else:
            close_list()

        # Normal paragraph
        lines.append(_escape_latex(p.text))
        lines.append("")
        i += 1

    close_list()
    return "\n".join(lines).rstrip() + "\n"


def main(argv: List[str]) -> int:
    if len(argv) != 3:
        print("Usage: python docx_to_latex.py <input.docx> <output.tex>", file=sys.stderr)
        return 2
    src = Path(argv[1]).expanduser().resolve()
    dst = Path(argv[2]).expanduser().resolve()
    paras = _read_docx_paragraphs(src)
    tex = _to_latex(paras)
    dst.write_text(tex, encoding="utf-8")
    print(f"Wrote {dst} ({len(paras)} paragraphs).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

