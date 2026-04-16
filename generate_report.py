"""
generate_report.py

Reads all *_report.md files from RESULT_DIR, combines them into a single
PDF report, and saves it as <RESULT_DIR>/<scene_name>_analysis_report.pdf.

Uses pandoc + xelatex (both available on this system).

Usage:
    python3 generate_report.py <scene_name> <result_dir>
"""

import sys
import os
import re
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path


# ── Report order ──────────────────────────────────────────────────────────────

REPORT_FILES = [
    "01_generate_dataset_report.md",
    "02_rt_comparison_report.md",
    "03_localization_report.md",
    "04_channel_charting_report.md",
]

# ── LaTeX / pandoc metadata header ───────────────────────────────────────────

def _latex_escape(text: str) -> str:
    """Escape characters that are special in LaTeX text mode."""
    for char, replacement in [
        ("\\", "\\textbackslash{}"),
        ("_", "\\_"),
        ("%", "\\%"),
        ("&", "\\&"),
        ("#", "\\#"),
        ("$", "\\$"),
        ("{", "\\{"),
        ("}", "\\}"),
        ("^", "\\^{}"),
        ("~", "\\textasciitilde{}"),
    ]:
        text = text.replace(char, replacement)
    return text


def make_yaml_header(scene_name: str) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    scene_latex = _latex_escape(scene_name)
    return f"""\
---
title: "Wireless Channel Analysis Report"
subtitle: "Scene: {scene_name}"
date: "{now}"
geometry: "left=2.5cm,right=2.5cm,top=2.5cm,bottom=2.5cm"
fontsize: 11pt
mainfont: "DejaVu Serif"
monofont: "DejaVu Sans Mono"
colorlinks: true
linkcolor: "NavyBlue"
urlcolor: "NavyBlue"
toc: true
toc-depth: 2
numbersections: false
header-includes:
  - \\usepackage{{booktabs}}
  - \\usepackage{{longtable}}
  - \\usepackage{{graphicx}}
  - \\usepackage{{float}}
  - \\usepackage{{fancyhdr}}
  - \\usepackage{{xcolor}}
  - \\definecolor{{codebg}}{{RGB}}{{245,245,245}}
  - \\usepackage{{fvextra}}
  - \\DefineVerbatimEnvironment{{Highlighting}}{{Verbatim}}{{breaklines,commandchars=\\\\\\{{\\}},fontsize=\\small,bgcolor=codebg}}
  - \\pagestyle{{fancy}}
  - \\fancyhf{{}}
  - \\fancyhead[L]{{\\small {scene_latex} --- Channel Analysis}}
  - \\fancyhead[R]{{\\small {now}}}
  - \\fancyfoot[C]{{\\thepage}}
  - \\renewcommand{{\\headrulewidth}}{{0.4pt}}
---

"""


# ── Fix image paths to absolute so pandoc finds them from the temp dir ────────

def fix_image_paths(md_text: str, base_dir: Path) -> str:
    """Replace relative image paths with absolute paths."""
    def _abs(match):
        alt = match.group(1)
        path = match.group(2)
        if path.startswith(("http://", "https://", "/")):
            return match.group(0)
        abs_path = (base_dir / path).resolve()
        if abs_path.exists():
            return f"![{alt}]({abs_path})"
        return match.group(0)  # leave as-is if not found

    return re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", _abs, md_text)


# ── Add a page break between reports ─────────────────────────────────────────

PAGE_BREAK = "\n\n\\newpage\n\n"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 generate_report.py <scene_name> <result_dir>")
        sys.exit(1)

    scene_name = sys.argv[1]
    result_dir = Path(sys.argv[2]).resolve()

    if not result_dir.is_dir():
        print(f"Error: result directory not found: {result_dir}")
        sys.exit(1)

    print(f"Generating PDF report for scene '{scene_name}' from {result_dir}")

    # ── Collect and combine Markdown sections ─────────────────────────────────
    sections = []
    for name in REPORT_FILES:
        md_file = result_dir / name
        if md_file.exists():
            text = md_file.read_text(encoding="utf-8")
            text = fix_image_paths(text, result_dir)
            sections.append(text)
            print(f"  + {name}")
        else:
            print(f"  - {name} (not found, skipping)")

    if not sections:
        print("No report .md files found — nothing to generate.")
        sys.exit(1)

    combined_md = make_yaml_header(scene_name) + PAGE_BREAK.join(sections)

    # ── Write combined Markdown to a temp file ────────────────────────────────
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".md", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(combined_md)
        tmp_path = tmp.name

    output_pdf = result_dir / f"{scene_name}_analysis_report.pdf"

    # ── Call pandoc ───────────────────────────────────────────────────────────
    cmd = [
        "pandoc",
        tmp_path,
        "--pdf-engine=xelatex",
        "--from=markdown+raw_tex",
        "--highlight-style=tango",
        "-o", str(output_pdf),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(result_dir),
        )
        if result.returncode != 0:
            print("pandoc stderr:\n", result.stderr[-3000:])
            sys.exit(result.returncode)
        print(f"PDF saved → {output_pdf}  ({output_pdf.stat().st_size / 1e3:.1f} KB)")
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    main()
