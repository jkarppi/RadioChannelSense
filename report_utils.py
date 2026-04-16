"""
report_utils.py — Export an executed notebook to a Markdown report.

Usage (add as last cell in any notebook, after saving with Ctrl+S):

    from report_utils import save_notebook_report
    out = save_notebook_report("01_generate_dataset.ipynb", SCENE_DIR)
    print(f"Report → {out}")

The report is written to  <SCENE_DIR>/<nb_stem>_report.md
Cell-output figures land in  <SCENE_DIR>/<nb_stem>_report_files/
Saved figures from  <SCENE_DIR>/pictures/  are appended as a gallery.
"""

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path


# ── MarkdownReport ─────────────────────────────────────────────────────────────

class _Tee:
    """Write to multiple streams simultaneously (for stdout tee-ing)."""
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)

    def flush(self):
        for s in self._streams:
            s.flush()


class MarkdownReport:
    """Lightweight Markdown report builder for plain-Python scripts.

    Usage::

        report = MarkdownReport()
        report.add("# Title\\n\\nIntro text.")
        with report.capture():
            my_function_that_prints()
        report.figure(PICTURES_DIR / 'plot.png', OUTPUT_DIR)
        report.save(OUTPUT_DIR / 'my_report.md')

    ``capture()`` tees stdout so output still appears in the terminal/log
    *and* is recorded for the report as an indented block.
    """

    def __init__(self):
        self._sections: list[str] = []

    def add(self, text: str) -> None:
        """Append a raw Markdown text block."""
        self._sections.append(text.rstrip())

    @contextlib.contextmanager
    def capture(self):
        """Tee stdout: output goes to terminal AND is captured for the report."""
        buf = io.StringIO()
        original = sys.stdout
        sys.stdout = _Tee(original, buf)
        try:
            yield
        finally:
            sys.stdout = original
            captured = buf.getvalue().strip()
            if captured:
                indented = "\n".join("    " + line for line in captured.splitlines())
                self._sections.append(indented)

    def figure(self, abs_path, base_dir) -> None:
        """Add a Markdown image link relative to base_dir (only if file exists)."""
        abs_path = Path(abs_path)
        if abs_path.exists():
            rel = abs_path.relative_to(Path(base_dir))
            caption = abs_path.stem.replace("_", " ").title()
            self._sections.append(f"![{caption}]({rel})")

    def save(self, path) -> Path:
        """Write the accumulated Markdown to *path* and return it."""
        path = Path(path)
        path.write_text("\n\n".join(self._sections) + "\n")
        print(f"Report → {path}")
        return path


def save_notebook_report(
    nb_path: str | Path,
    scene_dir: str | Path,
    pics_dir: str | Path | None = None,
) -> Path:
    """Convert an executed notebook to Markdown and save to scene_dir.

    Parameters
    ----------
    nb_path   : path to the .ipynb file (must be saved before calling)
    scene_dir : destination directory (the scene output folder)
    pics_dir  : directory of saved figures to append as a gallery.
                Defaults to <scene_dir>/pictures/ if not provided.

    Returns
    -------
    Path to the generated .md file.
    """
    import subprocess

    nb_path   = Path(nb_path).resolve()
    scene_dir = Path(scene_dir).resolve()
    scene_dir.mkdir(parents=True, exist_ok=True)

    out_stem = nb_path.stem + "_report"
    out_md   = scene_dir / f"{out_stem}.md"

    result = subprocess.run(
        [
            "jupyter", "nbconvert", "--to", "markdown",
            f"--output={out_stem}",
            f"--output-dir={scene_dir}",
            str(nb_path),
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"nbconvert failed for {nb_path.name}:\n{result.stderr}"
        )

    # ── Append gallery of saved figures from pictures/ ────────────────────────
    pics_dir = Path(pics_dir).resolve() if pics_dir is not None else scene_dir / "pictures"
    figs = []
    if pics_dir.exists():
        figs = sorted(pics_dir.glob("*.png")) + sorted(pics_dir.glob("*.jpg"))

    if figs:
        with open(out_md, "a") as f:
            f.write("\n\n---\n\n## Saved Figures\n\n")
            for fig_path in figs:
                rel = fig_path.relative_to(scene_dir)
                title = fig_path.stem.replace("_", " ").title()
                f.write(f"### {title}\n\n")
                f.write(f"![{title}]({rel})\n\n")

    print(f"Report    → {out_md}")
    files_dir = scene_dir / f"{out_stem}_files"
    if files_dir.exists():
        n = len(list(files_dir.glob("*")))
        print(f"Cell figs → {files_dir}  ({n} files)")

    return out_md
