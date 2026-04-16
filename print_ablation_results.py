#!/usr/bin/env python3
"""
print_ablation_results.py — Parse all ablation results and print a comparison table.

Reads every ``localization_summary.json`` found under the given result directory,
builds a comparison table, and sorts it by the primary metric (wKNN MAE by
default so you can quickly see which feature subset helps most).

Usage
-----
    python3 print_ablation_results.py <result_dir> [--sort-by <metric>] [--csv <file>] [--markdown <file>] [--pdf <file>]

    result_dir  — directory produced by run-feature-test.sh
                  (contains one sub-folder per tested combination)
    --sort-by   — column to sort by: wknn_mae | nn_reg_mae | cnn_mae (default: wknn_mae)
    --csv       — optionally write the table to a CSV file as well
    --markdown  — optionally write the table to a Markdown file
    --pdf       — optionally convert Markdown to PDF

Output columns
--------------
    Combination   — combination name
    N_feat        — number of feature columns
    Features ON   — enabled feature groups (abbreviated)
    wKNN_MAE      — wKNN Mean Absolute Error (m)
    wKNN_RMSE     — wKNN Root Mean Square Error (m)
    NNReg_MAE     — NN Regression MAE (m)
    NNReg_RMSE    — NN Regression RMSE (m)
    CNN_MAE       — CNN Regression MAE (m)
    CNN_RMSE      — CNN Regression RMSE (m)
    Best_MAE      — lowest MAE across all enabled methods
    Best_method   — which method achieved the best MAE
"""

import argparse
import glob
import json
import sys
from pathlib import Path

# Short abbreviations for feature group names (keeps columns narrow)
_SHORT = {
    "ofdm_mag_gd":     "OFDM",
    "tdoa":            "TDoA",
    "aoa":             "AoA",
    "rss":             "RSS",
    "path_loss":       "PL",
    "delay":           "Dly",
    "cov_eigenvalues": "CovE",
    "reached_flags":   "Rch",
}
_ALL_ORDER = list(_SHORT.keys())


def _feat_abbrev(enabled: list[str]) -> str:
    """Return a compact abbreviation string, e.g. 'OFDM+TDoA+AoA'."""
    return "+".join(_SHORT[f] for f in _ALL_ORDER if f in enabled) or "—"


def load_results(result_dir: str) -> list[dict]:
    """Load all localization_summary.json files under *result_dir*."""
    pattern = str(Path(result_dir) / "**" / "localization_summary.json")
    files   = sorted(glob.glob(pattern, recursive=True))
    if not files:
        # Also try direct children (flat layout)
        pattern2 = str(Path(result_dir) / "*" / "localization_summary.json")
        files = sorted(glob.glob(pattern2))
    if not files:
        print(f"No localization_summary.json files found under {result_dir}")
        sys.exit(1)

    rows = []
    for path in files:
        with open(path) as fh:
            data = json.load(fh)
        # Back-fill combo_name from directory name if absent
        if not data.get("combo_name"):
            data["combo_name"] = Path(path).parent.name
        rows.append(data)
    return rows


def build_table(rows: list[dict]) -> list[dict]:
    """Convert raw JSON summaries to flat table dicts."""
    table = []
    for row in rows:
        methods  = row.get("methods", {})
        wknn     = methods.get("WKNN (IDW)",      {})
        c2f      = methods.get("WKNN C2F",        {})
        nn_reg   = methods.get("NN Regression",   {})
        cnn_reg  = methods.get("CNN Regression",  {})

        # Collect all available MAEs to find the best
        all_maes = {name: m["mae"] for name, m in methods.items() if "mae" in m}
        best_mae    = min(all_maes.values()) if all_maes else float("nan")
        best_method = min(all_maes, key=all_maes.get) if all_maes else "—"

        table.append({
            "combo":       row.get("combo_name", "?"),
            "n_feat":      row.get("n_features", "?"),
            "feat_abbrev": _feat_abbrev(row.get("enabled_features", [])),
            "wknn_mae":    wknn.get("mae",    float("nan")),
            "wknn_rmse":   wknn.get("rmse",   float("nan")),
            "wknn_p90":    wknn.get("p90",    float("nan")),
            "c2f_mae":     c2f.get("mae",     float("nan")),
            "c2f_rmse":    c2f.get("rmse",    float("nan")),
            "c2f_p90":     c2f.get("p90",     float("nan")),
            "nn_mae":      nn_reg.get("mae",   float("nan")),
            "nn_rmse":     nn_reg.get("rmse",  float("nan")),
            "nn_p90":      nn_reg.get("p90",   float("nan")),
            "cnn_mae":     cnn_reg.get("mae",  float("nan")),
            "cnn_rmse":    cnn_reg.get("rmse", float("nan")),
            "cnn_p90":     cnn_reg.get("p90",  float("nan")),
            "best_mae":    best_mae,
            "best_method": best_method,
        })
    return table


def _fmt(v) -> str:
    """Format a float metric, or '-' if NaN."""
    try:
        f = float(v)
        return f"{f:6.2f}" if not __import__("math").isnan(f) else "  —   "
    except (TypeError, ValueError):
        return str(v)


def print_table(table: list[dict], sort_by: str = "wknn_mae") -> None:
    """Print the comparison table to stdout, sorted by *sort_by*."""
    sort_key = sort_by.lower().replace("-", "_")
    if sort_key not in table[0]:
        print(f"Unknown sort key '{sort_by}'. Valid: wknn_mae, c2f_mae, nn_mae, cnn_mae, best_mae")
        sort_key = "wknn_mae"

    def _sort_val(r):
        v = r.get(sort_key, float("nan"))
        try:
            f = float(v)
            return f if not __import__("math").isnan(f) else 9999.0
        except (TypeError, ValueError):
            return 9999.0

    table_s = sorted(table, key=_sort_val)

    # Column widths
    combo_w = max(len(r["combo"]) for r in table_s) + 2
    feat_w  = max(len(r["feat_abbrev"]) for r in table_s) + 2
    combo_w = max(combo_w, 22)
    feat_w  = max(feat_w,  20)

    # Header
    hdr = (
        f"{'Combination':<{combo_w}} {'N_feat':>6}  "
        f"{'Features ON':<{feat_w}}  "
        f"{'wKNN':^18}  {'wKNN C2F':^18}  {'NN Reg':^18}  {'CNN Reg':^18}  "
        f"{'Best_MAE':>8}  {'Best_method'}"
    )
    sub = (
        f"{'':<{combo_w}} {'':>6}  "
        f"{'':>{feat_w}}  "
        f"{'MAE':>6} {'RMSE':>6} {'P90':>6}  "
        f"{'MAE':>6} {'RMSE':>6} {'P90':>6}  "
        f"{'MAE':>6} {'RMSE':>6} {'P90':>6}  "
        f"{'MAE':>6} {'RMSE':>6} {'P90':>6}  "
        f"{'':>8}  {''}"
    )
    sep = "─" * len(hdr)

    print()
    print("╔" + "═" * (len(sep) - 2) + "╗")
    print("║" + " FEATURE ABLATION RESULTS ".center(len(sep) - 2) + "║")
    print("╚" + "═" * (len(sep) - 2) + "╝")
    print(f"  Sorted by: {sort_by}  (best → worst)\n")
    print(hdr)
    print(sub)
    print(sep)

    best_overall = table_s[0] if table_s else None

    for r in table_s:
        marker = " ◄" if r is best_overall else "  "
        line = (
            f"{r['combo']:<{combo_w}} {r['n_feat']:>6}  "
            f"{r['feat_abbrev']:<{feat_w}}  "
            f"{_fmt(r['wknn_mae'])} {_fmt(r['wknn_rmse'])} {_fmt(r['wknn_p90'])}  "
            f"{_fmt(r['c2f_mae'])} {_fmt(r['c2f_rmse'])} {_fmt(r['c2f_p90'])}  "
            f"{_fmt(r['nn_mae'])}  {_fmt(r['nn_rmse'])}  {_fmt(r['nn_p90'])}  "
            f"{_fmt(r['cnn_mae'])}  {_fmt(r['cnn_rmse'])}  {_fmt(r['cnn_p90'])}  "
            f"{_fmt(r['best_mae']):>8}  {r['best_method']}{marker}"
        )
        print(line)

    print(sep)
    print(f"\n  ◄ = best overall by '{sort_by}'")

    # Print a quick improvement summary vs baseline
    baseline = next((r for r in table_s if r["combo"] in ("ALL", "all", "baseline")), None)
    if baseline and len(table_s) > 1:
        print("\n  Δ vs ALL baseline (negative = improvement):")
        b_mae = float(baseline.get("best_mae", "nan"))
        for r in table_s:
            if r is baseline:
                continue
            r_mae = float(r.get("best_mae", "nan"))
            import math
            if not math.isnan(b_mae) and not math.isnan(r_mae):
                delta = r_mae - b_mae
                sign  = "+" if delta > 0 else ""
                print(f"    {r['combo']:<{combo_w}}  {sign}{delta:+.2f} m")


def write_csv(table: list[dict], csv_path: str) -> None:
    """Write the table to a CSV file."""
    import csv
    fieldnames = [
        "combo", "n_feat", "feat_abbrev",
        "wknn_mae", "wknn_rmse", "wknn_p90",
        "c2f_mae",  "c2f_rmse",  "c2f_p90",
        "nn_mae", "nn_rmse", "nn_p90",
        "cnn_mae", "cnn_rmse", "cnn_p90",
        "best_mae", "best_method",
    ]
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(table)
    print(f"\nCSV saved → {csv_path}")


def write_markdown(table: list[dict], md_path: str, sort_by: str = "wknn_mae") -> None:
    """Write the table to a Markdown file."""
    sort_key = sort_by.lower().replace("-", "_")
    if sort_key not in table[0]:
        sort_key = "wknn_mae"

    def _sort_val(r):
        v = r.get(sort_key, float("nan"))
        try:
            f = float(v)
            return f if not __import__("math").isnan(f) else 9999.0
        except (TypeError, ValueError):
            return 9999.0

    table_s = sorted(table, key=_sort_val)

    with open(md_path, "w") as fh:
        fh.write("# Feature Ablation Results\n\n")
        fh.write(f"**Sorted by:** {sort_by} (best → worst)\n\n")

        # Markdown table header
        fh.write("| Combination | N_feat | Features ON | wKNN MAE | wKNN RMSE | wKNN P90 | "
                 "C2F MAE | C2F RMSE | C2F P90 | NN MAE | NN RMSE | NN P90 | "
                 "CNN MAE | CNN RMSE | CNN P90 | Best MAE | Best Method |\n")
        fh.write("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")

        best_overall = table_s[0] if table_s else None

        for r in table_s:
            marker = " ◄" if r is best_overall else ""
            fh.write(
                f"| {r['combo']}{marker} | {r['n_feat']} | {r['feat_abbrev']} | "
                f"{_fmt(r['wknn_mae'])} | {_fmt(r['wknn_rmse'])} | {_fmt(r['wknn_p90'])} | "
                f"{_fmt(r['c2f_mae'])} | {_fmt(r['c2f_rmse'])} | {_fmt(r['c2f_p90'])} | "
                f"{_fmt(r['nn_mae'])} | {_fmt(r['nn_rmse'])} | {_fmt(r['nn_p90'])} | "
                f"{_fmt(r['cnn_mae'])} | {_fmt(r['cnn_rmse'])} | {_fmt(r['cnn_p90'])} | "
                f"{_fmt(r['best_mae'])} | {r['best_method']} |\n"
            )

        fh.write(f"\n◄ = best overall by '{sort_by}'\n\n")

        # Improvement summary vs baseline
        baseline = next((r for r in table_s if r["combo"] in ("ALL", "all", "baseline")), None)
        if baseline and len(table_s) > 1:
            fh.write("## Δ vs ALL baseline (negative = improvement)\n\n")
            b_mae = float(baseline.get("best_mae", "nan"))
            for r in table_s:
                if r is baseline:
                    continue
                r_mae = float(r.get("best_mae", "nan"))
                import math
                if not math.isnan(b_mae) and not math.isnan(r_mae):
                    delta = r_mae - b_mae
                    sign = "+" if delta > 0 else ""
                    fh.write(f"- **{r['combo']}**: {sign}{delta:+.2f} m\n")

    print(f"\nMarkdown saved → {md_path}")


def convert_markdown_to_pdf(md_path: str, pdf_path: str) -> None:
    """Convert Markdown file to PDF using markdown2 and weasyprint or pypandoc."""
    try:
        # Try using pypandoc first (most reliable)
        import pypandoc
        pypandoc.convert_file(md_path, "pdf", outputfile=pdf_path)
        print(f"PDF saved → {pdf_path}")
    except ImportError:
        try:
            # Fallback to markdown2 + weasyprint
            import markdown2
            from weasyprint import HTML, CSS

            with open(md_path, "r") as fh:
                md_content = fh.read()

            html_content = markdown2.markdown(md_content, extras=["tables"])
            html = f"""
            <html>
            <head>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1, h2 {{ color: #333; }}
                    table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #4CAF50; color: white; }}
                    tr:nth-child(even) {{ background-color: #f2f2f2; }}
                </style>
            </head>
            <body>
            {html_content}
            </body>
            </html>
            """
            HTML(string=html).write_pdf(pdf_path)
            print(f"PDF saved → {pdf_path}")
        except ImportError as e:
            print(
                f"Error: Could not convert to PDF. Please install 'pypandoc' or 'markdown2' + 'weasyprint':\n"
                f"  pip install pypandoc\n"
                f"  OR\n"
                f"  pip install markdown2 weasyprint\n"
                f"Original error: {e}"
            )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Print feature ablation results comparison table.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("result_dir", help="Directory containing ablation sub-folders")
    parser.add_argument(
        "--sort-by",
        default="wknn_mae",
        help="Sort column (default: wknn_mae). Options: wknn_mae, nn_mae, cnn_mae, best_mae",
    )
    parser.add_argument("--csv", metavar="FILE", help="Also save table as CSV")
    parser.add_argument("--markdown", metavar="FILE", help="Save table as Markdown file")
    parser.add_argument("--pdf", metavar="FILE", help="Convert Markdown to PDF (requires --markdown or auto-generates)")

    args  = parser.parse_args(argv)
    rows  = load_results(args.result_dir)
    table = build_table(rows)
    print_table(table, sort_by=args.sort_by)
    if args.csv:
        write_csv(table, args.csv)
    if args.markdown:
        write_markdown(table, args.markdown, sort_by=args.sort_by)
        if args.pdf:
            convert_markdown_to_pdf(args.markdown, args.pdf)
    elif args.pdf:
        # If PDF is requested but no markdown file specified, create temporary markdown
        md_path = args.pdf.replace(".pdf", ".md")
        write_markdown(table, md_path, sort_by=args.sort_by)
        convert_markdown_to_pdf(md_path, args.pdf)


if __name__ == "__main__":
    main()
