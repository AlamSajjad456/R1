"""
Generate a single-file LaTeX thesis for easier reading.

It inlines:
- miktex_test.tex (root)
- r1_content.tex
- project_details.tex
- thesis_assets/roc_xgboost.tex
- thesis_assets/threshold_tradeoff.tex
- thesis_assets/feature_importance_xgb.tex

Usage:
  ..\\r1\\Scripts\\python.exe make_thesis_single.py
"""

from __future__ import annotations

from pathlib import Path


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def main() -> None:
    root = Path("miktex_test.tex")
    out_path = Path("thesis_single.tex")

    main_tex = _read(root)
    r1 = _read(Path("r1_content.tex"))
    proj = _read(Path("project_details.tex"))

    # Inline common assets referenced by r1_content.tex
    assets: dict[str, str] = {}
    for p in [
        Path("thesis_assets/roc_xgboost.tex"),
        Path("thesis_assets/threshold_tradeoff.tex"),
        Path("thesis_assets/feature_importance_xgb.tex"),
    ]:
        if p.exists():
            assets[str(p).replace("\\", "/")] = _read(p).rstrip()

    for rel, body in assets.items():
        r1 = r1.replace(
            f"\\input{{{rel}}}",
            "\n% ==== BEGIN INLINE: " + rel + " ====\n" + body + "\n% ==== END INLINE: " + rel + " ====\n",
        )

    merged = main_tex
    merged = merged.replace(
        "\\input{r1_content.tex}",
        "\n% ==================================================\n% BEGIN INLINE: r1_content.tex\n% ==================================================\n"
        + r1.rstrip()
        + "\n% ==================================================\n% END INLINE: r1_content.tex\n% ==================================================\n",
    )
    merged = merged.replace(
        "\\input{project_details.tex}",
        "\n% ==================================================\n% BEGIN INLINE: project_details.tex\n% ==================================================\n"
        + proj.rstrip()
        + "\n% ==================================================\n% END INLINE: project_details.tex\n% ==================================================\n",
    )

    out_path.write_text(merged, encoding="utf-8")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()

