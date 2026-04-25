import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def generate_eda_report(df: pd.DataFrame, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    missing = df.isna().mean().sort_values(ascending=False)
    missing_df = missing.reset_index()
    missing_df.columns = ["column", "missing_ratio"]

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    describe_df = df[numeric_cols].describe().T if numeric_cols else pd.DataFrame()

    corr_path = out_dir / "correlation.png"
    if len(numeric_cols) >= 2:
        plt.figure(figsize=(10, 8))
        corr = df[numeric_cols].corr()
        sns.heatmap(corr, cmap="viridis", linewidths=0.5)
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(corr_path, dpi=200)
        plt.close()

    html_path = out_dir / "eda_report.html"
    with html_path.open("w", encoding="utf-8") as f:
        f.write("<html><head><title>EDA Report</title></head><body>")
        f.write("<h1>Auto EDA Report</h1>")
        f.write("<h2>Missing Values</h2>")
        f.write(missing_df.to_html(index=False))
        if not describe_df.empty:
            f.write("<h2>Numeric Summary</h2>")
            f.write(describe_df.to_html())
        if corr_path.exists():
            f.write("<h2>Correlations</h2>")
            f.write(f'<img src="{corr_path.name}" width="900"/>')
        f.write("</body></html>")

    return html_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate an EDA report for a CSV file.")
    parser.add_argument("csv_path", help="Path to CSV file.")
    parser.add_argument("--out-dir", default="ml_system/reports", help="Output folder.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    df = pd.read_csv(args.csv_path, low_memory=False)
    report_path = generate_eda_report(df, Path(args.out_dir))
    print(f"EDA report written to {report_path}")


if __name__ == "__main__":
    main()
