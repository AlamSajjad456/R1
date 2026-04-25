import argparse
import csv
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import math


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Deep audit for cardio_train_clean.csv: stats + outlier detection."
    )
    parser.add_argument("input_csv", help="Path to cardio_train_clean.csv")
    parser.add_argument(
        "--delimiter",
        default=";",
        help="CSV delimiter (default: ;).",
    )
    parser.add_argument(
        "--iqr-k",
        type=float,
        default=1.5,
        help="IQR multiplier for outlier detection.",
    )
    parser.add_argument(
        "--zscore",
        type=float,
        default=4.0,
        help="Z-score threshold for outlier detection.",
    )
    return parser


def _to_float(value: str) -> float:
    return float(value) if value != "" else float("nan")


def _percentile(sorted_vals: List[float], pct: float) -> float:
    if not sorted_vals:
        return float("nan")
    k = (len(sorted_vals) - 1) * pct
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    d0 = sorted_vals[f] * (c - k)
    d1 = sorted_vals[c] * (k - f)
    return d0 + d1


def _mean_std(vals: List[float]) -> Tuple[float, float]:
    if not vals:
        return float("nan"), float("nan")
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    return mean, math.sqrt(var)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = build_arg_parser().parse_args()

    path = Path(args.input_csv)
    if not path.exists():
        raise SystemExit(f"Input file not found: {path}")

    numeric_cols = [
        "age",
        "height",
        "weight",
        "ap_hi",
        "ap_lo",
        "cholesterol",
        "gluc",
        "smoke",
        "alco",
        "active",
    ]

    data: Dict[str, List[float]] = {c: [] for c in numeric_cols}
    total_rows = 0

    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=args.delimiter)
        header = next(reader)
        col_index = {name: idx for idx, name in enumerate(header)}
        missing = [c for c in numeric_cols if c not in col_index]
        if missing:
            raise SystemExit(f"Missing columns: {', '.join(missing)}")

        for row in reader:
            if len(row) != len(header):
                continue
            total_rows += 1
            for col in numeric_cols:
                val = _to_float(row[col_index[col]])
                if val == val:
                    data[col].append(val)

    logging.info("Rows scanned: %s", total_rows)

    stats = {}
    for col, vals in data.items():
        vals_sorted = sorted(vals)
        mean, std = _mean_std(vals_sorted)
        q1 = _percentile(vals_sorted, 0.25)
        q3 = _percentile(vals_sorted, 0.75)
        iqr = q3 - q1
        stats[col] = {
            "count": len(vals_sorted),
            "mean": mean,
            "std": std,
            "min": vals_sorted[0] if vals_sorted else float("nan"),
            "p25": q1,
            "median": _percentile(vals_sorted, 0.50),
            "p75": q3,
            "max": vals_sorted[-1] if vals_sorted else float("nan"),
            "iqr": iqr,
        }

    outliers_iqr = defaultdict(int)
    outliers_z = defaultdict(int)

    for col, vals in data.items():
        col_stats = stats[col]
        q1 = col_stats["p25"]
        q3 = col_stats["p75"]
        iqr = col_stats["iqr"]
        mean = col_stats["mean"]
        std = col_stats["std"]
        lower = q1 - args.iqr_k * iqr
        upper = q3 + args.iqr_k * iqr
        for v in vals:
            if v < lower or v > upper:
                outliers_iqr[col] += 1
            if std > 0 and abs(v - mean) / std > args.zscore:
                outliers_z[col] += 1

    logging.info("Column stats (count/mean/std/min/p25/median/p75/max):")
    for col, s in stats.items():
        logging.info(
            "%s: %s / %.4f / %.4f / %.4f / %.4f / %.4f / %.4f / %.4f",
            col,
            s["count"],
            s["mean"],
            s["std"],
            s["min"],
            s["p25"],
            s["median"],
            s["p75"],
            s["max"],
        )

    logging.info("Outlier counts (IQR k=%s): %s", args.iqr_k, dict(outliers_iqr))
    logging.info("Outlier counts (Z-score > %s): %s", args.zscore, dict(outliers_z))


if __name__ == "__main__":
    main()
