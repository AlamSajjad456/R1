import argparse
import csv
import logging
import math
from pathlib import Path
from typing import Dict, List


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Clean cardio_train.csv by removing physiologically implausible outliers."
    )
    parser.add_argument("input_csv", help="Path to cardio_train.csv")
    parser.add_argument("output_csv", help="Path to save cleaned CSV")
    parser.add_argument("--min-height", type=float, default=120.0)
    parser.add_argument("--max-height", type=float, default=220.0)
    parser.add_argument("--min-weight", type=float, default=30.0)
    parser.add_argument("--max-weight", type=float, default=200.0)
    parser.add_argument("--min-ap-hi", type=float, default=70.0)
    parser.add_argument("--max-ap-hi", type=float, default=240.0)
    parser.add_argument("--min-ap-lo", type=float, default=40.0)
    parser.add_argument("--max-ap-lo", type=float, default=160.0)
    parser.add_argument(
        "--min-bmi",
        type=float,
        default=10.0,
        help="Optional BMI lower bound. Set to 0 to disable.",
    )
    parser.add_argument(
        "--max-bmi",
        type=float,
        default=60.0,
        help="Optional BMI upper bound. Set to 0 to disable.",
    )
    parser.add_argument(
        "--min-age",
        type=float,
        default=18.0,
        help="Minimum age in years (age column is in days).",
    )
    parser.add_argument(
        "--max-age",
        type=float,
        default=100.0,
        help="Maximum age in years (age column is in days).",
    )
    parser.add_argument(
        "--min-pulse-pressure",
        type=float,
        default=10.0,
        help="Minimum ap_hi - ap_lo to keep.",
    )
    parser.add_argument(
        "--max-pulse-pressure",
        type=float,
        default=120.0,
        help="Maximum ap_hi - ap_lo to keep.",
    )
    parser.add_argument(
        "--add-features",
        action="store_true",
        help="Add engineered features (age_years, bmi, pulse_pressure, map).",
    )
    parser.add_argument(
        "--iqr-filter",
        action="store_true",
        help="Apply IQR-based outlier removal on continuous columns.",
    )
    parser.add_argument(
        "--iqr-k",
        type=float,
        default=1.5,
        help="IQR multiplier for outlier removal.",
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


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = build_arg_parser().parse_args()

    input_path = Path(args.input_csv)
    output_path = Path(args.output_csv)
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    removed: Dict[str, int] = {
        "height": 0,
        "weight": 0,
        "ap_hi": 0,
        "ap_lo": 0,
        "ap_order": 0,
        "bmi": 0,
        "age": 0,
        "pulse_pressure": 0,
        "gender": 0,
        "cholesterol": 0,
        "gluc": 0,
        "iqr": 0,
    }
    kept_rows: List[List[str]] = []

    with input_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=";")
        header = next(reader)
        rows = [row for row in reader if len(row) == len(header)]

    # Precompute IQR thresholds for continuous columns if requested
    iqr_limits = {}
    if args.iqr_filter:
        cont_cols = ["age", "height", "weight", "ap_hi", "ap_lo"]
        idx_map = {name: idx for idx, name in enumerate(header)}
        for col in cont_cols:
            values = []
            for row in rows:
                try:
                    val = _to_float(row[idx_map[col]])
                except Exception:
                    continue
                if val == val:
                    values.append(val)
            values.sort()
            if not values:
                continue
            q1 = _percentile(values, 0.25)
            q3 = _percentile(values, 0.75)
            iqr = q3 - q1
            lower = q1 - args.iqr_k * iqr
            upper = q3 + args.iqr_k * iqr
            iqr_limits[col] = (lower, upper)
        output_header = header[:]
        if args.add_features:
            output_header.extend(["age_years", "bmi", "pulse_pressure", "map"])
        kept_rows.append(output_header)
        for row in rows:
            try:
                height = _to_float(row[3])
                weight = _to_float(row[4])
                ap_hi = _to_float(row[5])
                ap_lo = _to_float(row[6])
                age_days = _to_float(row[1])
                gender = row[2]
                cholesterol = row[7]
                gluc = row[8]
            except Exception:
                continue

            age_years = age_days / 365.25 if age_days == age_days else float("nan")
            if not (args.min_age <= age_years <= args.max_age):
                removed["age"] += 1
                continue
            if gender not in {"1", "2"}:
                removed["gender"] += 1
                continue
            if cholesterol not in {"1", "2", "3"}:
                removed["cholesterol"] += 1
                continue
            if gluc not in {"1", "2", "3"}:
                removed["gluc"] += 1
                continue

            if not (args.min_height <= height <= args.max_height):
                removed["height"] += 1
                continue
            if not (args.min_weight <= weight <= args.max_weight):
                removed["weight"] += 1
                continue
            if not (args.min_ap_hi <= ap_hi <= args.max_ap_hi):
                removed["ap_hi"] += 1
                continue
            if not (args.min_ap_lo <= ap_lo <= args.max_ap_lo):
                removed["ap_lo"] += 1
                continue
            if not (ap_hi > ap_lo):
                removed["ap_order"] += 1
                continue
            pulse_pressure = ap_hi - ap_lo
            if not (args.min_pulse_pressure <= pulse_pressure <= args.max_pulse_pressure):
                removed["pulse_pressure"] += 1
                continue
            if args.iqr_filter:
                outlier = False
                for col, (lower, upper) in iqr_limits.items():
                    if col == "age":
                        val = age_days
                    elif col == "height":
                        val = height
                    elif col == "weight":
                        val = weight
                    elif col == "ap_hi":
                        val = ap_hi
                    elif col == "ap_lo":
                        val = ap_lo
                    else:
                        continue
                    if val < lower or val > upper:
                        outlier = True
                        break
                if outlier:
                    removed["iqr"] += 1
                    continue

            if args.min_bmi > 0 and args.max_bmi > 0:
                height_m = height / 100.0
                bmi = weight / (height_m * height_m)
                if not (args.min_bmi <= bmi <= args.max_bmi):
                    removed["bmi"] += 1
                    continue

            if args.add_features:
                height_m = height / 100.0
                bmi = weight / (height_m * height_m)
                pulse_pressure = ap_hi - ap_lo
                map_val = ap_lo + pulse_pressure / 3.0
                row = row + [
                    f"{age_years:.4f}",
                    f"{bmi:.4f}",
                    f"{pulse_pressure:.4f}",
                    f"{map_val:.4f}",
                ]
            kept_rows.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerows(kept_rows)

    total = len(kept_rows) - 1
    logging.info("Saved cleaned file to: %s", output_path)
    logging.info("Rows kept: %s", total)
    logging.info("Removed counts: %s", removed)


if __name__ == "__main__":
    main()
