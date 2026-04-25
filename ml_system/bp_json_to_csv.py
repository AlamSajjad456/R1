import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Iterable, List

try:
    import numpy as np
    import pandas as pd
except ImportError as exc:  # pragma: no cover
    missing = getattr(exc, "name", "required package")
    raise SystemExit(
        f"Missing dependency '{missing}'. Install project requirements before running bp_json_to_csv.py."
    ) from exc

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from ml_system.utils import setup_logging


SIGNAL_KEYS = ["data_PPG", "data_ECG", "data_PCG", "data_FSR"]
PEAK_KEYS = {"data_PPG", "data_ECG"}


def _compute_features(segment: np.ndarray) -> Dict[str, float]:
    if segment.size == 0:
        return {
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "median": np.nan,
            "p25": np.nan,
            "p75": np.nan,
            "iqr": np.nan,
            "rms": np.nan,
            "ptp": np.nan,
            "energy": np.nan,
            "skew": np.nan,
            "kurtosis": np.nan,
            "slope_mean": np.nan,
            "slope_std": np.nan,
        }
    p25 = float(np.percentile(segment, 25))
    p75 = float(np.percentile(segment, 75))
    std = float(np.std(segment))
    if std == 0.0:
        skew = 0.0
        kurtosis = 0.0
    else:
        centered = segment - float(np.mean(segment))
        m2 = float(np.mean(centered ** 2))
        m3 = float(np.mean(centered ** 3))
        m4 = float(np.mean(centered ** 4))
        skew = m3 / (m2 ** 1.5) if m2 > 0 else 0.0
        kurtosis = m4 / (m2 ** 2) - 3.0 if m2 > 0 else 0.0
    diffs = np.diff(segment) if segment.size > 1 else np.asarray([], dtype=float)
    slope_mean = float(np.mean(diffs)) if diffs.size else 0.0
    slope_std = float(np.std(diffs)) if diffs.size else 0.0
    return {
        "mean": float(np.mean(segment)),
        "std": std,
        "min": float(np.min(segment)),
        "max": float(np.max(segment)),
        "median": float(np.median(segment)),
        "p25": p25,
        "p75": p75,
        "iqr": p75 - p25,
        "rms": float(np.sqrt(np.mean(segment ** 2))),
        "ptp": float(np.ptp(segment)),
        "energy": float(np.mean(segment ** 2)),
        "skew": float(skew),
        "kurtosis": float(kurtosis),
        "slope_mean": slope_mean,
        "slope_std": slope_std,
    }

def _compute_peak_features(segment: np.ndarray) -> Dict[str, float]:
    if segment.size < 3:
        return {
            "peak_count": 0.0,
            "peak_interval_mean": np.nan,
            "peak_interval_std": np.nan,
            "peak_amp_mean": np.nan,
        }
    mean = float(np.mean(segment))
    std = float(np.std(segment))
    thresh = mean + 0.5 * std
    peaks = []
    for i in range(1, segment.size - 1):
        if segment[i] > thresh and segment[i] > segment[i - 1] and segment[i] > segment[i + 1]:
            peaks.append(i)
    if len(peaks) < 2:
        return {
            "peak_count": float(len(peaks)),
            "peak_interval_mean": np.nan,
            "peak_interval_std": np.nan,
            "peak_amp_mean": float(np.mean(segment[peaks])) if peaks else np.nan,
        }
    intervals = np.diff(np.asarray(peaks, dtype=float))
    return {
        "peak_count": float(len(peaks)),
        "peak_interval_mean": float(np.mean(intervals)),
        "peak_interval_std": float(np.std(intervals)),
        "peak_amp_mean": float(np.mean(segment[peaks])),
    }


def _segment_signal(values: Iterable[int], segments: int) -> List[np.ndarray]:
    arr = np.asarray(list(values), dtype=float)
    if segments <= 0:
        return []
    return [seg.astype(float, copy=False) for seg in np.array_split(arr, segments)]

def _normalize_signal(values: Iterable[int]) -> List[float]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return []
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    if std == 0.0:
        return (arr - mean).tolist()
    return ((arr - mean) / std).tolist()

def _window_segment(segment: np.ndarray, window_size: int, stride: int) -> List[np.ndarray]:
    if segment.size == 0:
        return []
    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size and stride must be positive.")
    if segment.size <= window_size:
        return [segment]
    windows: List[np.ndarray] = []
    start = 0
    while start + window_size <= segment.size:
        windows.append(segment[start : start + window_size])
        start += stride
    if not windows:
        windows.append(segment)
    return windows


def _build_rows(
    payload: Dict[str, object],
    source_name: str,
    window_size: int,
    stride: int,
    normalize_subject: bool,
    label_mode: str,
) -> List[Dict[str, object]]:
    bp_list = payload.get("data_BP", [])
    if not isinstance(bp_list, list) or len(bp_list) == 0:
        logging.warning("Skipping %s (missing data_BP).", source_name)
        return []

    segment_count = len(bp_list)
    signal_segments: Dict[str, List[np.ndarray]] = {}
    for key in SIGNAL_KEYS:
        raw = payload.get(key, [])
        if not isinstance(raw, list) or len(raw) == 0:
            logging.warning("Skipping %s (missing %s).", source_name, key)
            return []
        if normalize_subject:
            raw = _normalize_signal(raw)
        signal_segments[key] = _segment_signal(raw, segment_count)

    rows: List[Dict[str, object]] = []
    if label_mode == "subject_mean":
        sbp_vals = []
        dbp_vals = []
        for bp in bp_list:
            if isinstance(bp, dict):
                if bp.get("SBP") is not None:
                    sbp_vals.append(bp.get("SBP"))
                if bp.get("DBP") is not None:
                    dbp_vals.append(bp.get("DBP"))
        sbp_mean = float(np.mean(sbp_vals)) if sbp_vals else np.nan
        dbp_mean = float(np.mean(dbp_vals)) if dbp_vals else np.nan
        full_windows: Dict[str, List[np.ndarray]] = {}
        for key, segments in signal_segments.items():
            full_signal = np.concatenate(segments) if segments else np.asarray([], dtype=float)
            full_windows[key] = _window_segment(full_signal, window_size, stride)
        window_count = max((len(w) for w in full_windows.values()), default=0)
        for w_idx in range(window_count):
            row: Dict[str, object] = {
                "uid": payload.get("UID"),
                "age": payload.get("age"),
                "weight": payload.get("weight"),
                "height": payload.get("height"),
                "segment_index": -1,
                "segment_count": segment_count,
                "window_index": w_idx,
                "window_count": window_count,
                "source_file": source_name,
                "SBP": sbp_mean,
                "DBP": dbp_mean,
            }
            for key, windows in full_windows.items():
                segment_window = windows[w_idx] if w_idx < len(windows) else np.asarray([], dtype=float)
                feats = _compute_features(segment_window)
                if key in PEAK_KEYS:
                    feats.update(_compute_peak_features(segment_window))
                for feat_name, feat_value in feats.items():
                    row[f"{key}_{feat_name}"] = feat_value
            rows.append(row)
        return rows
    for idx, bp in enumerate(bp_list):
        windows_per_signal: Dict[str, List[np.ndarray]] = {}
        for key, segments in signal_segments.items():
            segment = segments[idx] if idx < len(segments) else np.asarray([], dtype=float)
            windows_per_signal[key] = _window_segment(segment, window_size, stride)

        window_count = max((len(w) for w in windows_per_signal.values()), default=0)
        for w_idx in range(window_count):
            row: Dict[str, object] = {
                "uid": payload.get("UID"),
                "age": payload.get("age"),
                "weight": payload.get("weight"),
                "height": payload.get("height"),
                "segment_index": idx,
                "segment_count": segment_count,
                "window_index": w_idx,
                "window_count": window_count,
                "source_file": source_name,
            }

            if isinstance(bp, dict):
                row["SBP"] = bp.get("SBP")
                row["DBP"] = bp.get("DBP")
            else:
                row["SBP"] = np.nan
                row["DBP"] = np.nan

            for key, windows in windows_per_signal.items():
                segment_window = windows[w_idx] if w_idx < len(windows) else np.asarray([], dtype=float)
                feats = _compute_features(segment_window)
                if key in PEAK_KEYS:
                    feats.update(_compute_peak_features(segment_window))
                for feat_name, feat_value in feats.items():
                    row[f"{key}_{feat_name}"] = feat_value

            rows.append(row)

    return rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert BP JSON files into a tabular CSV with segment features."
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Directory containing BP JSON files.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output CSV path.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Optional cap on number of JSON files processed (0 = all).",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=2000,
        help="Window size in samples for each BP segment.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1000,
        help="Stride in samples between windows.",
    )
    parser.add_argument(
        "--normalize-subject",
        action="store_true",
        help="Z-score normalize each subject's signals before feature extraction.",
    )
    parser.add_argument(
        "--label-mode",
        choices=["segment", "subject_mean"],
        default="segment",
        help="Labeling strategy: segment aligns BP to segments, subject_mean uses mean BP for all windows.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Drop non-feature metadata columns (indices, counts, source_file).",
    )
    return parser


def main() -> None:
    setup_logging()
    args = build_arg_parser().parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise SystemExit(f"Data directory not found: {data_dir}")

    files = sorted(data_dir.glob("*.json"))
    if args.max_files and args.max_files > 0:
        files = files[: args.max_files]

    if not files:
        raise SystemExit("No JSON files found.")

    rows: List[Dict[str, object]] = []
    for path in files:
        try:
            with path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as exc:
            logging.warning("Skipping %s (%s).", path.name, exc)
            continue

        if not isinstance(payload, dict):
            logging.warning("Skipping %s (unexpected JSON structure).", path.name)
            continue

        rows.extend(
            _build_rows(
                payload,
                path.name,
                args.window_size,
                args.stride,
                args.normalize_subject,
                args.label_mode,
            )
        )

    if not rows:
        raise SystemExit("No rows were generated. Check input files.")

    df = pd.DataFrame(rows)
    if args.clean:
        drop_cols = [
            "segment_index",
            "segment_count",
            "window_index",
            "window_count",
            "source_file",
        ]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logging.info("Saved %s rows to %s", len(df), out_path)


if __name__ == "__main__":
    main()
