import argparse
import csv
import gzip
import json
import math
import re
from collections import OrderedDict
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "Data"
MANIFEST_DIR = PROJECT_ROOT / "Results" / "manifests"


def _safe_float(value, default=0.0):
    try:
        if value is None:
            return default
        value = str(value).strip()
        if value == "":
            return default
        out = float(value)
        if not math.isfinite(out):
            return default
        return out
    except Exception:
        return default


def _safe_name(value):
    value = str(value)
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("_") or "sample"


def _write_sample_csv(path, header, rows):
    if path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    return True


def _is_lfs_pointer(path):
    if path.suffix == ".gz":
        return False
    if path.stat().st_size > 1024:
        return False
    try:
        first = path.read_text(encoding="utf-8", errors="ignore").splitlines()[0]
    except Exception:
        return False
    return first.startswith("version https://git-lfs.github.com/spec/")


def _open_text(path):
    if str(path).lower().endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace", newline="")
    return open(path, "r", encoding="utf-8", errors="replace", newline="")


def _discover_hai_files(origin_dir):
    files = []
    for path in sorted(origin_dir.glob("hai-*/*.csv*")):
        if path.name.startswith("label-") or path.name.startswith("summary"):
            continue
        if path.suffix.lower() not in {".csv", ".gz"} and not path.name.lower().endswith(".csv.gz"):
            continue
        if _is_lfs_pointer(path):
            continue
        files.append(path)
    return files


def _scan_hai_features(files):
    features = OrderedDict()
    attack_cols = OrderedDict()

    for path in files:
        with _open_text(path) as handle:
            first = handle.readline()
        delimiter = ";" if first.count(";") >= first.count(",") else ","
        header = next(csv.reader([first], delimiter=delimiter))
        for col in header:
            if col == "time":
                continue
            if col.startswith("attack"):
                attack_cols[col] = None
            else:
                features[col] = None

    return list(features.keys()), list(attack_cols.keys())


def preprocess_hai(data_root, window_size=128, stride=128):
    dataset_dir = data_root / "HAI"
    origin_dir = dataset_dir / "Origin_Datasets"
    out_dir = dataset_dir / "preprocess_data"
    summary = {
        "dataset": "HAI",
        "origin_dir": str(origin_dir),
        "output_dir": str(out_dir),
        "window_size": window_size,
        "stride": stride,
        "files_seen": 0,
        "files_processed": 0,
        "samples_written": 0,
        "samples_skipped_existing": 0,
        "normal_samples": 0,
        "abnormal_samples": 0,
        "feature_dim": 0,
        "notes": [],
    }

    if not origin_dir.exists():
        summary["notes"].append("Origin_Datasets directory not found.")
        return summary

    files = _discover_hai_files(origin_dir)
    summary["files_seen"] = len(files)
    if not files:
        summary["notes"].append("No usable HAI CSV/CSV.GZ files found. Git LFS pointer files are skipped.")
        return summary

    features, attack_cols = _scan_hai_features(files)
    summary["feature_dim"] = len(features)
    if not features:
        summary["notes"].append("No numeric feature columns found.")
        return summary

    out_dir.mkdir(parents=True, exist_ok=True)
    header = features
    sample_index = 0

    for path in files:
        rel = path.relative_to(origin_dir)
        print(f"[HAI] processing {rel}")
        with _open_text(path) as handle:
            first = handle.readline()
            delimiter = ";" if first.count(";") >= first.count(",") else ","
            fieldnames = next(csv.reader([first], delimiter=delimiter))
            reader = csv.DictReader(handle, fieldnames=fieldnames, delimiter=delimiter)

            buffer_rows = []
            buffer_attacks = []
            for row in reader:
                values = [_safe_float(row.get(col)) for col in features]
                attack = any(_safe_float(row.get(col)) > 0.0 for col in attack_cols)
                buffer_rows.append(values)
                buffer_attacks.append(attack)

                if len(buffer_rows) == window_size:
                    buffer_attack = any(buffer_attacks)
                    label = "abnormal" if buffer_attack else "normal"
                    output_name = f"hai_{sample_index:06d}_{label}.csv"
                    written = _write_sample_csv(out_dir / output_name, header, buffer_rows)
                    if written:
                        summary["samples_written"] += 1
                        summary[f"{label}_samples"] += 1
                    else:
                        summary["samples_skipped_existing"] += 1
                    sample_index += 1

                    if stride >= window_size:
                        buffer_rows = []
                        buffer_attacks = []
                    else:
                        buffer_rows = buffer_rows[stride:]
                        buffer_attacks = buffer_attacks[stride:]

        summary["files_processed"] += 1

    return summary


def _scan_st_awfd_features(files):
    features = OrderedDict()
    for path in files:
        with open(path, newline="", encoding="utf-8", errors="replace") as handle:
            reader = csv.reader(handle)
            header = next(reader)
        for col in header:
            if col in {"MaterialID", "is_test", "target"}:
                continue
            if col == "StepID" or col == "duration_ms" or col.startswith("feature_"):
                features[col] = None
    return list(features.keys())


def preprocess_st_awfd(data_root):
    dataset_dir = data_root / "ST-AWFD"
    origin_dir = dataset_dir / "Origin_Datasets"
    out_dir = dataset_dir / "preprocess_data"
    files = sorted(origin_dir.glob("D*/D*.csv")) if origin_dir.exists() else []
    features = _scan_st_awfd_features(files) if files else []

    summary = {
        "dataset": "ST-AWFD",
        "origin_dir": str(origin_dir),
        "output_dir": str(out_dir),
        "files_seen": len(files),
        "files_processed": 0,
        "samples_written": 0,
        "samples_skipped_existing": 0,
        "normal_samples": 0,
        "abnormal_samples": 0,
        "feature_dim": len(features),
        "notes": [],
    }

    if not files:
        summary["notes"].append("No ST-AWFD D*.csv files found.")
        return summary

    out_dir.mkdir(parents=True, exist_ok=True)

    for path in files:
        source = path.stem
        print(f"[ST-AWFD] processing {path.relative_to(origin_dir)}")
        current_mid = None
        current_rows = []
        current_target = 0
        sequence_index = 0

        def flush_group():
            nonlocal sequence_index
            if current_mid is None or not current_rows:
                return
            label = "abnormal" if current_target else "normal"
            name = f"st_awfd_{source}_{_safe_name(current_mid)}_{sequence_index:03d}_{label}.csv"
            written = _write_sample_csv(out_dir / name, features, current_rows)
            if written:
                summary["samples_written"] += 1
                summary[f"{label}_samples"] += 1
            else:
                summary["samples_skipped_existing"] += 1
            sequence_index += 1

        with open(path, newline="", encoding="utf-8", errors="replace") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                mid = row.get("MaterialID")
                if current_mid is not None and mid != current_mid:
                    flush_group()
                    current_rows = []
                    current_target = 0

                current_mid = mid
                current_target = max(current_target, int(_safe_float(row.get("target")) > 0.0))
                current_rows.append([_safe_float(row.get(col)) for col in features])

        flush_group()
        summary["files_processed"] += 1

    return summary


def summarize_dataset_state(data_root):
    state = {}
    for dataset_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        preprocess_dir = dataset_dir / "preprocess_data"
        origin_dir = dataset_dir / "Origin_Datasets"
        csv_count = len(list(preprocess_dir.glob("*.csv"))) if preprocess_dir.exists() else 0
        state[dataset_dir.name] = {
            "has_origin_datasets": origin_dir.exists(),
            "has_preprocess_data": preprocess_dir.exists(),
            "preprocess_csv_count": csv_count,
        }
    return state


def write_manifest(summaries):
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": datetime.now().isoformat(),
        "shape_contract": "Each output CSV is one [T,D] sample. TimeSeriesDataset batches them into [B,T,D].",
        "summaries": summaries,
        "dataset_state": summarize_dataset_state(DATA_ROOT),
    }
    manifest_path = MANIFEST_DIR / "preprocess_origin_datasets_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    return manifest_path


def parse_args():
    parser = argparse.ArgumentParser(description="Convert Origin_Datasets into SNNODEATT-compatible preprocess_data CSVs.")
    parser.add_argument("--datasets", nargs="*", default=["HAI", "ST-AWFD"], help="Datasets to process. Supported: HAI ST-AWFD")
    parser.add_argument("--hai-window", type=int, default=128)
    parser.add_argument("--hai-stride", type=int, default=128)
    return parser.parse_args()


def main():
    args = parse_args()
    summaries = []

    for dataset in args.datasets:
        key = dataset.upper()
        if key == "HAI":
            summaries.append(preprocess_hai(DATA_ROOT, window_size=args.hai_window, stride=args.hai_stride))
        elif key in {"ST-AWFD", "ST_AWFD"}:
            summaries.append(preprocess_st_awfd(DATA_ROOT))
        else:
            summaries.append({
                "dataset": dataset,
                "samples_written": 0,
                "notes": ["Unsupported or intentionally skipped by this script."],
            })

    manifest_path = write_manifest(summaries)
    print(f"[MANIFEST] wrote {manifest_path}")
    print(json.dumps(summaries, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
