import argparse
import csv
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = PROJECT_ROOT / "Results" / "benchmarks" / "distance_ranking" / "public_score_smoke_results.csv"
DEFAULT_METHODS = ["euclidean", "MahalanobisScorer", "Gaussian_W2", "Sliced_Wasserstein"]


def _write_rows(rows):
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["created_at", "mode", "dataset", "model", "distance_method", "import_status", "run_status", "notes"]
    with open(RESULT_PATH, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def check_distance_imports(methods):
    sys.path.insert(0, str(PROJECT_ROOT))
    from Distance.distance_factory import create_distance_calculator

    rows = []
    for dataset in ("HAI", "ST-AWFD"):
        for method in methods:
            try:
                calc = create_distance_calculator(method)
                import_status = f"ok:{calc.__class__.__name__}"
            except Exception as exc:
                import_status = f"failed:{exc}"
            rows.append({
                "created_at": datetime.now().isoformat(),
                "mode": "check-only",
                "dataset": dataset,
                "model": "SNNODEATT",
                "distance_method": method,
                "import_status": import_status,
                "run_status": "not_run",
                "notes": "Distance import check only; no scores generated.",
            })
    _write_rows(rows)
    print(f"[CHECK] wrote {RESULT_PATH}")
    return rows


def maybe_run_small_detection(methods):
    rows = check_distance_imports(methods)
    # The first implementation intentionally avoids fabricating results. Hooked
    # detection runs can be enabled once a smoke checkpoint exists.
    for row in rows:
        row["mode"] = "smoke-ready"
        row["notes"] = "Detection invocation is deferred until a smoke checkpoint is supplied."
    _write_rows(rows)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--distance_method", nargs="*", default=DEFAULT_METHODS)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.check_only:
        check_distance_imports(args.distance_method)
    else:
        maybe_run_small_detection(args.distance_method)


if __name__ == "__main__":
    main()
