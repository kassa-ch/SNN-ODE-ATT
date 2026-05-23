#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/root/autodl-tmp/SNNODEATT"
TARGET_DIR="${REPO_DIR}/src/snnodeatt/models/baselines"
REPORT_DIR="${REPO_DIR}/docs/model_inventory"
REPORT_MD="${REPORT_DIR}/server_baseline_model_file_inventory.md"
REPORT_JSON="${REPORT_DIR}/server_baseline_model_file_inventory.json"

mkdir -p "${TARGET_DIR}" "${REPORT_DIR}"
cd "${REPO_DIR}"

python - <<'PY'
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from pathlib import Path

repo = Path("/root/autodl-tmp/SNNODEATT")
target_dir = repo / "src/snnodeatt/models/baselines"
report_dir = repo / "docs/model_inventory"
report_md = report_dir / "server_baseline_model_file_inventory.md"
report_json = report_dir / "server_baseline_model_file_inventory.json"

search_roots = [
    Path("/root/autodl-tmp/wafer_att_ctsr_vca"),
    Path("/root/autodl-tmp/SNNODEATT"),
]

exclude_parts = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    "logs",
    "logs_mainline",
    "logs_sinkhorn",
    "results",
    "results_mainline",
    "results_final",
    "results_sinkhorn",
    "cache",
    "cache_sinkhorn",
    "saved_models",
    "saved_models_mainline",
    "models_sinkhorn_train",
    "data",
    "raw",
    "processed",
    "venv",
    ".venv",
}

methods = [
    {
        "category": "linear_prediction",
        "method": "LinearAR",
        "target": "linear_ar.py",
        "patterns": [r"\bLinearAR\b", r"linear[_-]?ar", r"autoregressive", r"\bAR\b"],
    },
    {
        "category": "sparse_linear",
        "method": "LassoAR",
        "target": "lasso_ar.py",
        "patterns": [r"\bLassoAR\b", r"lasso[_-]?ar", r"\bLasso\b", r"L1"],
    },
    {
        "category": "kernel_method",
        "method": "KRR",
        "target": "krr.py",
        "patterns": [r"\bKRR\b", r"kernel ridge", r"KernelRidge", r"kernel[_-]?ridge"],
    },
    {
        "category": "functional_statistics",
        "method": "FPCA",
        "target": "fpca.py",
        "patterns": [r"\bFPCA\b", r"functional PCA", r"FunctionalPCA", r"fPCA"],
    },
    {
        "category": "integral_trend",
        "method": "CumInt",
        "target": "cumulative_integral.py",
        "patterns": [r"\bCumInt\b", r"\bCumlnt\b", r"cumulative[_-]?integral", r"integral[_-]?trend"],
    },
    {
        "category": "spiking_neural_network",
        "method": "SNN",
        "target": "snn.py",
        "patterns": [r"\bSNN\b", r"Spiking", r"LIF", r"membrane"],
    },
    {
        "category": "gated_recurrent_network",
        "method": "GRU",
        "target": "gru.py",
        "patterns": [r"\bGRU\b", r"GRUCell", r"nn\.GRU"],
    },
    {
        "category": "continuous_time_latent_model",
        "method": "Latent ODE",
        "target": "latent_ode.py",
        "patterns": [r"LatentODE", r"Latent ODE", r"latent[_-]?ode"],
    },
    {
        "category": "continuous_time_recurrent_model",
        "method": "ODERNN",
        "target": "ode_rnn.py",
        "patterns": [r"\bODERNN\b", r"ODE[-_]?RNN", r"ode[_-]?rnn"],
    },
    {
        "category": "probabilistic_latent_factor_model",
        "method": "DF2M",
        "target": "df2m.py",
        "patterns": [r"\bDF2M\b", r"Deep Functional Factor", r"functional factor"],
    },
    {
        "category": "paper_method",
        "method": "SNN-ODE",
        "target": "snn_ode_baseline.py",
        "patterns": [r"SNN[-_]?ODE", r"SNN_ODE", r"Continuous.*SNN", r"ODEFunc"],
    },
]

def should_skip(path: Path) -> bool:
    parts = set(path.parts)
    if parts & exclude_parts:
        return True
    if target_dir in path.parents:
        return True
    return False

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def line_count(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            return sum(1 for _ in f)
    except Exception:
        return 0

def safe_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""

py_files: list[Path] = []
for root in search_roots:
    if not root.exists():
        continue
    for p in root.rglob("*.py"):
        if p.is_file() and not should_skip(p):
            try:
                if p.stat().st_size <= 2_000_000:
                    py_files.append(p)
            except OSError:
                pass

inventory = []

for spec in methods:
    candidates = []
    compiled = [re.compile(pat, re.IGNORECASE) for pat in spec["patterns"]]
    for p in py_files:
        name = p.name
        rel = str(p)
        text = safe_text(p)
        score = 0
        for rx in compiled:
            if rx.search(name):
                score += 50
            if rx.search(rel):
                score += 10
            matches = len(rx.findall(text))
            score += min(matches, 10)
        # Prefer actual model-looking files over orchestration/eval scripts.
        if re.search(r"class\s+\w+.*\(|torch\.nn|nn\.Module|fit\(|predict\(", text):
            score += 5
        if re.search(r"evaluate|orchestrator|resume|audit|diagnostic|plot|summary", name, re.IGNORECASE):
            score -= 10
        if score > 0:
            candidates.append((score, p))
    candidates.sort(key=lambda x: (x[0], -len(str(x[1]))), reverse=True)
    chosen = candidates[0][1] if candidates else None

    entry = {
        "category": spec["category"],
        "method": spec["method"],
        "target": str(Path("src/snnodeatt/models/baselines") / spec["target"]),
        "status": "missing",
        "source": None,
        "size_bytes": None,
        "lines": None,
        "sha256": None,
        "top_candidates": [
            {"score": score, "path": str(path), "size_bytes": path.stat().st_size}
            for score, path in candidates[:5]
        ],
    }

    if chosen is not None:
        dest = target_dir / spec["target"]
        shutil.copy2(chosen, dest)
        entry.update(
            {
                "status": "copied",
                "source": str(chosen),
                "size_bytes": dest.stat().st_size,
                "lines": line_count(dest),
                "sha256": sha256(dest),
            }
        )
    inventory.append(entry)

report_json.write_text(json.dumps(inventory, indent=2, ensure_ascii=False), encoding="utf-8")

lines = [
    "# Server Baseline Model File Inventory",
    "",
    "This report was generated on the server by `scripts/ops/collect_server_baseline_model_files.sh`.",
    "",
    "The script searches the original project read-only and copies small `.py` model implementations into `src/snnodeatt/models/baselines/`.",
    "",
    "## Summary",
    "",
    "| Category | Method | Status | Target | Source | Lines | Size bytes |",
    "|---|---|---|---|---|---:|---:|",
]
for e in inventory:
    lines.append(
        f"| {e['category']} | {e['method']} | {e['status']} | `{e['target']}` | "
        f"`{e['source'] or ''}` | {e['lines'] or ''} | {e['size_bytes'] or ''} |"
    )

lines.extend(["", "## Candidate Details", ""])
for e in inventory:
    lines.append(f"### {e['method']}")
    lines.append("")
    lines.append(f"- Status: `{e['status']}`")
    lines.append(f"- Target: `{e['target']}`")
    lines.append(f"- Source: `{e['source'] or ''}`")
    lines.append(f"- SHA256: `{e['sha256'] or ''}`")
    lines.append("")
    lines.append("Top candidates:")
    for c in e["top_candidates"]:
        lines.append(f"- score={c['score']} size={c['size_bytes']} path=`{c['path']}`")
    lines.append("")

report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

print(f"Scanned Python files: {len(py_files)}")
print(f"Copied methods: {sum(e['status'] == 'copied' for e in inventory)}/{len(inventory)}")
print(report_md)
print(report_json)
PY

echo "=== Copied baseline files ==="
find "${TARGET_DIR}" -maxdepth 1 -type f -name "*.py" -printf "%f %s bytes\n" | sort

echo "=== Safety scan on copied files ==="
token_marker="GITHUB_""TOKEN"
pat_marker="gh""p_"
private_key_marker="BEGIN OPENSSH PRIVATE"" KEY"
server_host_marker="connect."".bjb"
grep -R "${token_marker}\|${pat_marker}\|${private_key_marker}\|${server_host_marker}" -n "${TARGET_DIR}" "${REPORT_DIR}" || true

echo "=== Large file check for copied files ==="
find "${TARGET_DIR}" "${REPORT_DIR}" -type f -size +2M -print

if grep -R "${token_marker}\|${pat_marker}\|${private_key_marker}\|${server_host_marker}" -n "${TARGET_DIR}" "${REPORT_DIR}" >/tmp/snnodeatt_baseline_sensitive_matches.txt; then
  echo "Sensitive marker found in copied baseline files or reports. Not committing."
  cat /tmp/snnodeatt_baseline_sensitive_matches.txt
  exit 2
fi

if find "${TARGET_DIR}" "${REPORT_DIR}" -type f -size +2M | grep . >/tmp/snnodeatt_baseline_large_files.txt; then
  echo "Large copied file found. Not committing."
  cat /tmp/snnodeatt_baseline_large_files.txt
  exit 3
fi

git status --short
git add src/snnodeatt/models/baselines docs/model_inventory
git commit -m "Collect baseline model files from server" || true
git push origin main

echo "=== Latest server commit ==="
git rev-parse HEAD
