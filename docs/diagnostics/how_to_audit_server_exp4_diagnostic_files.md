# How to Audit Server-side Exp4 Diagnostic Files

This document describes how to safely inspect untracked exp4 diagnostic files on the server.

## Purpose

Before running a new formal `exp4_score_diagnostic`, the server may already contain untracked diagnostic files:

- `docs/exp4_score_diagnostic_report.md`
- `experiments/anomaly_score_benchmark/run_score_diagnostic.py`
- `experiments/anomaly_score_benchmark/score_diagnostic_exp4.csv`

These files may be useful previous results or incomplete leftovers. They should be audited and backed up before they are overwritten or replaced by canonical GitHub code.

## Server command

Run on the server:

```bash
cd /root/autodl-tmp/SNNODEATT
git pull origin main
bash scripts/ops/audit_server_exp4_diagnostic_files.sh
```

The script will:

- record the current Git status;
- inspect the target diagnostic files if present;
- copy them to `/root/autodl-tmp/SNNODEATT_SERVER_BACKUPS/`;
- generate `docs/diagnostics/server_exp4_untracked_diagnostic_audit.md`;
- generate `docs/diagnostics/server_exp4_untracked_diagnostic_audit_files.txt`;
- commit and push only the audit report and file list.

## What it does not do

- It does not run model training.
- It does not run formal exp4 diagnostics.
- It does not commit raw data, checkpoints, caches, or large output directories.
- It does not decide whether the untracked diagnostic files are canonical results.

After this audit, review the report before deciding whether to integrate, discard, or regenerate those diagnostic files.
