# Exp4 Untracked Diagnostic Decision

## 1. Source Audit

- Audit report path: `docs/diagnostics/server_exp4_untracked_diagnostic_audit.md`
- Audit file list path: `docs/diagnostics/server_exp4_untracked_diagnostic_audit_files.txt`
- Server commit hash: `49e7024cd11642f367960a039cf5b7cfb8d48246`
- Backup directory path: `/root/autodl-tmp/SNNODEATT_SERVER_BACKUPS/exp4_diagnostic_untracked_20260523_121527`

## 2. File-by-file Assessment

| File | Exists | Size | Lines | Sensitive scan | Completeness | Decision |
|---|---:|---:|---:|---|---|---|
| `docs/exp4_score_diagnostic_report.md` | yes | 17926 bytes | 123 | empty in file-level scan | Complete report structure is present: Executive Summary, Data and Cache, Direction Diagnostic, Threshold Diagnostic, Score Distribution, Local Reference Diagnostic, Comparison with Baselines, Root Cause, Decision, and Next Action. One Chinese decision sentence appears mojibake-encoded, but the metrics and conclusions are readable. | integrate |
| `experiments/anomaly_score_benchmark/run_score_diagnostic.py` | yes | 19699 bytes | 396 | empty in file-level scan | Looks like a complete runnable diagnostic script with imports, method registry use, quantile scan, AUC/PR-AUC direction checks, metrics helpers, local methods, and report generation logic. It has `argparse`/main-style structure in the audited preview and is not just a stub. | integrate |
| `experiments/anomaly_score_benchmark/score_diagnostic_exp4.csv` | yes | 16346 bytes | 41 | empty in file-level scan | Contains the expected diagnostic columns: method, direction, quantile, threshold, Accuracy/Precision/Recall/F1/F2, TP/FP/TN/FN, AUC/PR-AUC, and normal/abnormal score distribution summaries. It appears to be real exp4 diagnostic output rather than toy data. | integrate |

## 3. Findings

- All three target files exist on the server and were backed up before any integration decision.
- The audit report includes a concrete backup directory and SHA256 checksums for all target files.
- The report file is structurally complete and includes the expected diagnostic sections.
- The diagnostic script appears complete enough to preserve and integrate.
- The CSV contains real exp4 diagnostic metrics and the expected result columns.
- File-level sensitive scans for all three target files are empty.
- Repository-level sensitive scan in the audit report is empty.
- Repository-level large-file check in the audit report is empty.
- These files are likely valid previous diagnostic outputs rather than abandoned half-finished artifacts.
- The only quality issue found is mojibake text in one Chinese summary line of the report; that should be corrected or normalized when integrating, but it does not justify discarding the result.

## 4. Recommendation

Integrate existing server diagnostic files into GitHub, then analyze their metrics before rerunning.

Rationale:

- The script, report, and CSV are mutually consistent.
- The report already includes the requested direction, threshold, distribution, local reference, baseline comparison, root-cause, and decision sections.
- The CSV contains the expected method/direction/quantile metric grid.
- No sensitive information or large files were found by the server audit.
- Rerunning immediately would risk overwriting useful, already-backed-up diagnostic evidence.

## 5. Next Action

Ask the server to restore the backed-up files into the repository and commit/push them as a small integration commit:

```bash
cd /root/autodl-tmp/SNNODEATT
cp -a /root/autodl-tmp/SNNODEATT_SERVER_BACKUPS/exp4_diagnostic_untracked_20260523_121527/docs/exp4_score_diagnostic_report.md docs/exp4_score_diagnostic_report.md
cp -a /root/autodl-tmp/SNNODEATT_SERVER_BACKUPS/exp4_diagnostic_untracked_20260523_121527/experiments/anomaly_score_benchmark/run_score_diagnostic.py experiments/anomaly_score_benchmark/run_score_diagnostic.py
cp -a /root/autodl-tmp/SNNODEATT_SERVER_BACKUPS/exp4_diagnostic_untracked_20260523_121527/experiments/anomaly_score_benchmark/score_diagnostic_exp4.csv experiments/anomaly_score_benchmark/score_diagnostic_exp4.csv
git status --short
git add docs/exp4_score_diagnostic_report.md experiments/anomaly_score_benchmark/run_score_diagnostic.py experiments/anomaly_score_benchmark/score_diagnostic_exp4.csv
git commit -m "Integrate exp4 score diagnostic results"
git push origin main
```

After the server pushes that integration commit, Local Codex should pull it and review:

- whether the report's mojibake sentence should be cleaned in a follow-up documentation-only commit;
- whether the diagnostic metrics are sufficient to close the wafer exp4 score-comparison branch;
- whether any canonical rerun is still needed for reproducibility.

## 6. Do Not Do Yet

- Do not run full exp1~4.
- Do not retrain.
- Do not overwrite server backup.
- Do not commit raw cache or checkpoints.
- Do not regenerate exp4 diagnostics until the existing files have been integrated and reviewed.
