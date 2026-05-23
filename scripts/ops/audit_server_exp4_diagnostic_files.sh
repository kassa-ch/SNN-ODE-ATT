#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/root/autodl-tmp/SNNODEATT"
BACKUP_ROOT="/root/autodl-tmp/SNNODEATT_SERVER_BACKUPS"
TS="$(date -u +%Y%m%d_%H%M%S)"
BACKUP_DIR="${BACKUP_ROOT}/exp4_diagnostic_untracked_${TS}"
REPORT_DIR="${REPO_DIR}/docs/diagnostics"
REPORT_MD="${REPORT_DIR}/server_exp4_untracked_diagnostic_audit.md"
REPORT_TXT="${REPORT_DIR}/server_exp4_untracked_diagnostic_audit_files.txt"

mkdir -p "${BACKUP_DIR}"
mkdir -p "${REPORT_DIR}"

cd "${REPO_DIR}"

token_marker="GITHUB_""TOKEN"
pat_marker="gh""p_"
private_key_marker="BEGIN OPENSSH PRIVATE"" KEY"
server_host_marker="connect."".bjb"
sensitive_pattern="${token_marker}\|${pat_marker}\|${private_key_marker}\|${server_host_marker}"

echo "# Server Exp4 Untracked Diagnostic File Audit" > "${REPORT_MD}"
echo "" >> "${REPORT_MD}"
echo "- Timestamp UTC: ${TS}" >> "${REPORT_MD}"
echo "- Repository: ${REPO_DIR}" >> "${REPORT_MD}"
echo "- Backup directory: ${BACKUP_DIR}" >> "${REPORT_MD}"
echo "" >> "${REPORT_MD}"

echo "## Git status before audit" >> "${REPORT_MD}"
echo '```text' >> "${REPORT_MD}"
git status --short >> "${REPORT_MD}" || true
echo '```' >> "${REPORT_MD}"
echo "" >> "${REPORT_MD}"

TARGETS=(
  "docs/exp4_score_diagnostic_report.md"
  "experiments/anomaly_score_benchmark/run_score_diagnostic.py"
  "experiments/anomaly_score_benchmark/score_diagnostic_exp4.csv"
)

echo "## Target files" >> "${REPORT_MD}"
: > "${REPORT_TXT}"

for f in "${TARGETS[@]}"; do
  echo "- ${f}" >> "${REPORT_MD}"
  if [ -f "${f}" ]; then
    echo "${f}" >> "${REPORT_TXT}"

    mkdir -p "${BACKUP_DIR}/$(dirname "${f}")"
    cp -a "${f}" "${BACKUP_DIR}/${f}"

    SIZE="$(wc -c < "${f}" | tr -d ' ')"
    LINES="$(wc -l < "${f}" | tr -d ' ')"
    SHA="$(sha256sum "${f}" | awk '{print $1}')"

    echo "" >> "${REPORT_MD}"
    echo "### ${f}" >> "${REPORT_MD}"
    echo "" >> "${REPORT_MD}"
    echo "- Exists: yes" >> "${REPORT_MD}"
    echo "- Size bytes: ${SIZE}" >> "${REPORT_MD}"
    echo "- Lines: ${LINES}" >> "${REPORT_MD}"
    echo "- SHA256: ${SHA}" >> "${REPORT_MD}"
    echo "- Backed up to: ${BACKUP_DIR}/${f}" >> "${REPORT_MD}"
    echo "" >> "${REPORT_MD}"

    echo "Preview first 80 lines:" >> "${REPORT_MD}"
    echo '```text' >> "${REPORT_MD}"
    sed -n '1,80p' "${f}" >> "${REPORT_MD}" || true
    echo '```' >> "${REPORT_MD}"
    echo "" >> "${REPORT_MD}"

    echo "Sensitive scan for ${f}:" >> "${REPORT_MD}"
    echo '```text' >> "${REPORT_MD}"
    grep -n "${sensitive_pattern}" "${f}" >> "${REPORT_MD}" || true
    echo '```' >> "${REPORT_MD}"
    echo "" >> "${REPORT_MD}"
  else
    echo "" >> "${REPORT_MD}"
    echo "### ${f}" >> "${REPORT_MD}"
    echo "" >> "${REPORT_MD}"
    echo "- Exists: no" >> "${REPORT_MD}"
    echo "" >> "${REPORT_MD}"
  fi
done

echo "## Large file check in repository" >> "${REPORT_MD}"
echo '```text' >> "${REPORT_MD}"
find . -type f -size +50M -not -path "./.git/*" -print >> "${REPORT_MD}" || true
echo '```' >> "${REPORT_MD}"
echo "" >> "${REPORT_MD}"

echo "## Sensitive scan in repository" >> "${REPORT_MD}"
echo '```text' >> "${REPORT_MD}"
grep -R "${sensitive_pattern}" -n . \
  --exclude-dir=.git \
  --exclude="server_exp4_untracked_diagnostic_audit.md" \
  --exclude="server_exp4_untracked_diagnostic_audit_files.txt" >> "${REPORT_MD}" || true
echo '```' >> "${REPORT_MD}"
echo "" >> "${REPORT_MD}"

echo "## Action taken" >> "${REPORT_MD}"
echo "" >> "${REPORT_MD}"
echo "The target untracked diagnostic files, if present, were copied to the backup directory." >> "${REPORT_MD}"
echo "They are not automatically committed as primary diagnostic results in this audit step." >> "${REPORT_MD}"
echo "" >> "${REPORT_MD}"

echo "## Recommendation" >> "${REPORT_MD}"
echo "" >> "${REPORT_MD}"
echo "1. Review this audit report first." >> "${REPORT_MD}"
echo "2. If the untracked diagnostic files are valid, decide whether to integrate them." >> "${REPORT_MD}"
echo "3. If they are obsolete or incomplete, regenerate exp4_score_diagnostic from the canonical GitHub workflow." >> "${REPORT_MD}"
echo "" >> "${REPORT_MD}"

echo "=== Audit report generated ==="
echo "${REPORT_MD}"
echo "=== Backup directory ==="
echo "${BACKUP_DIR}"

git add "${REPORT_MD}" "${REPORT_TXT}"
git commit -m "Add server exp4 untracked diagnostic audit" || true
git push origin main

echo "=== Latest server commit ==="
git rev-parse HEAD
