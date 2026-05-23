#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/SNNODEATT

echo "=== Server path ==="
pwd

echo "=== Git status ==="
git status
git remote -v
git branch
git log --oneline -5

echo "=== GitHub SSH auth ==="
ssh -T git@github.com || true

echo "=== Fetch and pull ==="
git fetch origin
git pull origin main --rebase

mkdir -p docs/link_checks

cat > docs/link_checks/server_to_github_check.md <<'EOR'
# Server to GitHub Link Check

This file is generated on the server to verify that the server can pull from and push to GitHub.

- Link: Server -> GitHub
- Repo: git@github.com:kassa-ch/SNN-ODE-ATT.git
- Server project path: /root/autodl-tmp/SNNODEATT/
- Purpose: verify GitHub-centered workflow without local file transfer.
EOR

date -u >> docs/link_checks/server_to_github_check.md

echo "=== Sensitive scan ==="
token_marker="GITHUB_""TOKEN"
pat_marker="gh""p_"
private_key_marker="BEGIN OPENSSH PRIVATE"" KEY"
grep -R "${token_marker}\|${pat_marker}\|${private_key_marker}" -n . \
  --exclude-dir=.git \
  --exclude="server_to_github_check.md" || true

echo "=== Large files >50MB ==="
find . -type f -size +50M -not -path "./.git/*" -print

git add docs/link_checks/server_to_github_check.md
git commit -m "Add server to GitHub link check" || true
git push origin main

echo "=== Done. Latest commit ==="
git rev-parse HEAD
