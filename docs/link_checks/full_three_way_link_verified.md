# Full Three-way GitHub-centered Link Verified

The GitHub-centered workflow has been verified:

Local Codex -> GitHub -> Server -> GitHub -> Local Codex

Verified meaning:
- Local Codex can push code and small docs to GitHub.
- Server can pull the latest code from GitHub.
- Server can generate a small report and push it back to GitHub.
- Local Codex can pull the server-generated report from GitHub.
- Direct Local Codex -> Server file transfer is not required for code and small-result exchange.

Recommended workflow:
1. Local Codex modifies code/config/docs and pushes to GitHub.
2. Server pulls GitHub updates and runs experiments.
3. Server commits small summaries/reports to GitHub.
4. Local Codex pulls reports from GitHub for review.
5. Large datasets, checkpoints, caches, and raw results remain on the server.

2026-05-23 12:02:50 UTC
