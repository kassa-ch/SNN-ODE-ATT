# Three-way Link Check Summary

Target workflow:

Local Codex -> GitHub -> Server
Server -> GitHub -> Local Codex

Current result:
- Local HTTPS read/clone works.
- Local SSH push is blocked until a local Codex public key is added to GitHub.
- Local non-interactive SSH to the server is unavailable, so direct server commands were not attempted.
- Server-side self-check script is provided at `Scrips/utils/ops/check_server_github_link.sh`.
- Large files should remain on the server.
- Code, configs, docs, and small reports can be exchanged through GitHub after both sides have GitHub SSH auth.
2026-05-23 11:21:37Z
