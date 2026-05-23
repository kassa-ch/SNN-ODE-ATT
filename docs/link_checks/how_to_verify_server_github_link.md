# How to Verify Server-GitHub Link

The local Codex environment cannot SSH to the server directly, so use GitHub as the synchronization center.

On the server, run:

```bash
cd /root/autodl-tmp/SNNODEATT
git pull origin main
bash Scrips/utils/ops/check_server_github_link.sh
```

After the server pushes `docs/link_checks/server_to_github_check.md`, run locally:

```bash
git pull origin main
ls docs/link_checks
```

If both files are present, the workflow is:

- Local Codex -> GitHub -> Server
- Server -> GitHub -> Local Codex
