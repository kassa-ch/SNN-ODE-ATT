# SNN-ODE-ATT

This local directory is the source of truth for the project:

`C:\Users\kassa\Desktop\SNNODEATT`

GitHub is only a remote mirror for source code, configuration, lightweight experiment plans, and lightweight manifests. Do not use remote content to overwrite local work unless that action is explicitly reviewed and approved.

## Workflow Policy

- Local workspace is authoritative.
- GitHub mirrors local code and lightweight project metadata.
- Servers are used only for clone/pull, training, and temporary generated outputs.
- Servers should not manually edit code or push to GitHub.
- `Data/` and `origin_samples/` are local or server-side data locations and are not committed to GitHub.
- Large experiment outputs, checkpoints, logs, score arrays, and plots are not committed to GitHub.
- `Results/manifests/` is the allowed location for lightweight manifests, experiment plans, comparison plans, and short text summaries that should be mirrored to GitHub.
- Training results produced on the server should be copied back to the local workspace under `Results/<run_id>/`.

## Current Data Layout

The local Wafer preprocessed data currently lives at:

`Data/Wafer/preprocess_data`

The current training configuration still defaults to:

`./origin_samples/preprocess_data`

Until the training entrypoint is adjusted, pass the data directory explicitly when running training.

## Server Result Handling

Server training may write temporary outputs under `Results/<run_id>/` or another reviewed run directory. After training finishes, copy results back to:

`C:\Users\kassa\Desktop\SNNODEATT\Results\<run_id>\`

Those result folders stay local and are not pushed to GitHub. Only selected lightweight manifest files under `Results/manifests/` should be committed.
