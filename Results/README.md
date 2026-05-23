# Results

Central result-management directory for future experiments.

Commit allowed:

- README files
- `.gitkeep`
- small reports and small manifest examples

Do not commit:

- checkpoints
- best model weights
- hidden caches
- large benchmark outputs
- raw data
- logs containing secrets

Suggested benchmark naming:

```text
Results/benchmarks/offline/{dataset}_{model}_{distance}_offline.csv
Results/benchmarks/prefix/{dataset}_{model}_{distance}_prefix.csv
Results/reports/{phase_or_dataset}_summary.md
```
