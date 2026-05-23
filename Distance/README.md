# Distance

Paper-facing anomaly score and distance method organization layer.

Active scoring implementations remain under `src/snnodeatt/scoring/`. Files in this directory are wrappers or status markers that map canonical method names to implementation paths.

Status legend:

- ACTIVE: implemented and available through the scoring registry or direct scorer.
- WRAPPER: this directory forwards to an active implementation.
- NOT_IMPLEMENTED_OR_NOT_FOUND: no implementation was found during Phase R0.
