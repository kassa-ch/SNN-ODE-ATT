# Bosch Production Line Performance

Bosch Production Line Performance is a production-line quality/failure dataset
originally released for a Kaggle competition. It is useful as a supplementary
high-end manufacturing quality anomaly dataset, but its temporal meaning should
be stated carefully.

Important modeling note:

- This is a staged production-line dataset, not a native continuous-time sensor
  curve.
- It can be converted to a **staged BTD** representation where `T` is the
  ordered station/process-stage axis and `D` contains measurements observed at
  that stage.
- In SNN-ODE-ATT papers, describe it as a discrete stage sequence or a
  discretized approximation to a functional process profile, not as a naturally
  continuous trajectory.

Recommended interpretation:

- `B`: manufactured parts.
- `T`: ordered production stations, stages, or grouped measurement blocks.
- `D`: multivariate measurements within each station/stage.
- `label`: `Response`, indicating production failure/quality anomaly.

Raw files are not committed to this repository. Place downloaded files under
`data/BoschProductionLine/raw/`, then adapt and run:

```bash
python scripts/prepare_bosch.py --raw-dir data/BoschProductionLine/raw --out-dir data/BoschProductionLine/processed
```

Provider page:

- https://www.kaggle.com/c/bosch-production-line-performance
