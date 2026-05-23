# Reproduction

1. Prepare raw data under `data/<DATASET>/raw`.
2. Run `scripts/prepare_<dataset>.py`.
3. Train wafer experiments via `scripts/train_exp1.py` ... `scripts/train_exp4.py` after setting config paths.
4. Evaluate with `scripts/eval_exp1_exp4.py`.
5. For Sinkhorn post-hoc or regularized variants, see `scripts/eval_sinkhorn_exp4.py` and `demos/demo4_sinkhorn_regularized_training.py`.

Additional manufacturing dataset entrypoints:

- ST-AWFD: use `configs/experiments/st_awfd.yaml` and `scripts/prepare_st_awfd.py`.
- Bosch Production Line Performance: use `configs/experiments/bosch_production_line.yaml` and `scripts/prepare_bosch.py`.

Bosch should be reported as staged BTD, not as a native continuous-time sensor
curve. Its station/stage axis can still support SNN-ODE-ATT as a discrete
approximation to a functional production profile.
