# Reproduction

1. Prepare raw data under `Data/<DATASET>/raw`.
2. Run `Scrips/data_loader/prepare_<dataset>.py`.
3. Train wafer experiments via `Scrips/train/train_exp1.py` ... `Scrips/train/train_exp4.py` after setting config paths.
4. Evaluate with `Scrips/main/eval_exp1_exp4.py`.
5. For Sinkhorn post-hoc or regularized variants, see `Scrips/main/eval_sinkhorn_exp4.py` and `Detect/demo4/demo4_sinkhorn_regularized_training.py`.

Additional manufacturing dataset entrypoints:

- ST-AWFD: use `Scrips/config/experiments/st_awfd.yaml` and `Scrips/data_loader/prepare_st_awfd.py`.
- Bosch Production Line Performance: use `Scrips/config/experiments/bosch_production_line.yaml` and `Scrips/data_loader/prepare_bosch.py`.

Bosch should be reported as staged BTD, not as a native continuous-time sensor
curve. Its station/stage axis can still support SNN-ODE-ATT as a discrete
approximation to a functional production profile.
