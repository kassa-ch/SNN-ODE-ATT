# Reproduction

1. Prepare raw data under `data/<DATASET>/raw`.
2. Run `scripts/prepare_<dataset>.py`.
3. Train wafer experiments via `scripts/train_exp1.py` ... `scripts/train_exp4.py` after setting config paths.
4. Evaluate with `scripts/eval_exp1_exp4.py`.
5. For Sinkhorn post-hoc or regularized variants, see `scripts/eval_sinkhorn_exp4.py` and `demos/demo4_sinkhorn_regularized_training.py`.
