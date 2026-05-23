# Server Baseline Model File Inventory

This report was generated on the server by `scripts/ops/collect_server_baseline_model_files.sh`.

The script searches the original project read-only and copies small `.py` model implementations into `src/snnodeatt/models/baselines/`.

## Summary

| Category | Method | Status | Target | Source | Lines | Size bytes |
|---|---|---|---|---|---:|---:|
| linear_prediction | LinearAR | copied | `src/snnodeatt/models/baselines/linear_ar.py` | `/root/autodl-tmp/wafer_att_ctsr_vca/models/STAT_MODELS.py` | 131 | 5129 |
| sparse_linear | LassoAR | copied | `src/snnodeatt/models/baselines/lasso_ar.py` | `/root/autodl-tmp/wafer_att_ctsr_vca/models/baseline.py` | 92 | 2496 |
| kernel_method | KRR | copied | `src/snnodeatt/models/baselines/krr.py` | `/root/autodl-tmp/wafer_att_ctsr_vca/models/baseline.py` | 92 | 2496 |
| functional_statistics | FPCA | copied | `src/snnodeatt/models/baselines/fpca.py` | `/root/autodl-tmp/wafer_att_ctsr_vca/models/STAT_MODELS.py` | 131 | 5129 |
| integral_trend | CumInt | copied | `src/snnodeatt/models/baselines/cumulative_integral.py` | `/root/autodl-tmp/wafer_att_ctsr_vca/models/STAT_MODELS.py` | 131 | 5129 |
| spiking_neural_network | SNN | copied | `src/snnodeatt/models/baselines/snn.py` | `/root/autodl-tmp/wafer_att_ctsr_vca/models/SNN.py` | 223 | 6683 |
| gated_recurrent_network | GRU | copied | `src/snnodeatt/models/baselines/gru.py` | `/root/autodl-tmp/wafer_att_ctsr_vca/models/RNN_GRU_with_delta_t.py` | 134 | 4713 |
| continuous_time_latent_model | Latent ODE | copied | `src/snnodeatt/models/baselines/latent_ode.py` | `/root/autodl-tmp/wafer_att_ctsr_vca/models/SNN_LatentODE.py` | 188 | 5783 |
| continuous_time_recurrent_model | ODERNN | copied | `src/snnodeatt/models/baselines/ode_rnn.py` | `/root/autodl-tmp/wafer_att_ctsr_vca/models/ODE_RNN.py` | 261 | 8653 |
| probabilistic_latent_factor_model | DF2M | copied | `src/snnodeatt/models/baselines/df2m.py` | `/root/autodl-tmp/wafer_att_ctsr_vca/models/DF2M.py` | 243 | 7200 |
| paper_method | SNN-ODE | copied | `src/snnodeatt/models/baselines/snn_ode_baseline.py` | `/root/autodl-tmp/wafer_att_ctsr_vca/models/SNN_ODEATT.py` | 304 | 10588 |

## Candidate Details

### LinearAR

- Status: `copied`
- Target: `src/snnodeatt/models/baselines/linear_ar.py`
- Source: `/root/autodl-tmp/wafer_att_ctsr_vca/models/STAT_MODELS.py`
- SHA256: `6a189a9e521811aabd0dc16541b7cbc239d3a843e09a3615c51867b3e1a6d0c5`

Top candidates:
- score=10 size=5129 path=`/root/autodl-tmp/wafer_att_ctsr_vca/models/STAT_MODELS.py`
- score=7 size=76376 path=`/root/autodl-tmp/wafer_att_ctsr_vca/demo1.py`
- score=7 size=2496 path=`/root/autodl-tmp/wafer_att_ctsr_vca/models/baseline.py`
- score=5 size=34295 path=`/root/autodl-tmp/wafer_att_ctsr_vca/train.py`
- score=5 size=1106 path=`/root/autodl-tmp/wafer_att_ctsr_vca/common.py`

### LassoAR

- Status: `copied`
- Target: `src/snnodeatt/models/baselines/lasso_ar.py`
- Source: `/root/autodl-tmp/wafer_att_ctsr_vca/models/baseline.py`
- SHA256: `ea75e39cf45e432d87b74c4a1b6a59d3f7100d99ff5d45f2ceeb884784fa2e78`

Top candidates:
- score=10 size=2496 path=`/root/autodl-tmp/wafer_att_ctsr_vca/models/baseline.py`
- score=8 size=5129 path=`/root/autodl-tmp/wafer_att_ctsr_vca/models/STAT_MODELS.py`
- score=7 size=76376 path=`/root/autodl-tmp/wafer_att_ctsr_vca/demo1.py`
- score=5 size=34295 path=`/root/autodl-tmp/wafer_att_ctsr_vca/train.py`
- score=5 size=1106 path=`/root/autodl-tmp/wafer_att_ctsr_vca/common.py`

### KRR

- Status: `copied`
- Target: `src/snnodeatt/models/baselines/krr.py`
- Source: `/root/autodl-tmp/wafer_att_ctsr_vca/models/baseline.py`
- SHA256: `ea75e39cf45e432d87b74c4a1b6a59d3f7100d99ff5d45f2ceeb884784fa2e78`

Top candidates:
- score=11 size=2496 path=`/root/autodl-tmp/wafer_att_ctsr_vca/models/baseline.py`
- score=7 size=5129 path=`/root/autodl-tmp/wafer_att_ctsr_vca/models/STAT_MODELS.py`
- score=5 size=76376 path=`/root/autodl-tmp/wafer_att_ctsr_vca/demo1.py`
- score=5 size=34295 path=`/root/autodl-tmp/wafer_att_ctsr_vca/train.py`
- score=5 size=1106 path=`/root/autodl-tmp/wafer_att_ctsr_vca/common.py`

### FPCA

- Status: `copied`
- Target: `src/snnodeatt/models/baselines/fpca.py`
- Source: `/root/autodl-tmp/wafer_att_ctsr_vca/models/STAT_MODELS.py`
- SHA256: `6a189a9e521811aabd0dc16541b7cbc239d3a843e09a3615c51867b3e1a6d0c5`

Top candidates:
- score=10 size=5129 path=`/root/autodl-tmp/wafer_att_ctsr_vca/models/STAT_MODELS.py`
- score=8 size=2496 path=`/root/autodl-tmp/wafer_att_ctsr_vca/models/baseline.py`
- score=7 size=76376 path=`/root/autodl-tmp/wafer_att_ctsr_vca/demo1.py`
- score=5 size=34295 path=`/root/autodl-tmp/wafer_att_ctsr_vca/train.py`
- score=5 size=1106 path=`/root/autodl-tmp/wafer_att_ctsr_vca/common.py`

### CumInt

- Status: `copied`
- Target: `src/snnodeatt/models/baselines/cumulative_integral.py`
- Source: `/root/autodl-tmp/wafer_att_ctsr_vca/models/STAT_MODELS.py`
- SHA256: `6a189a9e521811aabd0dc16541b7cbc239d3a843e09a3615c51867b3e1a6d0c5`

Top candidates:
- score=7 size=5129 path=`/root/autodl-tmp/wafer_att_ctsr_vca/models/STAT_MODELS.py`
- score=5 size=76376 path=`/root/autodl-tmp/wafer_att_ctsr_vca/demo1.py`
- score=5 size=34295 path=`/root/autodl-tmp/wafer_att_ctsr_vca/train.py`
- score=5 size=1106 path=`/root/autodl-tmp/wafer_att_ctsr_vca/common.py`
- score=5 size=6683 path=`/root/autodl-tmp/wafer_att_ctsr_vca/models/SNN.py`

### SNN

- Status: `copied`
- Target: `src/snnodeatt/models/baselines/snn.py`
- Source: `/root/autodl-tmp/wafer_att_ctsr_vca/models/SNN.py`
- SHA256: `9ade6dabc671f8762d1cd4d597b7103a98f32e99284f9f8141899b691c2a196f`

Top candidates:
- score=79 size=6683 path=`/root/autodl-tmp/wafer_att_ctsr_vca/models/SNN.py`
- score=13 size=76376 path=`/root/autodl-tmp/wafer_att_ctsr_vca/demo1.py`
- score=12 size=9374 path=`/root/autodl-tmp/wafer_att_ctsr_vca/models/SNN_ODE.py`
- score=10 size=17020 path=`/root/autodl-tmp/wafer_att_ctsr_vca/ctsr_vca/scripts/bounded_residual_generator_quick_20260423.py`
- score=10 size=19215 path=`/root/autodl-tmp/wafer_att_ctsr_vca/ctsr_vca/scripts/exp3_pchip_viewaware_reference_calibration_20260423.py`

### GRU

- Status: `copied`
- Target: `src/snnodeatt/models/baselines/gru.py`
- Source: `/root/autodl-tmp/wafer_att_ctsr_vca/models/RNN_GRU_with_delta_t.py`
- SHA256: `d86efb20ac068f9befd0669ae30e1a18f34cdc4d6043658263f96a5e7a1267eb`

Top candidates:
- score=12 size=4713 path=`/root/autodl-tmp/wafer_att_ctsr_vca/models/RNN_GRU_with_delta_t.py`
- score=10 size=3439 path=`/root/autodl-tmp/wafer_att_ctsr_vca/models/RNN_GRU.py`
- score=9 size=8653 path=`/root/autodl-tmp/wafer_att_ctsr_vca/models/ODE_RNN.py`
- score=9 size=7492 path=`/root/autodl-tmp/wafer_att_ctsr_vca/ctsr_vca/scripts/train_bounded_residual_model_20260423.py`
- score=7 size=7200 path=`/root/autodl-tmp/wafer_att_ctsr_vca/models/DF2M.py`

### Latent ODE

- Status: `copied`
- Target: `src/snnodeatt/models/baselines/latent_ode.py`
- Source: `/root/autodl-tmp/wafer_att_ctsr_vca/models/SNN_LatentODE.py`
- SHA256: `59907b1d2711411cd93e7aad3b540b60f395aa66de1ceac5fa6e17f183ee7757`

Top candidates:
- score=144 size=5783 path=`/root/autodl-tmp/wafer_att_ctsr_vca/models/SNN_LatentODE.py`
- score=16 size=14251 path=`/root/autodl-tmp/wafer_att_ctsr_vca/model_factory.py`
- score=15 size=20096 path=`/root/autodl-tmp/wafer_att_ctsr_vca/config.py`
- score=15 size=20096 path=`/root/autodl-tmp/SNNODEATT/src/snnodeatt/utils/legacy_config.py`
- score=12 size=76376 path=`/root/autodl-tmp/wafer_att_ctsr_vca/demo1.py`

### ODERNN

- Status: `copied`
- Target: `src/snnodeatt/models/baselines/ode_rnn.py`
- Source: `/root/autodl-tmp/wafer_att_ctsr_vca/models/ODE_RNN.py`
- SHA256: `a021159f2c52b0a66aab5167fbca474e96fd0424a755bc8d467f1b7c7873ff86`

Top candidates:
- score=133 size=8653 path=`/root/autodl-tmp/wafer_att_ctsr_vca/models/ODE_RNN.py`
- score=18 size=20096 path=`/root/autodl-tmp/wafer_att_ctsr_vca/config.py`
- score=18 size=14251 path=`/root/autodl-tmp/wafer_att_ctsr_vca/model_factory.py`
- score=18 size=20096 path=`/root/autodl-tmp/SNNODEATT/src/snnodeatt/utils/legacy_config.py`
- score=11 size=4713 path=`/root/autodl-tmp/wafer_att_ctsr_vca/models/RNN_GRU_with_delta_t.py`

### DF2M

- Status: `copied`
- Target: `src/snnodeatt/models/baselines/df2m.py`
- Source: `/root/autodl-tmp/wafer_att_ctsr_vca/models/DF2M.py`
- SHA256: `5f4c6ad52379cda49acfa19e9f0151b3404882278b01d4eddfa21c7a4ca388bf`

Top candidates:
- score=65 size=7200 path=`/root/autodl-tmp/wafer_att_ctsr_vca/models/DF2M.py`
- score=6 size=76376 path=`/root/autodl-tmp/wafer_att_ctsr_vca/demo1.py`
- score=5 size=34295 path=`/root/autodl-tmp/wafer_att_ctsr_vca/train.py`
- score=5 size=1106 path=`/root/autodl-tmp/wafer_att_ctsr_vca/common.py`
- score=5 size=6683 path=`/root/autodl-tmp/wafer_att_ctsr_vca/models/SNN.py`

### SNN-ODE

- Status: `copied`
- Target: `src/snnodeatt/models/baselines/snn_ode_baseline.py`
- Source: `/root/autodl-tmp/wafer_att_ctsr_vca/models/SNN_ODEATT.py`
- SHA256: `febddf1d868cbd781d4723e1da03b028e0ededdc6491132e458abcc7155ae476`

Top candidates:
- score=142 size=10588 path=`/root/autodl-tmp/wafer_att_ctsr_vca/models/SNN_ODEATT.py`
- score=142 size=10588 path=`/root/autodl-tmp/SNNODEATT/src/snnodeatt/models/snn_odeatt.py`
- score=135 size=9374 path=`/root/autodl-tmp/wafer_att_ctsr_vca/models/SNN_ODE.py`
- score=126 size=4856 path=`/root/autodl-tmp/wafer_att_ctsr_vca/scripts/train_snn_odeatt_smoke.py`
- score=125 size=12646 path=`/root/autodl-tmp/wafer_att_ctsr_vca/scripts/evaluate_snn_odeatt_layered.py`

