# SNN-ODE-ATT

SNN-ODE-ATT is a reproducible research framework for multivariate functional-profile and industrial long-sequence anomaly detection. It combines a continuous-time spiking neural representation, ODE-style state evolution, causal time-weighted attention, and multiple anomaly scoring backends.

当前仓库整理自 wafer SNN-ODEATT 实验项目，目标是提供一个标准 GitHub 框架，便于复现实验和迁移到 HAI / SWaT / WADI / TEP 等工业时序数据集。原始大型数据、私有数据、训练 checkpoint 和服务器实验结果默认不提交。

## Method Overview

```text
X(t) -> preprocessing / PCHIP dense or irregular view
     -> SNN-ODEATT hidden trajectory H(t)
     -> latent/readout summary z
     -> anomaly score: Mahalanobis / rich global-local / Sinkhorn / W2 / sliced Wasserstein
```

## Core Equations

For an irregular multivariate input \(x(t_k)\), the SNN-ODE state evolves as a continuous-time hidden state with membrane and synaptic variables:

\[
\dot h(t) = f_\theta(h(t), x(t), \Delta t), \quad
m_t^* = \operatorname{soft\_reset}(m_t, s_t), \quad
s_t = \sigma_\beta(m_t - \vartheta).
\]

Causal time-weighted attention summarizes hidden trajectories:

\[
a_{ij} = \operatorname{softmax}_{j\le i}\left(\frac{q_i^\top k_j}{\sqrt d} + b(\Delta t_{ij})\right),\quad
\tilde h_i = \sum_{j\le i} a_{ij}v_j.
\]

The unsupervised objective follows the original project:

\[
\mathcal L_{\text{base}} =
\omega_{\text{rec}}\mathcal L_{\text{rec}}+
\omega_{\text{pred}}\mathcal L_{\text{pred}}+
\omega_{\text{stab}}\mathcal L_{\text{stability}}.
\]

Mahalanobis scoring uses train/validation normal latent statistics:

\[
s_M(z) = \sqrt{(z-\mu_N)^\top(\Sigma_N+\lambda I)^{-1}(z-\mu_N)}.
\]

Post-hoc Sinkhorn scoring represents each hidden trajectory as an empirical measure:

\[
Q_i=\sum_t \pi_{it}\delta_{u_{it}},\quad
u_{it}=[\sqrt{\lambda_t}\tau_{it},\, \bar m^*_{it}],
\]

and computes:

\[
S_\varepsilon(Q_i,P_N)=OT_\varepsilon(Q_i,P_N)
-\frac12OT_\varepsilon(Q_i,Q_i)-\frac12OT_\varepsilon(P_N,P_N).
\]

## Quick Start

```bash
pip install -r requirements.txt
python demos/demo1_wafer_mahalanobis.py --help
python demos/demo2_wafer_sinkhorn_posthoc.py --help
python tests/test_sinkhorn.py
python -m compileall src scripts demos tests
```

## Data

Raw HAI, SWaT, WADI, TEP, ST-AWFD, and Bosch Production Line files are not bundled. SWaT/WADI require access through the original providers; Bosch follows Kaggle terms; ST-AWFD follows the provider repository terms. Place raw files under `data/<DATASET>/raw/`, then run the corresponding `scripts/prepare_*.py` script after adapting the field names.

Additional manufacturing-quality datasets:

- `ST_AWFD`: public semiconductor wafer fault detection data. This is a natural supplement for wafer process monitoring and multivariate coupling anomaly detection.
- `BoschProductionLine`: production-line quality/failure data. In this repo it is explicitly treated as **staged BTD**, where `T` is the ordered station/stage axis. It is not a native continuous-time curve, but it can be used as a discrete approximation to a functional production profile.

## Reproduction

Wafer experiment templates are under `configs/experiments/wafer_exp1.yaml` to `wafer_exp4.yaml`:

- exp1: augmented anomalies, aligned/non-Poisson view.
- exp2: original anomalies, aligned/non-Poisson view.
- exp3: augmented anomalies with Poisson/nonuniform view.
- exp4: original anomalies with Poisson/nonuniform view; current best in the original project is M1+M2 exact replay.

The current wafer experiments found that Mahalanobis/rich global-local scoring is stronger than post-hoc Sinkhorn on exp4. Sinkhorn remains a research extension rather than the default scoring branch.

## Important Notes

- Do not commit private raw datasets or licensed data.
- Use Git LFS for binary checkpoints, `.pt`, `.pth`, `.pkl`, `.npy`, `.npz`, and large `.csv` files.
- This repo contains framework smoke tests and reusable code. Some end-to-end dataset-specific train/eval entrypoints are wrappers and require local data paths.

## Citation

Please cite the associated paper once available. Acknowledgements should include the original dataset providers and any required SWaT/WADI access terms.

## Anomaly score benchmark library

This repo includes a CPU-friendly benchmark for comparing hidden-trajectory anomaly scores:

```bash
python experiments/anomaly_score_benchmark/run_benchmark.py --toy
```

It currently supports Euclidean latent distance, Sobolev H1, negative Sobolev H-1, trajectory Mahalanobis energy, MMD, Energy distance, quantum-inspired density matrix distances, entropy/purity, Bures distance, Hamiltonian energy, and ODE residual energy. See `docs/anomaly_score_benchmark.md`.
