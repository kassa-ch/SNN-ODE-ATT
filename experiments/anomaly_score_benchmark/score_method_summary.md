| method | input shape | main statistic | detects | pros | cons | priority |
|---|---|---|---|---|---|---|
| euclidean | [N,T,D] | weighted latent mean | global shifts | simple | ignores dynamics | medium |
| sobolev_h1 | [N,T,D] | value + derivative | shape and slope drift | functional | needs aligned time grid | high |
| sobolev_hminus1 | [N,T,D] | spectral low-frequency residual | smooth drifts | noise-robust | approximate FFT | medium |
| trajectory_mahalanobis | [N,T,D] | covariance energy | hidden trajectory deviations | strong baseline | covariance assumptions | high |
| mmd_cov_kernel | empirical points | RBF kernel MMD | distribution shift | flexible | bandwidth sensitive | high |
| energy_distance | empirical points | pairwise distance energy | broad distribution shift | simple | prototype choice | high |
| trace_distance | density matrix | nuclear norm | covariance geometry | stable | quantum-inspired only | medium |
| qre | density matrix | relative entropy | asymmetric covariance shift | expressive | eigenvalue sensitive | medium |
| entropy/purity | density matrix | spectral complexity | variance concentration | interpretable | not directional | medium |
| bures | density matrix | fidelity distance | covariance manifold shift | geometric | expensive | medium |
| hamiltonian_energy | [N,T,D] | inverse covariance energy | high-energy dynamics | simple | diagonal default | high |
| ode_residual_energy | [N,T,D] | finite difference residual | dynamic inconsistency | change-point friendly | approximate without f_ODE | high |
