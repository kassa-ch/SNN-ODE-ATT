"""Registry for anomaly score benchmark methods."""
from .functional_scores import (
    covariance_aware_trajectory_energy,
    euclidean_latent_score,
    negative_sobolev_hminus1_score,
    sobolev_h1_score,
)
from .kernel_scores import energy_distance_score, mmd_covariance_kernel_score
from .quantum_scores import (
    bures_distance_score,
    purity_score,
    quantum_relative_entropy_score,
    trace_distance_score,
    von_neumann_entropy_score,
)
from .trajectory_energy import hamiltonian_energy_score, ode_residual_energy_score

SCORE_REGISTRY = {
    "euclidean": euclidean_latent_score,
    "sobolev_h1": sobolev_h1_score,
    "sobolev_hminus1": negative_sobolev_hminus1_score,
    "trajectory_mahalanobis": covariance_aware_trajectory_energy,
    "mmd_cov_kernel": mmd_covariance_kernel_score,
    "energy_distance": energy_distance_score,
    "trace_distance": trace_distance_score,
    "qre": quantum_relative_entropy_score,
    "von_neumann_entropy": von_neumann_entropy_score,
    "purity": purity_score,
    "bures": bures_distance_score,
    "hamiltonian_energy": hamiltonian_energy_score,
    "ode_residual_energy": ode_residual_energy_score,
}
