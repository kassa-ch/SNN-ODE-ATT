"""Hamiltonian-inspired energy score wrapper."""

from snnodeatt.scoring.trajectory_energy import hamiltonian_energy_score

SCORE_NAME = "hamiltonian_energy"
score = hamiltonian_energy_score
