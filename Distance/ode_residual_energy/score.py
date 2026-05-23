"""ODE residual energy score wrapper."""

from snnodeatt.scoring.trajectory_energy import ode_residual_energy_score

SCORE_NAME = "ode_residual_energy"
score = ode_residual_energy_score
