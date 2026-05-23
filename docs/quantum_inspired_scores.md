# Quantum-Inspired Scores

The benchmark constructs a density matrix from hidden trajectory covariance:

rho = M^T W M + eps I, normalized by trace.

Implemented scores:

- Trace distance: 0.5 ||rho - sigma||_*
- Quantum relative entropy: Tr rho (log rho - log sigma)
- von Neumann entropy: -Tr rho log rho
- Purity: Tr rho^2
- Bures distance from fidelity
- Hamiltonian energy using inverse normal covariance

These are quantum-inspired geometric summaries of hidden states. They are not claims that the industrial system is a physical quantum system.
