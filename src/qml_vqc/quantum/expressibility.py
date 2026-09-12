"""Ansatz expressibility diagnostics based on the Haar fidelity distribution."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pennylane as qml


@dataclass(frozen=True)
class ExpressibilityScore:
    depth: int
    kl_to_haar: float
    mean_fidelity: float

    def to_dict(self) -> dict[str, int | float]:
        return {
            "depth": self.depth,
            "kl_to_haar": self.kl_to_haar,
            "mean_fidelity": self.mean_fidelity,
        }


def _sample_states(n_qubits: int, depth: int, n_samples: int, seed: int) -> np.ndarray:
    device = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(device)
    def ansatz(weights: np.ndarray):
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return qml.state()

    rng = np.random.default_rng(seed)
    shape = qml.StronglyEntanglingLayers.shape(n_layers=depth, n_wires=n_qubits)
    return np.stack(
        [np.asarray(ansatz(rng.uniform(0, 2 * np.pi, shape))) for _ in range(n_samples)]
    )


def expressibility_score(
    n_qubits: int,
    depth: int,
    n_samples: int,
    seed: int,
    bins: int = 20,
) -> ExpressibilityScore:
    """Estimate KL divergence from the exact Haar pairwise-fidelity law.

    Lower KL divergence indicates a sampled ansatz distribution closer to Haar
    random states. It is a distributional proxy, not a task-performance score.
    """

    states = _sample_states(n_qubits, depth, n_samples, seed)
    gram = np.abs(states @ states.conj().T) ** 2
    fidelities = gram[np.triu_indices(n_samples, k=1)]
    edges = np.linspace(0.0, 1.0, bins + 1)
    empirical, _ = np.histogram(fidelities, bins=edges)
    empirical = empirical.astype(float) + 1e-12
    empirical /= empirical.sum()

    dimension = 2**n_qubits
    # Haar CDF: 1 - (1 - F) ** (dimension - 1)
    cdf = 1.0 - (1.0 - edges) ** (dimension - 1)
    haar = np.diff(cdf) + 1e-12
    haar /= haar.sum()
    kl = float(np.sum(empirical * np.log(empirical / haar)))
    return ExpressibilityScore(depth=depth, kl_to_haar=kl, mean_fidelity=float(fidelities.mean()))
