"""Qiskit feature maps and exact or finite-shot quantum kernels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import SVC


@dataclass(frozen=True)
class KernelScore:
    model: str
    accuracy: float
    f1: float

    def to_dict(self) -> dict[str, str | float]:
        return {"model": self.model, "accuracy": self.accuracy, "f1": self.f1}


def feature_map_circuit(
    x: np.ndarray,
    n_qubits: int = 2,
    reps: int = 2,
    feature_scale: float = 0.5,
) -> QuantumCircuit:
    """Build a data-reuploading feature map with nearest-neighbor ZZ phases."""

    # The dataset lives in [-pi, pi]. Shift to [0, pi] before applying the
    # bandwidth so the ZZ phase follows the common Pauli-feature-map form.
    values = np.resize((np.asarray(x, dtype=float) + np.pi) / 2.0, n_qubits)
    values *= feature_scale
    circuit = QuantumCircuit(n_qubits)
    for _ in range(reps):
        for qubit, value in enumerate(values):
            circuit.h(qubit)
            circuit.p(2.0 * value, qubit)
        if n_qubits > 1:
            for qubit in range(n_qubits):
                target = (qubit + 1) % n_qubits
                if n_qubits == 2 and qubit == 1:
                    continue
                angle = 2.0 * (np.pi - values[qubit]) * (np.pi - values[target])
                circuit.cx(qubit, target)
                circuit.rz(angle, target)
                circuit.cx(qubit, target)
    return circuit


def statevectors(
    x: np.ndarray,
    n_qubits: int = 2,
    reps: int = 2,
    feature_scale: float = 0.5,
) -> np.ndarray:
    """Return one normalized feature-map statevector per sample."""

    return np.stack(
        [
            np.asarray(
                Statevector.from_instruction(
                    feature_map_circuit(row, n_qubits, reps, feature_scale)
                )
            )
            for row in x
        ]
    )


def fidelity_kernel(states_a: np.ndarray, states_b: np.ndarray | None = None) -> np.ndarray:
    """Compute the pure-state fidelity kernel ``|<a|b>|^2``."""

    states_b = states_a if states_b is None else states_b
    overlaps = states_a.conj() @ states_b.T
    return np.clip(np.abs(overlaps) ** 2, 0.0, 1.0).real


def _readout_transition(n_qubits: int, readout_error: float) -> np.ndarray:
    single = np.array(
        [[1.0 - readout_error, readout_error], [readout_error, 1.0 - readout_error]]
    )
    transition = np.array([[1.0]])
    for _ in range(n_qubits):
        transition = np.kron(transition, single)
    return transition


def sampled_probability_features(
    states: np.ndarray,
    shots: int,
    noise_strength: float,
    seed: int,
) -> np.ndarray:
    """Sample noisy computational-basis distributions.

    ``noise_strength`` combines a depolarizing mixture with symmetric readout
    flips. This is a transparent NISQ proxy, not a hardware calibration model.
    The square-root distributions form a measurement-derived feature embedding.
    """

    if shots <= 0:
        raise ValueError("shots must be positive")
    if not 0.0 <= noise_strength < 1.0:
        raise ValueError("noise_strength must lie in [0, 1)")

    probabilities = np.abs(states) ** 2
    dimension = probabilities.shape[1]
    n_qubits = int(np.log2(dimension))
    mixed = (1.0 - noise_strength) * probabilities + noise_strength / dimension
    readout_error = min(noise_strength / 2.0, 0.49)
    measured = mixed @ _readout_transition(n_qubits, readout_error)
    measured /= measured.sum(axis=1, keepdims=True)

    rng = np.random.default_rng(seed)
    frequencies = np.stack([rng.multinomial(shots, row) / shots for row in measured])
    return np.sqrt(frequencies)


def measurement_kernel(features_a: np.ndarray, features_b: np.ndarray | None = None) -> np.ndarray:
    """Return the squared Bhattacharyya measurement kernel."""

    features_b = features_a if features_b is None else features_b
    return np.clip(features_a @ features_b.T, 0.0, 1.0) ** 2


def stabilize_train_kernel(kernel: np.ndarray, epsilon: float = 1e-9) -> tuple[np.ndarray, float]:
    """Symmetrize a sampled Gram matrix and shift its spectrum when needed."""

    symmetric = (kernel + kernel.T) / 2.0
    minimum = float(np.linalg.eigvalsh(symmetric).min())
    shift = max(0.0, epsilon - minimum)
    if shift:
        symmetric = symmetric + shift * np.eye(len(symmetric))
    return symmetric, shift


def evaluate_kernel_svm(
    train_kernel: np.ndarray,
    test_kernel: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    name: str = "Qiskit fidelity kernel SVM",
) -> KernelScore:
    """Fit and score an SVM from precomputed train and test kernels."""

    model = SVC(kernel="precomputed", C=1.0)
    model.fit(train_kernel, y_train)
    predicted = model.predict(test_kernel)
    return KernelScore(
        model=name,
        accuracy=float(accuracy_score(y_test, predicted)),
        f1=float(f1_score(y_test, predicted)),
    )
