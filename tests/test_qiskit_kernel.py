import numpy as np

from qml_vqc.quantum.qiskit_kernel import (
    fidelity_kernel,
    measurement_kernel,
    sampled_probability_features,
    stabilize_train_kernel,
    statevectors,
)


def test_fidelity_kernel_is_symmetric_psd_with_unit_diagonal() -> None:
    x = np.array([[-1.0, 0.2], [0.4, 1.1], [1.2, -0.7]])
    states = statevectors(x, n_qubits=2, reps=1)
    kernel = fidelity_kernel(states)
    np.testing.assert_allclose(kernel, kernel.T, atol=1e-12)
    np.testing.assert_allclose(np.diag(kernel), 1.0, atol=1e-12)
    assert np.linalg.eigvalsh(kernel).min() >= -1e-10


def test_sampled_kernel_is_reproducible_and_stabilized() -> None:
    states = statevectors(np.array([[0.1, 0.2], [0.3, -0.4], [1.0, 0.5]]), reps=1)
    first = sampled_probability_features(states, shots=100, noise_strength=0.05, seed=9)
    second = sampled_probability_features(states, shots=100, noise_strength=0.05, seed=9)
    np.testing.assert_allclose(first, second)
    kernel = measurement_kernel(first)
    stable, shift = stabilize_train_kernel(kernel)
    assert shift >= 0.0
    assert np.linalg.eigvalsh(stable).min() >= -1e-8
