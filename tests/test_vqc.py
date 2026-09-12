import numpy as np
import torch

from qml_vqc.quantum.pennylane_vqc import HybridVQC, train_vqc


def test_hybrid_vqc_forward_shape_and_gradients() -> None:
    torch.manual_seed(0)
    model = HybridVQC(n_features=2, n_qubits=2, n_layers=1)
    inputs = torch.tensor([[0.1, -0.2], [0.3, 0.4]], dtype=torch.float32)
    output = model(inputs)
    assert output.shape == (2,)
    output.sum().backward()
    assert model.quantum.weights.grad is not None
    assert torch.isfinite(model.quantum.weights.grad).all()


def test_short_training_run_records_diagnostics() -> None:
    x_train = np.array([[-1.0, -0.8], [-0.8, -1.0], [0.8, 1.0], [1.0, 0.8]])
    y_train = np.array([0, 0, 1, 1])
    result = train_vqc(
        x_train,
        y_train,
        x_train,
        y_train,
        n_qubits=2,
        n_layers=1,
        epochs=2,
        learning_rate=0.05,
        seed=4,
    )
    assert len(result.history) == 2
    assert 0.0 <= result.accuracy <= 1.0
    assert all(float(row["gradient_norm"]) >= 0.0 for row in result.history)
