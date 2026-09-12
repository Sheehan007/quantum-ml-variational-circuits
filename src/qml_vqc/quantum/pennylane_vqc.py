"""Differentiable PennyLane circuit wrapped in a PyTorch model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pennylane as qml
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn


@dataclass(frozen=True)
class VQCResult:
    seed: int
    accuracy: float
    f1: float
    history: list[dict[str, float | int]]


class QuantumLayer(nn.Module):
    """Angle encoding followed by a trainable, entangling ansatz."""

    def __init__(self, n_qubits: int, n_layers: int) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.weights = nn.Parameter(0.05 * torch.randn(n_layers, n_qubits, 3))
        device = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(device, interface="torch", diff_method="backprop")
        def circuit(features: torch.Tensor, weights: torch.Tensor):
            qml.AngleEmbedding(features, wires=range(n_qubits), rotation="Y")
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return tuple(qml.expval(qml.PauliZ(wire)) for wire in range(n_qubits))

        self.circuit = circuit

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = [torch.stack(self.circuit(row, self.weights)) for row in inputs]
        return torch.stack(outputs).to(dtype=inputs.dtype)


class HybridVQC(nn.Module):
    """Small classical projection, quantum layer, and classical readout."""

    def __init__(self, n_features: int, n_qubits: int, n_layers: int) -> None:
        super().__init__()
        self.requires_projection = n_features != n_qubits
        self.project = (
            nn.Linear(n_features, n_qubits) if self.requires_projection else nn.Identity()
        )
        self.quantum = QuantumLayer(n_qubits=n_qubits, n_layers=n_layers)
        self.readout = nn.Linear(n_qubits, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        projected = self.project(inputs)
        angles = torch.tanh(projected) * torch.pi if self.requires_projection else projected
        return self.readout(self.quantum(angles)).squeeze(-1)


def train_vqc(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    n_qubits: int,
    n_layers: int,
    epochs: int,
    learning_rate: float,
    seed: int,
) -> VQCResult:
    """Train one deterministic VQC run and retain stability diagnostics."""

    np.random.seed(seed)
    torch.manual_seed(seed)
    train_x = torch.as_tensor(x_train, dtype=torch.float32)
    train_y = torch.as_tensor(y_train, dtype=torch.float32)
    test_x = torch.as_tensor(x_test, dtype=torch.float32)

    model = HybridVQC(x_train.shape[1], n_qubits, n_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    history: list[dict[str, float | int]] = []

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        logits = model(train_x)
        loss = criterion(logits, train_y)
        loss.backward()
        grad_norm = float(
            torch.sqrt(
                sum(
                    torch.sum(parameter.grad.detach() ** 2)
                    for parameter in model.parameters()
                    if parameter.grad is not None
                )
            )
        )
        optimizer.step()
        with torch.no_grad():
            train_accuracy = float(((torch.sigmoid(logits) >= 0.5) == train_y).float().mean())
            test_prediction = (torch.sigmoid(model(test_x)) >= 0.5).cpu().numpy().astype(int)
            test_accuracy = float(accuracy_score(y_test, test_prediction))
        history.append(
            {
                "seed": seed,
                "epoch": epoch,
                "loss": float(loss.detach()),
                "gradient_norm": grad_norm,
                "train_accuracy": train_accuracy,
                "test_accuracy": test_accuracy,
            }
        )

    with torch.no_grad():
        predicted = (torch.sigmoid(model(test_x)) >= 0.5).cpu().numpy().astype(int)
    return VQCResult(
        seed=seed,
        accuracy=float(accuracy_score(y_test, predicted)),
        f1=float(f1_score(y_test, predicted)),
        history=history,
    )
