"""Classical reference models for the same train/test split."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


@dataclass(frozen=True)
class ModelScore:
    model: str
    accuracy: float
    f1: float

    def to_dict(self) -> dict[str, str | float]:
        return {"model": self.model, "accuracy": self.accuracy, "f1": self.f1}


def _score(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> ModelScore:
    return ModelScore(
        model=name,
        accuracy=float(accuracy_score(y_true, y_pred)),
        f1=float(f1_score(y_true, y_pred)),
    )


def evaluate_classical_baselines(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
) -> list[ModelScore]:
    """Fit RBF-SVM and compact MLP reference models."""

    models = {
        "RBF SVM": SVC(kernel="rbf", C=1.0, gamma="scale"),
        "Classical MLP": MLPClassifier(
            hidden_layer_sizes=(16, 8),
            activation="tanh",
            solver="adam",
            learning_rate_init=0.01,
            max_iter=1000,
            early_stopping=False,
            random_state=seed,
        ),
    }
    scores: list[ModelScore] = []
    for name, model in models.items():
        model.fit(x_train, y_train)
        scores.append(_score(name, y_test, model.predict(x_test)))
    return scores
