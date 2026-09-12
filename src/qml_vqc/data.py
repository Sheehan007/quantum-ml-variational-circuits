"""Dataset creation and preprocessing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler


@dataclass(frozen=True)
class DatasetSplit:
    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray


def make_binary_dataset(
    n_samples: int = 120,
    noise: float = 0.16,
    test_size: float = 0.30,
    seed: int = 42,
) -> DatasetSplit:
    """Create a stratified two-moons split scaled to quantum rotation angles.

    Standardization is fit only on the training partition. A second training-only
    transform maps features to ``[-pi, pi]`` for angle encoding.
    """

    x, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )
    standard = StandardScaler().fit(x_train)
    x_train_std = standard.transform(x_train)
    x_test_std = standard.transform(x_test)
    angle_scaler = MinMaxScaler(feature_range=(-np.pi, np.pi), clip=True).fit(x_train_std)
    return DatasetSplit(
        x_train=angle_scaler.transform(x_train_std).astype(np.float64),
        x_test=angle_scaler.transform(x_test_std).astype(np.float64),
        y_train=y_train.astype(np.int64),
        y_test=y_test.astype(np.int64),
    )
