"""Consistent, headless plotting for experiment artifacts."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.switch_backend("Agg")


COLORS = {"classical": "#334155", "quantum": "#7c3aed", "noise": "#e11d48"}


def plot_model_comparison(rows: list[dict[str, object]], destination: Path) -> None:
    labels = [str(row["model"]) for row in rows]
    values = [float(row["accuracy"]) for row in rows]
    colors = [
        COLORS["quantum"]
        if "Qiskit" in label or "VQC" in label
        else COLORS["classical"]
        for label in labels
    ]
    figure, axis = plt.subplots(figsize=(9, 4.8))
    bars = axis.bar(labels, values, color=colors)
    axis.set_ylim(0.0, 1.05)
    axis.set_ylabel("Test accuracy")
    axis.set_title("Quantum models and classical baselines")
    axis.tick_params(axis="x", rotation=18)
    axis.bar_label(bars, fmt="%.3f", padding=3)
    figure.tight_layout()
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def plot_noise_sweep(rows: list[dict[str, object]], destination: Path) -> None:
    levels = np.array([float(row["noise_strength"]) for row in rows])
    means = np.array([float(row["accuracy_mean"]) for row in rows])
    stds = np.array([float(row["accuracy_std"]) for row in rows])
    figure, axis = plt.subplots(figsize=(7.5, 4.6))
    axis.errorbar(levels, means, yerr=stds, marker="o", capsize=4, color=COLORS["noise"])
    axis.set_ylim(0.0, 1.05)
    axis.set_xlabel("Simulated noise strength")
    axis.set_ylabel("Test accuracy")
    axis.set_title("Finite-shot measurement-kernel robustness")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def plot_training_curves(rows: list[dict[str, object]], destination: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for seed in sorted({int(row["seed"]) for row in rows}):
        selected = [row for row in rows if int(row["seed"]) == seed]
        epochs = [int(row["epoch"]) for row in selected]
        axes[0].plot(epochs, [float(row["loss"]) for row in selected], label=f"seed {seed}")
        axes[1].plot(
            epochs,
            [float(row["gradient_norm"]) for row in selected],
            label=f"seed {seed}",
        )
    axes[0].set(title="VQC optimization", xlabel="Epoch", ylabel="BCE loss")
    axes[1].set(title="Training stability", xlabel="Epoch", ylabel="Gradient norm")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def plot_expressibility(rows: list[dict[str, object]], destination: Path) -> None:
    depths = [int(row["depth"]) for row in rows]
    divergences = [float(row["kl_to_haar"]) for row in rows]
    figure, axis = plt.subplots(figsize=(7.5, 4.6))
    axis.plot(depths, divergences, marker="o", color=COLORS["quantum"])
    axis.set_xlabel("Ansatz depth")
    axis.set_ylabel("KL divergence to Haar (lower is closer)")
    axis.set_title("Strongly-entangling ansatz expressibility")
    axis.set_xticks(depths)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(destination, dpi=180)
    plt.close(figure)
