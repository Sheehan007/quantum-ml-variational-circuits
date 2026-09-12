"""End-to-end experiment runner and command-line interface."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pennylane as qml
import qiskit
import sklearn
import torch

from qml_vqc.config import ExperimentConfig, get_config
from qml_vqc.data import make_binary_dataset
from qml_vqc.models.classical import evaluate_classical_baselines
from qml_vqc.plotting import (
    plot_expressibility,
    plot_model_comparison,
    plot_noise_sweep,
    plot_training_curves,
)
from qml_vqc.quantum.expressibility import expressibility_score
from qml_vqc.quantum.pennylane_vqc import train_vqc
from qml_vqc.quantum.qiskit_kernel import (
    evaluate_kernel_svm,
    fidelity_kernel,
    measurement_kernel,
    sampled_probability_features,
    stabilize_train_kernel,
    statevectors,
)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "qiskit": qiskit.__version__,
        "pennylane": qml.__version__,
        "torch": torch.__version__,
    }


def _noise_experiment(
    config: ExperimentConfig,
    train_states: np.ndarray,
    test_states: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for index, level in enumerate(config.noise_levels):
        accuracies: list[float] = []
        shifts: list[float] = []
        raw_min_eigenvalues: list[float] = []
        for repeat in range(config.noise_repeats):
            run_seed = config.seed + 1000 * index + repeat
            train_features = sampled_probability_features(
                train_states, config.shots, level, run_seed
            )
            test_features = sampled_probability_features(
                test_states, config.shots, level, run_seed + 50_000
            )
            raw_train_kernel = measurement_kernel(train_features)
            raw_min_eigenvalues.append(float(np.linalg.eigvalsh(raw_train_kernel).min()))
            train_kernel, shift = stabilize_train_kernel(raw_train_kernel)
            score = evaluate_kernel_svm(
                train_kernel,
                measurement_kernel(test_features, train_features),
                y_train,
                y_test,
                name="Finite-shot measurement kernel SVM",
            )
            accuracies.append(score.accuracy)
            shifts.append(shift)
        rows.append(
            {
                "noise_strength": level,
                "shots": config.shots,
                "repeats": config.noise_repeats,
                "accuracy_mean": float(np.mean(accuracies)),
                "accuracy_std": float(np.std(accuracies)),
                "raw_min_eigenvalue_mean": float(np.mean(raw_min_eigenvalues)),
                "psd_shift_mean": float(np.mean(shifts)),
            }
        )
    return rows


def run_experiments(config: ExperimentConfig, output_dir: Path) -> dict[str, Any]:
    """Execute all comparisons and persist tables, figures, and metadata."""

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = make_binary_dataset(
        n_samples=config.n_samples,
        noise=config.noise,
        test_size=config.test_size,
        seed=config.seed,
    )

    model_rows = [
        score.to_dict()
        for score in evaluate_classical_baselines(
            dataset.x_train,
            dataset.y_train,
            dataset.x_test,
            dataset.y_test,
            config.seed,
        )
    ]

    train_states = statevectors(
        dataset.x_train,
        n_qubits=config.n_qubits,
        reps=config.feature_map_reps,
        feature_scale=config.feature_map_scale,
    )
    test_states = statevectors(
        dataset.x_test,
        n_qubits=config.n_qubits,
        reps=config.feature_map_reps,
        feature_scale=config.feature_map_scale,
    )
    quantum_kernel_score = evaluate_kernel_svm(
        fidelity_kernel(train_states),
        fidelity_kernel(test_states, train_states),
        dataset.y_train,
        dataset.y_test,
    )
    model_rows.append(quantum_kernel_score.to_dict())

    noise_rows = _noise_experiment(
        config,
        train_states,
        test_states,
        dataset.y_train,
        dataset.y_test,
    )

    vqc_results = [
        train_vqc(
            dataset.x_train,
            dataset.y_train,
            dataset.x_test,
            dataset.y_test,
            n_qubits=config.n_qubits,
            n_layers=config.vqc_layers,
            epochs=config.vqc_epochs,
            learning_rate=config.learning_rate,
            seed=seed,
        )
        for seed in config.vqc_seeds
    ]
    training_rows = [row for result in vqc_results for row in result.history]
    model_rows.append(
        {
            "model": "PennyLane-PyTorch VQC",
            "accuracy": float(np.mean([result.accuracy for result in vqc_results])),
            "f1": float(np.mean([result.f1 for result in vqc_results])),
            "accuracy_std": float(np.std([result.accuracy for result in vqc_results])),
        }
    )

    expressibility_rows = [
        expressibility_score(
            config.n_qubits,
            depth,
            config.expressibility_samples,
            config.seed + depth,
        ).to_dict()
        for depth in config.expressibility_depths
    ]

    _write_csv(output_dir / "model_comparison.csv", model_rows)
    _write_csv(output_dir / "noise_sweep.csv", noise_rows)
    _write_csv(output_dir / "vqc_training.csv", training_rows)
    _write_csv(output_dir / "expressibility.csv", expressibility_rows)
    plot_model_comparison(model_rows, output_dir / "model_comparison.png")
    plot_noise_sweep(noise_rows, output_dir / "noise_robustness.png")
    plot_training_curves(training_rows, output_dir / "vqc_training.png")
    plot_expressibility(expressibility_rows, output_dir / "expressibility.png")

    summary: dict[str, Any] = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "config": config.to_dict(),
        "versions": _versions(),
        "dataset": {
            "train_samples": len(dataset.x_train),
            "test_samples": len(dataset.x_test),
            "features": dataset.x_train.shape[1],
        },
        "model_comparison": model_rows,
        "noise_sweep": noise_rows,
        "expressibility": expressibility_rows,
        "interpretation_notes": [
            "Reported values are simulator results and do not establish quantum advantage.",
            (
                "The finite-shot noise sweep is a transparent depolarizing/readout proxy, "
                "not a device calibration."
            ),
            "VQC variability is summarized across explicit random seeds.",
            (
                "Lower expressibility KL means closer to Haar-random states, not necessarily "
                "better generalization."
            ),
        ],
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("quick", "full"), default="quick")
    parser.add_argument("--output-dir", type=Path, default=Path("results/quick"))
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = get_config(args.profile, seed=args.seed)
    summary = run_experiments(config, args.output_dir)
    print(json.dumps(summary["model_comparison"], indent=2))
    print(f"Artifacts written to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
