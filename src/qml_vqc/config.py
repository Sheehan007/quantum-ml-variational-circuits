"""Configuration profiles for reproducible experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ExperimentConfig:
    """Settings shared by the complete experiment pipeline."""

    profile: str
    seed: int
    n_samples: int
    test_size: float
    noise: float
    n_qubits: int
    feature_map_reps: int
    feature_map_scale: float
    shots: int
    noise_repeats: int
    noise_levels: tuple[float, ...]
    vqc_layers: int
    vqc_epochs: int
    vqc_seeds: tuple[int, ...]
    learning_rate: float
    expressibility_samples: int
    expressibility_depths: tuple[int, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def get_config(profile: str, seed: int = 42) -> ExperimentConfig:
    """Return a quick CI-friendly or fuller research profile."""

    if profile == "quick":
        return ExperimentConfig(
            profile=profile,
            seed=seed,
            n_samples=72,
            test_size=0.30,
            noise=0.16,
            n_qubits=2,
            feature_map_reps=2,
            feature_map_scale=0.5,
            shots=512,
            noise_repeats=3,
            noise_levels=(0.0, 0.02, 0.05, 0.10),
            vqc_layers=2,
            vqc_epochs=35,
            vqc_seeds=(3, 11, 29),
            learning_rate=0.05,
            expressibility_samples=24,
            expressibility_depths=(1, 2, 3),
        )
    if profile == "full":
        return ExperimentConfig(
            profile=profile,
            seed=seed,
            n_samples=200,
            test_size=0.30,
            noise=0.18,
            n_qubits=2,
            feature_map_reps=3,
            feature_map_scale=0.5,
            shots=4096,
            noise_repeats=10,
            noise_levels=(0.0, 0.01, 0.02, 0.05, 0.10, 0.15),
            vqc_layers=3,
            vqc_epochs=120,
            vqc_seeds=(3, 7, 11, 19, 29),
            learning_rate=0.03,
            expressibility_samples=80,
            expressibility_depths=(1, 2, 3, 4),
        )
    raise ValueError(f"Unknown profile {profile!r}; choose 'quick' or 'full'.")
