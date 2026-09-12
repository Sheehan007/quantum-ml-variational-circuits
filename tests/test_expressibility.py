import math

from qml_vqc.quantum.expressibility import expressibility_score


def test_expressibility_score_is_finite() -> None:
    score = expressibility_score(n_qubits=2, depth=1, n_samples=8, seed=2, bins=5)
    assert score.depth == 1
    assert score.kl_to_haar >= 0.0
    assert math.isfinite(score.kl_to_haar)
    assert 0.0 <= score.mean_fidelity <= 1.0
