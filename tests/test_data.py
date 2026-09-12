import numpy as np

from qml_vqc.data import make_binary_dataset


def test_dataset_is_reproducible_and_stratified() -> None:
    first = make_binary_dataset(n_samples=60, seed=7)
    second = make_binary_dataset(n_samples=60, seed=7)
    np.testing.assert_allclose(first.x_train, second.x_train)
    np.testing.assert_array_equal(first.y_test, second.y_test)
    assert set(first.y_train) == {0, 1}
    assert set(first.y_test) == {0, 1}
    assert first.x_train.shape == (42, 2)
    assert first.x_test.shape == (18, 2)


def test_angles_are_bounded() -> None:
    split = make_binary_dataset(n_samples=50, seed=5)
    assert np.max(split.x_train) <= np.pi
    assert np.min(split.x_train) >= -np.pi
    assert np.max(split.x_test) <= np.pi
    assert np.min(split.x_test) >= -np.pi
