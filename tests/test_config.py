import pytest

from qml_vqc.config import get_config


def test_quick_profile_is_smaller_than_full() -> None:
    quick = get_config("quick")
    full = get_config("full")
    assert quick.n_samples < full.n_samples
    assert quick.vqc_epochs < full.vqc_epochs
    assert quick.noise_repeats < full.noise_repeats


def test_unknown_profile_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown profile"):
        get_config("overnight")
