from __future__ import annotations

from qmodem.battery.dispatch import build_discriminator
from qmodem.battery.train import QAVITrainHyperparameters

# ---------------------------------------------------------------------------
# dispatch factory tests
# ---------------------------------------------------------------------------


def test_build_discriminator_returns_discriminator_with_correct_input_dim() -> None:
    hp = QAVITrainHyperparameters(
        window_size=10,
        discriminator_hidden_size=32,
    )

    discriminator = build_discriminator(hp)

    # input_dim == 2 * window_size + 1
    expected_input_dim = 2 * hp.window_size + 1
    assert discriminator.l1.in_features == expected_input_dim
    assert discriminator.l1.out_features == hp.discriminator_hidden_size


def test_build_discriminator_uses_configured_activation() -> None:
    hp = QAVITrainHyperparameters(discriminator_act_fn="relu")

    discriminator = build_discriminator(hp)

    assert discriminator.act_fn is not None
    assert discriminator.act_fn.__name__ == "relu"
