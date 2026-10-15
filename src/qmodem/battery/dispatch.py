from __future__ import annotations

import functools

from flax import nnx
from pydantic import BaseModel, ConfigDict

from .models import Discriminator
from .train import Method, QAVITrainHyperparameters


class ModelBuildParameters(BaseModel):
    """Adapter class to hold the parameters that define the data-driven model."""

    model_config = ConfigDict(extra="ignore")

    method: Method
    conv_n_filters: int
    conv_kernel_size: int
    dropout_rate: float
    activation_function: str
    net_init_seed: int


def build_discriminator(hp: QAVITrainHyperparameters) -> Discriminator:
    """Builds and returns the discriminator for QAVI adversarial training.

    Args:
        hp: QAVI training hyperparameters.

    Returns:
        A ``Discriminator`` instance sized for the window and feature dimensions.
    """
    if hp.discriminator_act_fn == "leaky_relu":
        discriminator_act_fn = functools.partial(nnx.leaky_relu, negative_slope=0.2)
    else:
        discriminator_act_fn = getattr(nnx, hp.discriminator_act_fn, None)
        if discriminator_act_fn is None:
            raise ValueError(
                f"Unsupported discriminator activation function: {hp.discriminator_act_fn}"
            )

    return Discriminator(
        input_dim=2 * hp.window_size + 1,  # 2 channels (load, voltage) + 1 RUL
        hidden=hp.discriminator_hidden_size,
        act_fn=discriminator_act_fn,
        rngs=nnx.Rngs(hp.discriminator_init_seed),
    )
