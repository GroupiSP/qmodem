from __future__ import annotations

from flax import nnx
from pydantic import BaseModel, ConfigDict

from .models import CNN, ConvType
from .train import Method


class ModelBuildParameters(BaseModel):
    """Adapter class to hold the parameters that define the data-driven model."""

    model_config = ConfigDict(extra="ignore")

    method: Method
    conv_n_filters: int
    conv_kernel_size: int
    dropout_rate: float
    activation_function: str
    net_init_seed: int


def build_model(parameters: ModelBuildParameters) -> CNN:
    """Selects and returns the correct model based on the training hyperparameters.

    Note: this function is only meant for models that are trained in a fully supervised manner.
    If you are using QAVI, train it adversarially in its own script.
    """
    match parameters.method:
        case Method.HNN:
            return CNN(
                conv_type=ConvType.DETERMINISTIC,
                in_features=2,
                n_filters=parameters.conv_n_filters,
                kernel_size=parameters.conv_kernel_size,  # Integer kernel size means 1D convolution, tuple means 2D convolution
                dropout_rate=parameters.dropout_rate,
                act_fn=getattr(nnx, parameters.activation_function),
                mcd=False,
                rngs=nnx.Rngs(parameters.net_init_seed),
            )
        case Method.MCD:
            return CNN(
                conv_type=ConvType.DETERMINISTIC,
                in_features=2,
                n_filters=parameters.conv_n_filters,
                kernel_size=parameters.conv_kernel_size,  # Integer kernel size means 1D convolution, tuple means 2D convolution
                dropout_rate=parameters.dropout_rate,
                act_fn=getattr(nnx, parameters.activation_function),
                mcd=True,
                rngs=nnx.Rngs(parameters.net_init_seed),
            )
        case Method.BNN:
            return CNN(
                conv_type=ConvType.BAYESIAN,
                in_features=2,
                n_filters=parameters.conv_n_filters,
                kernel_size=parameters.conv_kernel_size,  # Integer kernel size means 1D convolution, tuple means 2D convolution
                dropout_rate=parameters.dropout_rate,
                act_fn=getattr(nnx, parameters.activation_function),
                mcd=False,
                rngs=nnx.Rngs(parameters.net_init_seed),
            )
        case Method.QAVI:
            raise ValueError(
                "QAVI is not supported in this function. Please run adversarial training instead."
            )
