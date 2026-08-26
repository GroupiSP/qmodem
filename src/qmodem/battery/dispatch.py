from __future__ import annotations

import functools

import pennylane as qp
from flax import nnx
from pydantic import BaseModel, ConfigDict

from qmodem.quantum_circuits import ContinuousCircuitFactory

from .models import CNN, ContinuousWeightsGenerator, ConvType, Discriminator
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


def build_model(parameters: ModelBuildParameters) -> CNN:
    """Selects and returns the correct model based on the training hyperparameters.

    Note: this function is only meant for models that are trained in a fully supervised manner.
    If you are using QAVI, train it adversarially in its own script.
    """
    # TODO: Add in_features to the model build parameters
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


def build_qavi_model(hp: QAVITrainHyperparameters) -> CNN:
    """Builds and returns the quantum-generated CNN for QAVI training.

    Args:
        hp: QAVI training hyperparameters.

    Returns:
        A ``CNN`` instance with a quantum-generated convolutional layer.
    """
    # TODO: Add in_features to hyperparameters
    circuit_factory = ContinuousCircuitFactory(
        n_qubits=hp.pqc_n_qubits, n_layers=hp.pqc_n_layers
    )
    device = qp.device("default.qubit", wires=hp.pqc_n_qubits)
    weight_generator = ContinuousWeightsGenerator(
        circuit_factory=circuit_factory,
        device=device,
        kernel_size=hp.conv_kernel_size,
        in_features=2,
        out_features=hp.conv_n_filters,
    )
    return CNN(
        conv_type=ConvType.QUANTUM_GENERATED,
        in_features=2,
        n_filters=hp.conv_n_filters,
        kernel_size=hp.conv_kernel_size,
        dropout_rate=hp.dropout_rate,
        generator=weight_generator,
        act_fn=getattr(nnx, hp.activation_function),
        rngs=nnx.Rngs(hp.net_init_seed),
    )


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
        discriminator_act_fn = getattr(nnx, hp.discriminator_act_fn)

    return Discriminator(
        input_dim=2 * hp.window_size + 1,  # 2 channels (load, voltage) + 1 RUL
        hidden=hp.discriminator_hidden_size,
        act_fn=discriminator_act_fn,
        rngs=nnx.Rngs(hp.discriminator_init_seed),
    )
