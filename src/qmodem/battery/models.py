from __future__ import annotations

import functools
from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import nnx
from stagereg.jax import Gamma, GammaHead

from qmodem.pennylane_typing import Device
from qmodem.quantum_circuits import (
    BinaryCircuitFactory,
    ContinuousCircuitFactory,
)

IN_FEATURES = 2
"""Input channels of a battery window (load, voltage)."""

type Activation = Callable[[jax.Array], jax.Array]


def _pooled_conv_features(
    conv: nnx.Conv, activation: Activation, x: jax.Array
) -> jax.Array:
    """Apply ``Conv1D -> activation -> global average pooling``.

    Args:
        conv: The convolution, mapping ``(B, W, C)`` to ``(B, W', F)``.
        activation: Elementwise activation.
        x: Input windows of shape ``(B, W, C)``.

    Returns:
        Pooled features of shape ``(B, F)``.
    """
    return jnp.mean(activation(conv(x)), axis=-2)


def _conv(kernel_size: int, n_filters: int, rngs: nnx.Rngs) -> nnx.Conv:
    return nnx.Conv(
        in_features=IN_FEATURES,
        out_features=n_filters,
        kernel_size=kernel_size,
        padding="VALID",
        rngs=rngs,
    )


class MeanNet(nnx.Module):
    """Mean network ``Conv1D -> act -> GAP -> Linear(n_filters, 1)``.

    It is deterministic (no dropout) and outputs the mean only, so the shared
    mean checkpoint is method-neutral. The convolution is the attribute ``conv``,
    which the posterior stages select as the inferred subnetwork.
    """

    def __init__(
        self,
        kernel_size: int,
        n_filters: int,
        activation: Activation,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize the network.

        Args:
            kernel_size: Width of the 1D convolution kernel.
            n_filters: Number of convolution filters.
            activation: Activation applied after the convolution.
            rngs: RNGs of the parameter initializers.
        """
        self.conv = _conv(kernel_size, n_filters, rngs)
        self.activation = activation
        self.linear = nnx.Linear(n_filters, 1, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Predict the mean RUL.

        Args:
            x: Windows of shape ``(B, W, IN_FEATURES)``.

        Returns:
            Means of shape ``(B, 1)``.
        """
        return self.linear(_pooled_conv_features(self.conv, self.activation, x))


class VarianceNet(nnx.Module):
    """Variance network ``Conv1D -> act -> GAP -> GammaHead``.

    It has the mean network's backbone shape but its own weights, and outputs a Gamma
    distribution over the squared residual (stagereg's convention).
    """

    def __init__(
        self,
        kernel_size: int,
        n_filters: int,
        activation: Activation,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize the network.

        Args:
            kernel_size: Width of the 1D convolution kernel.
            n_filters: Number of convolution filters.
            activation: Activation applied after the convolution.
            rngs: RNGs of the parameter initializers.
        """
        self.conv = _conv(kernel_size, n_filters, rngs)
        self.activation = activation
        self.head = GammaHead(n_filters, 1, rngs=rngs)

    def __call__(self, x: jax.Array) -> Gamma:
        """Predict the distribution of the squared residual.

        Args:
            x: Windows of shape ``(B, W, IN_FEATURES)``.

        Returns:
            A Gamma distribution with batch shape ``(B, 1)``.
        """
        return self.head(_pooled_conv_features(self.conv, self.activation, x))


class Discriminator(nnx.Module):
    def __init__(
        self,
        input_dim: int,
        hidden: int = 64,
        act_fn: Callable[[jax.Array], jax.Array] | None = None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """Discriminator network for adversarial training. The input is expected to have
        shape (batch, window_size, channels+1), where the extra channel is meant to be
        the output value (y, RUL).

        The nework outputs a single value corresponding to the probability of the input being real (1) or fake (0).

        Args:
            input_dim (int): Input dimension (window_size * channels + 1).
            hidden (int, optional): Number of hidden units. Defaults to 64.
            act_fn (Callable[[jax.Array], jax.Array] | None, optional): Activation function for hidden layers.
                If ``None``, uses ``leaky_relu`` with ``negative_slope=0.2``.
            rngs (nnx.Rngs): RNGs for the flax internal modules.
        """
        self.l1 = nnx.Linear(input_dim, hidden, rngs=rngs)
        self.l2 = nnx.Linear(hidden, hidden, rngs=rngs)
        self.l3 = nnx.Linear(hidden, 1, rngs=rngs)
        self.act_fn = act_fn or functools.partial(nnx.leaky_relu, negative_slope=0.2)

    def __call__(self, x: jax.Array, rngs: nnx.Rngs) -> jax.Array:
        """Forward pass through the discriminator.

        Args:
            x (jax.Array): Input with shape (batch, window_size, channels).
            rngs (nnx.Rngs): RNGs for the flax internal modules.

        Returns:
            jax.Array: Output
        """
        x = x.reshape((x.shape[0], -1))  # Flatten the time and channels
        x = self.act_fn(self.l1(x))
        x = self.act_fn(self.l2(x))
        return nnx.sigmoid(self.l3(x))


class BaseConvGenerator(nnx.Module):
    def __init__(
        self,
        kernel_size: int,
        in_features: int,
        out_features: int,
    ) -> None:
        """Base class for generating weights for a 1D convolutional layer.

        Args:
            kernel_size: Size of the convolutional kernel.
            in_features: Number of input features (channels).
            out_features: Number of output features (filters).
        """
        self.kernel_size = kernel_size
        self.in_features = in_features
        self.out_features = out_features

    @property
    def w_conv_shape(self) -> tuple[int, int, int]:
        """Utility property to get the shape of the convolutional weights generated by
        this module.

        Returns:
            tuple[int, int, int]: Shape of the convolutional weights (kernel_size, in_features, out_features).
        """
        return (self.kernel_size, self.in_features, self.out_features)

    @property
    def w_conv_size(self) -> int:
        """Utility property to get the size of the convolutional weights generated by
        this module.

        Returns:
            int: Size of the convolutional weights (kernel_size * in_features * out_features).
        """
        return self.kernel_size * self.in_features * self.out_features


class ContinuousWeightsGenerator(BaseConvGenerator):
    def __init__(
        self,
        circuit_factory: ContinuousCircuitFactory,
        device: Device,
        kernel_size: int,
        in_features: int,
        out_features: int,
    ) -> None:
        """Generates weights for a 1D convolutional layer using a parameterized quantum
        circuit (PQC) and a linear layer. The generated weights are those of the kernel,
        while the bias is assumed to be generated separately or deterministic. The
        linear layer expands the output of the PQC to match the size of the
        convolutional weights ($W_{conv}$).

        The PQC maps a scalar sample from a classical distribution (e.g., uniform or normal) to a quantum state.
        The expectation values of the Pauli-Z operators for the single qubits (e.g. ZIII, IZII, IIZI, IIIZ) are then
        measured and later linearly transformed and expanded.

        Args:
            circuit_factory: A factory that returns a parameterized quantum circuit (PQC) and contains properties useful to generate the weights of the convolutional layer.
            device: A PennyLane device on which the PQC will be executed.
            kernel_size: Size of the convolutional kernel.
            in_features: Number of input features (channels).
            out_features: Number of output features (filters).
        """
        self.circuit = circuit_factory.make_circuit(device=device)
        self.params_shape = circuit_factory.params_example.shape
        self.n_qubits = circuit_factory.n_qubits

        super().__init__(
            kernel_size=kernel_size,
            in_features=in_features,
            out_features=out_features,
        )

    def init_params(self, rngs: nnx.Rngs) -> None:
        """Initialize the parameters of the PQC and the linear layer."""
        # Parameters of the PQC
        self.params_circuit = nnx.Param(
            rngs.params.uniform(
                shape=self.params_shape,
                minval=0.0,
                maxval=2 * jnp.pi,
            )
        )

        # Paramers for the linear layer that maps PQC outputs to convolutional weights
        init_func_linear_w = jax.nn.initializers.he_normal()
        self.params_linear_w = nnx.Param(
            init_func_linear_w(rngs.params(), (self.n_qubits, self.w_conv_size))
        )
        self.params_linear_b = nnx.Param(jnp.zeros((self.w_conv_size,)))

    def __call__(self, rngs: nnx.Rngs) -> jax.Array:
        """Generate the weights for a 1D convolutional layer."""

        # Generate random scalar input for the PQC
        x = rngs.uniform(shape=(), minval=0.0, maxval=2 * jnp.pi)

        # Evaluate the PQC
        pqc_out = jnp.stack(self.circuit(x, self.params_circuit))  # (n_qubits,)

        # Linear transformation to map PQC outputs to convolutional weights
        w_conv = pqc_out @ self.params_linear_w + self.params_linear_b[None]

        return w_conv.reshape(
            self.w_conv_shape
        )  # (kernel_size, in_features, out_features)


class BinaryWeightsGenerator(BaseConvGenerator):
    def __init__(
        self,
        circuit_factory: BinaryCircuitFactory,
        device: Device,
        kernel_size: int,
        in_features: int,
        out_features: int,
        n_shots: int = 1,
    ) -> None:
        """Generates binary weights for a 1D convolutional layer using a parameterized
        quantum circuit (PQC) and a linear layer. The generated weights are those of the
        kernel, while the bias is assumed to be generated separately or deterministic.

        A separate PQC is used for each output feature (filter). The number of qubits of the PQC is determined by the kernel size and the number of input features, i.e. n_qubits = kernel_size * in_features.

        Each PQC acts as a Quantum Circuit Born Machine (QCBM), i.e. a parametrized distribution over
        the basis states (bitstrings) of the qubits. The weight sample is a single measurement on
        the Z-basis of each qubit. No further transformation is applied to the output of the PQC.

        Args:
            circuit_factory: A factory that returns a parameterized quantum circuit (PQC) and contains properties useful to generate the weights of the convolutional layer.
            device: A PennyLane device on which the PQC will be executed.
            kernel_size: Size of the convolutional kernel.
            in_features: Number of input features (channels).
            out_features: Number of output features (filters).
            n_shots: Number of shots to use for the PQC. Defaults to 1.
        """
        if circuit_factory.n_qubits != kernel_size * in_features:
            raise ValueError(
                f"Number of qubits in the circuit ({circuit_factory.n_qubits}) must match kernel_size * in_features ({kernel_size * in_features})."
            )

        self.n_qubits = circuit_factory.n_qubits
        self.params_shape = circuit_factory.params_example.shape
        self.circuits = [
            circuit_factory.make_circuit(device=device, n_shots=n_shots)
            for _ in range(out_features)
        ]

        super().__init__(
            kernel_size=kernel_size,
            in_features=in_features,
            out_features=out_features,
        )

    def init_params(self, rngs: nnx.Rngs) -> None:
        """Initialize the parameters of the PQC and the linear layer."""
        # Parameters of the PQC
        self.params_circuits = [
            nnx.Param(
                rngs.params.uniform(
                    shape=self.params_shape,
                    minval=0.0,
                    maxval=2 * jnp.pi,
                )
            )
            for _ in range(self.out_features)
        ]

        self.params_linear_b = nnx.Param(jnp.zeros(shape=self.w_conv_shape))

    def __call__(self, rngs: nnx.Rngs) -> jax.Array:
        """Generate the weights for a 1D convolutional layer."""

        w_conv = jnp.zeros(shape=self.w_conv_shape)

        # Evaluate the PQCs
        for i in range(self.out_features):
            pqc_out = self.circuits[i](self.params_circuits[i])
            filter_i = pqc_out.reshape((self.kernel_size, self.in_features))
            w_conv = w_conv.at[:, :, i].set(filter_i)

        return w_conv + self.params_linear_b
