from __future__ import annotations

from enum import StrEnum, auto

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import pennylane as qp

from qmodem.module import (
    ConvWeightGenerator,
    GaussianBlock,
    GeneratorConv1D,
    StandardBayesConv1D,
)
from qmodem.module import (
    mc_sample as _mc_sample,
)

# TODO: There should be a single backbone CNN class that includes dropout (MCD only if mode=train during inference)
# TODO: The final layer should be an input argument to the network initializer.
# TODO: The convolutional layer should be an input argument to the network initializer (standard, bayesian, flipout, generator).


class Discriminator(nnx.Module):
    def __init__(self, input_dim: int, hidden: int = 64, *, rngs: nnx.Rngs) -> None:
        """Discriminator network for adversarial training. The input is expected to have
        shape (batch, window_size, channels+1), where the extra channel is meant to be
        the output value (y, RUL).

        The nework outputs a single value corresponding to the probability of the input being real (1) or fake (0).

        Args:
            input_dim (int): Input dimension (window_size * channels + 1).
            hidden (int, optional): Number of hidden units. Defaults to 64.
            rngs (nnx.Rngs): RNGs for the flax internal modules.
        """
        self.l1 = nnx.Linear(input_dim, hidden, rngs=rngs)
        self.l2 = nnx.Linear(hidden, hidden, rngs=rngs)
        self.l3 = nnx.Linear(hidden, 1, rngs=rngs)

    def __call__(self, x: jax.Array, rngs: nnx.Rngs) -> jax.Array:
        """Forward pass through the discriminator.

        Args:
            x (jax.Array): Input with shape (batch, window_size, channels).
            rngs (nnx.Rngs): RNGs for the flax internal modules.

        Returns:
            jax.Array: Output
        """
        x = x.reshape((x.shape[0], -1))  # Flatten the time and channels
        x = nnx.leaky_relu(self.l1(x), negative_slope=0.2)
        x = nnx.leaky_relu(self.l2(x), negative_slope=0.2)
        return nnx.sigmoid(self.l3(x))


class ConvType(StrEnum):
    DETERMINISTIC = auto()
    BAYESIAN = auto()
    QUANTUM_GENERATED = auto()


class _DetConv1D(nnx.Module):
    def __init__(self, **kwargs) -> None:
        """Thin wrapper to filter the rngs at call time for the deterministic conv
        layer."""
        self._conv = nnx.Conv(**kwargs)

    def __call__(self, x: jax.Array, rngs: nnx.Rngs | None = None) -> jax.Array:
        return self._conv(x)


def _conv_layer_factory(conv_type: ConvType, **kwargs) -> nnx.Module:
    if conv_type == ConvType.DETERMINISTIC:
        return _DetConv1D(**kwargs)
    elif conv_type == ConvType.BAYESIAN:
        return StandardBayesConv1D(**kwargs)
    elif conv_type == ConvType.QUANTUM_GENERATED:
        generator = kwargs.pop("generator")
        return GeneratorConv1D(generator=generator, **kwargs)
    else:
        raise ValueError(f"Unsupported convolution type: {conv_type}")


class CNN(nnx.Module):
    def __init__(
        self,
        conv_type: ConvType = ConvType.DETERMINISTIC,
        in_features: int = 1,
        n_filters: int = 4,
        kernel_size: int = 5,
        dropout_rate: float = 0.1,
        act_fn: nnx.Module = nnx.gelu,
        generator: ConvWeightGenerator | None = None,
        mcd: bool = False,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """1D CNN for time-series RUL prediction with uncertainty.

        Architecture: Conv1D -> Activation -> Dropout -> Global Average Pooling ->
        GaussianBlock. Accepts variable-length input windows.

        Args:
            conv_type (ConvType, optional): Type of convolutional layer. Defaults to DETERMINISTIC.
            input_dim (int, optional): Number of input features. Defaults to 1.
            n_filters (int, optional): Number of convolutional filters. Defaults to 4.
            kernel_size (int, optional): Size of the convolutional kernel. Defaults to 5.
            dropout_rate (float, optional): Dropout rate. Defaults to 0.1.
            act_fn (nnx.Module, optional): Activation function. Defaults to nnx.gelu.
            generator (ConvWeightGenerator | None, optional): Weight generator for quantum-generated convolution.
                Required if conv_type is QUANTUM_GENERATED, ignored otherwise.
            mcd (bool, optional): Whether to use Monte Carlo Dropout. Defaults to False.
            rngs (nnx.Rngs): RNGs for the flax internal modules.
        """
        if conv_type == ConvType.QUANTUM_GENERATED and generator is None:
            raise ValueError(
                "generator must be provided when conv_type is QUANTUM_GENERATED"
            )

        self.act_fn = act_fn
        self.mcd = mcd

        conv_kwargs = {
            "in_features": in_features,
            "out_features": n_filters,
            "kernel_size": kernel_size,
            "padding": "VALID",
            "rngs": rngs,
        }

        if conv_type == ConvType.QUANTUM_GENERATED:
            conv_kwargs["generator"] = generator

        self.conv = _conv_layer_factory(conv_type, **conv_kwargs)
        self.dropout = nnx.Dropout(dropout_rate)

        # GaussianBlock to output mean and variance
        self.gauss = GaussianBlock(n_filters, 1, rngs=rngs)

    def __call__(self, x: jax.Array, rngs: nnx.Rngs) -> jax.Array:
        """Forward pass through the CNN.

        Args:
            x (jax.Array): Input with shape (batch, window_size, in_features).
                           Accepts variable-length windows.
            rngs (nnx.Rngs): RNGs for dropout sampling and weight sampling if applicable.

        Returns:
            jax.Array: Concatenated [mu, var_positive] with shape (batch, 2).
        """
        # Conv1D with activation and dropout
        x = self.conv(x, rngs=rngs)
        x = self.act_fn(x)
        x = self.dropout(x, rngs=rngs)

        # Global Average Pooling: (batch, window_size, n_filters) -> (batch, n_filters)
        x = jnp.mean(x, axis=-2)

        # GaussianBlock: (batch, n_filters) -> (batch, 2)
        return self.gauss(x)

    def mc_sample(self, key: jax.Array, X: jax.Array, n_samples: int) -> jax.Array:
        """Draw ``n_samples`` Monte Carlo predictions for a single input window.

        Args:
            key: PRNG key.
            X: Input with shape ``(1, window_size, in_features)``.
            n_samples: Number of Monte Carlo samples.
        """
        splits = jax.random.split(key, num=2 * n_samples)
        keys_weights = splits[:n_samples]
        keys_noise = splits[n_samples:]
        return _mc_sample(self, X, keys_weights, keys_noise)

    def train(self, **attributes) -> CNN:
        """Set the module to training mode.

        Args:
            **attributes: Additional attributes to set.

        Returns:
            The module itself.
        """
        return super().train(**attributes)

    def eval(self, **attributes) -> CNN:
        """Set the module to evaluation mode, unless Monte Carlo Dropout is enabled, in
        which case the model remains in training mode to allow Monte Carlo sampling of
        the activations.

        Args:
            **attributes: Additional attributes to set.

        Returns:
            The module itself.
        """
        if self.mcd:
            # If Monte Carlo Dropout is enabled, keep the module in training mode during evaluation
            return super().train(**attributes)

        return super().eval(**attributes)


class CNNForELBO(nnx.Module):
    @staticmethod
    def _validate_cnn(cnn: CNN) -> None:
        if not hasattr(cnn.conv, "kl_divergence"):
            raise ValueError("The provided CNN does not have a kl_divergence method. ")
        if not hasattr(cnn.conv, "mean_posterior_variance"):
            raise ValueError(
                "The provided CNN does not have a mean_posterior_variance method."
            )

    def __init__(self, cnn: CNN) -> None:
        """Wrapper around CNN to expose KL divergence for Bayesian layers.

        Args:
            cnn (CNN): An instance of the CNN class.
        """
        self._validate_cnn(cnn)
        self.cnn = cnn

    def __call__(self, x: jax.Array, rngs: nnx.Rngs) -> jax.Array:
        return self.cnn(x, rngs=rngs)

    def kl_divergence(self) -> jax.Array:
        """Total KL divergence across all Bayesian layers in the CNN."""
        return self.cnn.conv.kl_divergence()

    def conv_mean_posterior_variance(self) -> jax.Array:
        """Mean posterior variance of the convolutional layer in the CNN."""
        return self.cnn.conv.mean_posterior_variance()

    def mc_sample(self, key: jax.Array, X: jax.Array, n_samples: int) -> jax.Array:
        """Draw ``n_samples`` Monte Carlo predictions for a single input window.

        Args:
            key: PRNG key.
            X: Input with shape ``(1, window_size, in_features)``.
            n_samples: Number of Monte Carlo samples.
        """
        return self.cnn.mc_sample(key, X, n_samples)


class ContinuousWeightsGenerator(nnx.Module):
    def __init__(
        self,
        n_qubits: int,
        n_layers: int,
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
            n_qubits: Number of qubits in the PQC.
            n_layers: Number of layers in the PQC.
            kernel_size: Size of the convolutional kernel.
            in_features: Number of input features (channels).
            out_features: Number of output features (filters).
        """
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.in_features = in_features
        self.out_features = out_features

        @qp.qnode(device=qp.device("default.qubit", wires=n_qubits), interface="jax")
        def circuit(x: jax.Array, params: jax.Array) -> list[float]:
            """Parameterized quantum circuit (PQC) that generates a quantum state based
            on the input x and parameters.

            x is assumed to be a sample from a given distribution (e.g., uniform or
            normal) and is used to encode the input into the quantum state.
            """
            for i in range(n_qubits):
                qp.RY(x, wires=i)
            for layer in range(n_layers):
                for q in range(n_qubits):
                    qp.RY(params[layer, q, 0], wires=q)
                    qp.RZ(params[layer, q, 1], wires=q)
                for q in range(n_qubits):
                    qp.CNOT(wires=[q, (q + 1) % n_qubits])
            return [qp.expval(qp.PauliZ(i)) for i in range(n_qubits)]

        self.circuit = circuit

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

    def init_params(self, rngs: nnx.Rngs) -> None:
        """Initialize the parameters of the PQC and the linear layer."""
        # Parameters of the PQC
        self.params_circuit = nnx.Param(
            rngs.params.uniform(
                shape=(self.n_layers, self.n_qubits, 2),
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
        """Generate the weights for a 1D convolutional layer.

        Args:
            in_features: Number of input features (channels).
            out_features: Number of output features (filters).
            kernel_size: Size of the convolutional kernel.
            keys: Random number generator keys.
        """

        # Generate random scalar input for the PQC
        x = rngs.uniform(shape=(), minval=0.0, maxval=2 * jnp.pi)

        # Evaluate the PQC
        pqc_out = jnp.array(self.circuit(x, self.params_circuit))  # (n_qubits,)

        # Linear transformation to map PQC outputs to convolutional weights
        w_conv = pqc_out @ self.params_linear_w + self.params_linear_b[None]

        return w_conv.reshape(
            self.w_conv_shape
        )  # (kernel_size, in_features, out_features)
