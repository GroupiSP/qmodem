from __future__ import annotations

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import pennylane as qp

from qmodem.module import ConvWeightGenerator, GaussianBlock, GeneratorConv1D


class WeightGenerator(nnx.Module):
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
        linear layer expands the output of the PQC to match the kernel size.

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
        return (self.kernel_size, self.in_features, self.out_features)

    @property
    def w_conv_size(self) -> int:
        return self.kernel_size * self.in_features * self.out_features

    def init_params(self, rngs: nnx.Rngs) -> None:
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

        return

    def __call__(self, rngs: nnx.Rngs) -> tuple[jax.Array, jax.Array]:
        """Generate the weights and bias for a 1D convolutional layer.

        Args:
            in_features: Number of input features (channels).
            out_features: Number of output features (filters).
            kernel_size: Size of the convolutional kernel.
            keys: Random number generator keys.
        """

        # Generate random input for the PQC
        x = rngs.uniform(shape=(), minval=0.0, maxval=2 * jnp.pi)

        # Evaluate the PQC
        pqc_out = jnp.array(self.circuit(x, self.params_circuit))  # (n_qubits,)

        # Linear transformation to map PQC outputs to convolutional weights
        w_conv = pqc_out @ self.params_linear_w + self.params_linear_b[None]

        return w_conv.reshape(
            self.w_conv_shape
        )  # (kernel_size, in_features, out_features)


class Net(nnx.Module):
    def __init__(
        self,
        n_filters: int,
        kernel_size: int,
        generator: ConvWeightGenerator,
        act_fn: nnx.Module = nnx.gelu,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """Bayesian 1D CNN for time-series RUL prediction with uncertainty.

        Architecture: BayesConv1D -> Activation -> Global Average Pooling ->
        GaussianBlock. Bayesian version of :class:`HeteroscedasticCNN1D`,
        trainable with ELBO loss (Bayes by Backprop). Accepts variable-length
        input windows.

        Args:
            bayes_conv: Bayesian convolution layer class
                (:class:`StandardBayesConv1D` or :class:`FlipoutConv1D`).
            n_filters: Number of convolutional filters. Defaults to 4.
            kernel_size: Size of the convolutional kernel. Defaults to 5.
            act_fn: Activation function. Defaults to ``nnx.gelu``.
            rngs: RNGs for the flax internal modules.
        """
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.act_fn = act_fn

        self.conv = GeneratorConv1D(
            in_features=1,
            out_features=n_filters,
            kernel_size=kernel_size,
            padding="VALID",
            generator=generator,
            rngs=rngs,
        )
        # GaussianBlock to output mean and variance
        self.gauss = GaussianBlock(n_filters, 1, rngs=rngs)

    def __call__(self, x: jax.Array, rngs: nnx.Rngs) -> jax.Array:
        """Forward pass through the Bayesian CNN.

        Args:
            x: Input with shape ``(batch, 1, window_size)``.
                Will be transposed to ``(batch, window_size, 1)``.
                Accepts variable-length windows.
            rngs: RNGs for weight sampling. The ``params`` stream is used
                to draw a key for the Bayesian convolution layer.

        Returns:
            Concatenated ``[mu, var_positive]`` with shape ``(batch, 2)``.
        """
        # Bayesian Conv1D with activation
        x = self.conv(x, rngs=rngs)
        x = self.act_fn(x)

        # Global Average Pooling: (batch, length, n_filters) -> (batch, n_filters)
        x = jnp.mean(x, axis=-2)

        # GaussianBlock: (batch, n_filters) -> (batch, 2)
        return self.gauss(x)
