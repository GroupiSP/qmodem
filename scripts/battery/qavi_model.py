from __future__ import annotations

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import jaxtyping
import pennylane as qp

from qmodem.module import PQC, GaussianBlock, PQCConv1D


class LayeredPQC:
    def __init__(self, n_qubits: int, n_layers: int):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.params_shape = (n_layers, n_qubits, 2)  # 2 for RY and RZ angles
        self.device = qp.device("default.qubit", wires=n_qubits)
        self.circuit = qp.qnode(self.device, interface="jax")(self._circuit)

    def _circuit(self, x: float, params: jaxtyping.ArrayLike) -> list[float]:
        for i in range(self.n_qubits):
            qp.RY(x, wires=i)
        for layer in range(self.n_layers):
            for q in range(self.n_qubits):
                qp.RY(params[layer, q, 0], wires=q)
                qp.RZ(params[layer, q, 1], wires=q)
            for q in range(self.n_qubits):
                qp.CNOT(wires=[q, (q + 1) % self.n_qubits])
        return [qp.expval(qp.PauliZ(i)) for i in range(self.n_qubits)]

    def __call__(self, x: jax.Array, params: jax.Array) -> jax.Array:
        return jnp.array(self.circuit(x, params))


class Net(nnx.Module):
    def __init__(
        self,
        n_filters: int = 4,
        kernel_size: int = 5,
        act_fn: nnx.Module = nnx.gelu,
        quantum_circuit: PQC = LayeredPQC(n_qubits=5, n_layers=1),
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

        self.conv = PQCConv1D(
            in_features=1,
            out_features=n_filters,
            kernel_size=kernel_size,
            padding="VALID",
            quantum_circuit=quantum_circuit,
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
