from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx
from jax.flatten_util import ravel_pytree
from stagereg.jax import Gamma, GammaHead

from qmodem.avi import LatentMap

from .data_processing import WINDOW_SIZE

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


class MCDropoutNet(nnx.Module):
    """MCD posterior network: the mean network with input dropout before the conv.

    ``Dropout(rate) -> Conv1D -> act -> GAP -> Linear(n_filters, 1)``. The
    dropout is elementwise over time and channels and its mask is broadcast over
    the batch axis, so one mask set is one function of the inputs. Dropout has
    no parameters, so the parameter tree is the mean network's.
    """

    def __init__(
        self,
        kernel_size: int,
        n_filters: int,
        activation: Activation,
        dropout_rate: float,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize the network.

        Args:
            kernel_size: Width of the 1D convolution kernel.
            n_filters: Number of convolution filters.
            activation: Activation applied after the convolution.
            dropout_rate: Probability of dropping an input element.
            rngs: RNGs of the parameter initializers and of the dropout masks.
        """
        self.dropout = nnx.Dropout(dropout_rate, broadcast_dims=(0,), rngs=rngs)
        self.conv = _conv(kernel_size, n_filters, rngs)
        self.activation = activation
        self.linear = nnx.Linear(n_filters, 1, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Predict the mean RUL under the current dropout mask.

        Args:
            x: Windows of shape ``(B, W, IN_FEATURES)``.

        Returns:
            Means of shape ``(B, 1)``.
        """
        return self.linear(
            _pooled_conv_features(self.conv, self.activation, self.dropout(x))
        )


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


POSTPROCESSOR_SCALE = 1e-2
"""The fixed factor α of the postprocessor's He-normal initial weights."""

DISCRIMINATOR_HIDDEN = (64, 64)
"""Hidden widths of the (fixed, untuned) discriminator."""


class ConvGenerator(nnx.Module):
    """The AVI generator of the battery CNN: latent map -> postprocessor -> conv layer.

    The postprocessor is a linear map from the latent vector to the flattened
    inferred subnetwork (the conv kernel **and** bias), unflattened to its tree. Its
    bias starts at the flattened warm start and its weights at He-normal × ``scale``,
    so every initial draw is close to the trained mean's conv layer.
    """

    def __init__(
        self,
        latent_map: LatentMap,
        warm_start: Any,
        *,
        scale: float = POSTPROCESSOR_SCALE,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize the postprocessor around the warm start.

        Args:
            latent_map: Maps the random scalar to a latent vector of width ``q``.
            warm_start: The inferred subnetwork's trained-mean parameters.
            scale: The fixed factor α of the initial weights.
            rngs: RNGs of the weight initializer.
        """
        flat, self.unflatten = ravel_pytree(warm_start)
        self.latent_map = latent_map
        self.weights = nnx.Param(
            scale
            * jax.nn.initializers.he_normal()(
                rngs.params(), (latent_map.width, flat.size), flat.dtype
            )
        )
        self.bias = nnx.Param(jnp.copy(flat))

    def __call__(self, z: jax.Array) -> Any:
        """Generate one draw of the conv layer.

        Args:
            z: A scalar random input.

        Returns:
            The conv layer's parameters, in the warm start's tree.
        """
        return self.unflatten(self.latent_map(z) @ self.weights[...] + self.bias[...])


class WindowDiscriminator(nnx.Module):
    """The AVI discriminator: a 64-64 LeakyReLU MLP over a window and its target.

    It flattens a window ``(B, W, C)``, appends the target ``(B, 1)`` and returns the
    logit that the pair is real.
    """

    def __init__(
        self,
        window_size: int = WINDOW_SIZE,
        in_features: int = IN_FEATURES,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize the MLP.

        Args:
            window_size: Window length W.
            in_features: Input channels C.
            rngs: RNGs of the parameter initializers.
        """
        widths = (window_size * in_features + 1, *DISCRIMINATOR_HIDDEN)
        self.hidden = DISCRIMINATOR_HIDDEN
        self.layers = nnx.List(
            [
                nnx.Linear(n_in, n_out, rngs=rngs)
                for n_in, n_out in zip(widths[:-1], widths[1:], strict=True)
            ]
        )
        self.output = nnx.Linear(widths[-1], 1, rngs=rngs)

    def __call__(self, inputs: jax.Array, targets: jax.Array) -> jax.Array:
        """Score window-target pairs.

        Args:
            inputs: Windows ``(B, W, C)``.
            targets: Targets ``(B, 1)``, real or generated.

        Returns:
            Logits of ``P(real)``, shape ``(B, 1)``.
        """
        x = jnp.concatenate([inputs.reshape((inputs.shape[0], -1)), targets], axis=-1)
        for layer in self.layers:
            x = nnx.leaky_relu(layer(x), negative_slope=0.2)
        return self.output(x)
