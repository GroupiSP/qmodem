from __future__ import annotations

from typing import Optional, Sequence

import jax
import jax.numpy as jnp
from flax import nnx


class GaussianLayer(nnx.Module):
    def __init__(self, input_dim: int, output_dim: int, *, rngs: nnx.Rngs) -> None:
        self.linear_1 = nnx.Linear(input_dim, output_dim, rngs=rngs)
        self.linear_2 = nnx.Linear(input_dim, output_dim, rngs=rngs)

    def __call__(self, x: jax.Array, rngs: Optional[nnx.Rngs] = None) -> jax.Array:
        mu = self.linear_1(x)
        var = self.linear_2(x)
        var_positive = nnx.softplus(var)
        return jnp.concat([mu, var_positive], axis=1)


class ResNetLayer(nnx.Module):
    def __init__(
        self, input_dim: int, output_dim: int, act_fn: nnx.gelu, *, rngs: nnx.Rngs
    ) -> None:
        self.linear_1 = nnx.Linear(input_dim, output_dim, rngs=rngs)
        self.linear_2 = nnx.Linear(output_dim, output_dim, rngs=rngs)

        self.act_fn = act_fn

    def __call__(self, x: jax.Array, rngs: Optional[nnx.Rngs] = None) -> jax.Array:
        x = self.act_fn(self.linear_1(x))
        x1 = self.act_fn(self.linear_2(x))
        return x1 + x


class HeteroscedasticMLP(nnx.Module):
    def __init__(
        self,
        dimensions: Sequence[int],
        act_fn: nnx.Module = nnx.gelu,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """Multi-layer perceptron. The first output is intended as the predicted mean
        and the second one as the predicted variance, turned positive by a softplus
        activation. The uncertainty is heteroscedastic, because the variance is also a
        function of the input vector.

        Args:
            dimensions (Sequence[int]): The dimensions of the layers. The first
                dimension is the number of input features, while the others are
                the number of neurons of the hidden layers. Therefore,
                `N_hidden_layers = len(dimensions) - 1`.
            act_fn (nnx.Module, optional): The activation function. Defaults to nnx.gelu.
            rngs (nnx.Rngs): RNGs for the flax internal modules.
        """
        self.dim_in = dimensions[0]
        self.n_hid_layers = len(dimensions) - 1
        self.act_fn = act_fn

        self.layers = nnx.List(
            [
                nnx.Linear(d_i, d_j, rngs=rngs)
                for d_i, d_j in zip(dimensions[:-1], dimensions[1:])
            ]
        )

        # Final layer is a Gaussian one (output=[mu, softplus(var)])
        self.layers.append(
            GaussianLayer(input_dim=dimensions[-1], output_dim=1, rngs=rngs)
        )

    def __call__(self, x: jax.Array, rngs: Optional[nnx.Rngs] = None) -> jax.Array:
        for layer in self.layers[:-1]:
            x = self.act_fn(layer(x))

        # Gaussian layer is applied w/o act function.
        return self.layers[-1](x)


class HeteroscedasticResNet(nnx.Module):
    def __init__(
        self,
        dim_in: int = 1,
        dim_out: int = 1,
        dim_linear_start: int = 100,
        dim_resnet_layers: int = 50,
        num_resnet_layers: int = 2,
        dim_linear_end: int = 10,
        act_fn: nnx.Module = nnx.gelu,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.act_fn = act_fn

        self.linear_start = nnx.Linear(dim_in, dim_linear_start, rngs=rngs)

        resnet_ins = [dim_linear_start] + [
            dim_resnet_layers for _ in range(num_resnet_layers - 1)
        ]
        resnet_outs = [dim_resnet_layers for _ in range(num_resnet_layers)]
        self.resnets = nnx.List(
            [
                ResNetLayer(d_i, d_j, act_fn=act_fn, rngs=rngs)
                for d_i, d_j in zip(resnet_ins, resnet_outs)
            ]
        )

        self.linear_end = nnx.Linear(dim_resnet_layers, dim_linear_end, rngs=rngs)

        self.gaussian = GaussianLayer(dim_linear_end, dim_out, rngs=rngs)

    def __call__(self, x: jax.Array, rngs: Optional[nnx.Rngs] = None) -> jax.Array:
        x = self.act_fn(self.linear_start(x))

        for resnet in self.resnets:
            # incl. already activation function
            x = resnet(x, rngs=rngs)

        x = self.act_fn(self.linear_end(x))
        return self.gaussian(x, rngs=rngs)


def nll_loss(
    model: nnx.Module, batch: jax.Array, rngs: Optional[nnx.Rngs] = None
) -> jax.Array:
    """Negative log-liklihood loss, based on a Gaussian predictive distribution of the model.
    It implements Equation (31) in https://doi.org/10.1016/j.ymssp.2023.110796.

    Args:
        model (nnx.Module): Gaussian neural network with 2 outputs (mean and variance).
        batch (jax.Array): batched input data.
        rngs (nnx.Rngs): passed to the forward method of the model.

    Returns:
        jax.Array: loss value for the batch.
    """

    xs, labels = batch
    outputs = model(xs, rngs=rngs)
    means, variances = outputs[:, 0], outputs[:, 1]
    losses = 0.5 * jnp.log(variances) + 0.5 * jnp.square(labels - means) / variances

    return jnp.mean(losses)
