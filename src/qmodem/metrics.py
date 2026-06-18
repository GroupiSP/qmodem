from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, SupportsIndex

import jax
import jax.numpy as jnp
from flax import nnx

from .module import RandomCallModel, mc_sample


def _cdf(x, samples):
    if len(samples) == 0:
        return 0.0

    sorted_samples = jnp.sort(samples)
    count = jnp.sum(jnp.where(sorted_samples <= x, 1, 0))
    cdf_value = count / len(sorted_samples)

    return cdf_value


def _point_crps(y_true, samples_predicted, x_grid):
    F0 = jnp.where(x_grid < y_true, 0.0, 1.0)
    F1 = jax.vmap(_cdf, in_axes=(0, None), out_axes=0)(x_grid, samples_predicted)

    crps_value = jnp.trapezoid(jnp.square(F0 - F1), x_grid)
    return crps_value


class LabelledDataSource(Protocol):
    X: jax.Array  # axis 0 is the batch dimension
    y: jax.Array

    def __len__(self) -> int: ...
    def __getitem__(self, idx: SupportsIndex) -> tuple[jax.Array, jax.Array]: ...


@dataclass(frozen=True)
class MetricsContext:
    num_mc_samples: int = 10
    eval_grid_resolution: int = 100


def compute_rmse(
    test_datasource: LabelledDataSource,
    model: RandomCallModel,
    context: MetricsContext = MetricsContext(),
) -> float:
    """Computes the root mean square error between the predictions on the model on a
    test dataset and the labels.

    Args:
        test_datasource (LabelledDataSource): test dataset
        model (nnx.Module): data model, assumed to be deterministic (e.g. dropout is disabled)
        context (MetricsContext): metrics configuration. Not used for RMSE
            but included for consistency with other metrics.

    Returns:
        float: RMSE value.
    """
    labels = test_datasource.y
    predictions = model(test_datasource.X, rngs=nnx.Rngs(0))

    mse_losses = jnp.mean((predictions - labels) ** 2, axis=0)
    return jnp.sqrt(mse_losses).item()


def compute_point_crps(
    test_datasource: LabelledDataSource,
    model: RandomCallModel,
    context: MetricsContext = MetricsContext(),
) -> float:
    """Computes the mean CRPS between the model samples and the labels on a test
    dataset.

    Args:
        test_datasource (LabelledDataSource): test dataset
        model (nnx.Module): data model, assumed to be stochastic (e.g. Monte Carlo Dropout)
        context (MetricsContext, optional): metrics configuration, including number of
            Monte Carlo samples and grid resolution for CRPS evaluation.

    Returns:
        float: mean CRPS value.
    """

    keys = jax.random.split(jax.random.key(0), context.num_mc_samples)
    y_pred_samples = mc_sample(
        model, test_datasource.X, keys
    )  # pass entire dataset at once

    total_crps = 0.0

    # y_pred_samples has shape (num_samples, batch_size), but we need to iterate over the batch dimension
    # TODO: vectorise the loop with jax.vmap or jax.lax.fori_loop
    for i in range(test_datasource.X.shape[0]):
        y_true = test_datasource.y[i]
        samples_predicted = y_pred_samples[:, i]

        x_grid = jnp.linspace(
            jnp.min(samples_predicted),
            jnp.max(samples_predicted),
            num=context.eval_grid_resolution,
        )

        total_crps += _point_crps(y_true, samples_predicted, x_grid)

    return (total_crps / len(test_datasource)).item()
