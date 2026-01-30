import jax
import jax.numpy as jnp


def cdf(x, samples):
    if len(samples) == 0:
        return 0.0

    sorted_samples = jnp.sort(samples)
    count = jnp.sum(jnp.where(sorted_samples <= x, 1, 0))
    cdf_value = count / len(sorted_samples)

    return cdf_value


def crps(samples_dist_0, samples_dist_1, x_grid):
    F0 = jax.vmap(cdf, in_axes=(0, None), out_axes=0)(x_grid, samples_dist_0)
    F1 = jax.vmap(cdf, in_axes=(0, None), out_axes=0)(x_grid, samples_dist_1)

    crps_value = jnp.trapezoid(jnp.square(F0 - F1), x_grid)
    return crps_value
