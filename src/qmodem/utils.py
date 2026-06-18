from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
from flax import nnx
from jax.typing import ArrayLike

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DATA_DIR_PATH = ROOT_DIR / "data" / "raw"
CMAPSS_DIR_PATH = RAW_DATA_DIR_PATH / "CMAPSSData"
PROCESSED_DATA_DIR_PATH = ROOT_DIR / "data" / "processed"


@dataclass(frozen=True)
class Statistics:
    mean: jax.Array
    std: jax.Array
    p_025: jax.Array
    p_975: jax.Array
    count: int


def get_statistics(arr: jax.Array, dim: int) -> Statistics:
    """Compute mean, std, 2.5th and 97.5th percentiles along a specified dimension."""
    mean = jnp.mean(arr, axis=dim)
    std = jnp.std(arr, axis=dim)
    p_025 = jnp.percentile(arr, 2.5, axis=dim)
    p_975 = jnp.percentile(arr, 97.5, axis=dim)
    count = arr.shape[dim]
    return Statistics(mean=mean, std=std, p_025=p_025, p_975=p_975, count=count)


def states_equal(s1: ArrayLike, s2: ArrayLike) -> bool:
    """Checks if two JAX pytrees (eg model parameters) are equal.

    Args:
        s1 (ArrayLike): First pytree to compare.
        s2 (ArrayLike): Second pytree to compare.

    Returns:
        bool: True if the pytrees are equal, False otherwise.
    """
    leaves_equal = jax.tree.leaves(jax.tree.map(jnp.array_equal, s1, s2))
    return all(leaves_equal)


def count_parameters(model: nnx.Module) -> int:
    """Count the total number of parameters in a flax.nnx.Module."""
    return sum(p.size for p in jax.tree.leaves(nnx.state(model, nnx.Param)))
