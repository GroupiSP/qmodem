from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from qmodem.module import mc_sample


@pytest.fixture
def x_mock() -> jax.Array:
    return jnp.array([[1.0], [2.0], [3.0]])


@pytest.fixture
def rng_keys_mock() -> jax.Array:
    def make_keys(seed: int = 0, num_keys: int = 5):
        return jax.random.split(jax.random.key(seed), num_keys)

    return make_keys


@pytest.fixture
def model_call_mock() -> jax.Array:
    # Mock implementation of the model's __call__ method
    def call(x: jax.Array, rngs: nnx.Rngs) -> jax.Array:
        return x + rngs.normal(shape=x.shape) * 0.1

    return call


def test_mc_sample_output_shape(model_call_mock, x_mock, rng_keys_mock):
    samples = mc_sample(model_call_mock, x_mock, rng_keys_mock())
    assert samples.shape == (5, 3, 1), f"Expected shape (5, 3, 1), got {samples.shape}"


def test_mc_sample_output_type(model_call_mock, x_mock, rng_keys_mock):
    samples = mc_sample(model_call_mock, x_mock, rng_keys_mock())
    assert samples.dtype == jnp.float32, f"Expected dtype float32, got {samples.dtype}"


def test_mc_sample_determinstic_for_same_keys(model_call_mock, x_mock, rng_keys_mock):
    samples1 = mc_sample(model_call_mock, x_mock, rng_keys_mock())
    samples2 = mc_sample(model_call_mock, x_mock, rng_keys_mock())
    assert jnp.allclose(samples1, samples2), (
        "Expected samples to be the same for the same keys"
    )


def test_mc_sample_different_for_different_keys(model_call_mock, x_mock, rng_keys_mock):
    samples1 = mc_sample(model_call_mock, x_mock, rng_keys_mock(seed=0))
    samples2 = mc_sample(model_call_mock, x_mock, rng_keys_mock(seed=1))
    assert samples1.shape == samples2.shape, (
        f"Expected samples to have the same shape, got {samples1.shape} and {samples2.shape}"
    )
    assert not jnp.allclose(samples1, samples2), (
        "Expected samples to be different for different keys"
    )
