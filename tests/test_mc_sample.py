from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from qmodem.module import mc_sample

_NUM_SAMPLES = 5
_BATCH_SIZE = 4
_WINDOW_SIZE = 8
_IN_FEATURES = 3


@pytest.fixture
def x_mock() -> jax.Array:
    # Shape (batch, window_size, in_features)
    return jax.random.normal(
        jax.random.key(0), (_BATCH_SIZE, _WINDOW_SIZE, _IN_FEATURES)
    )


@pytest.fixture
def rng_keys_mock():
    """Return a factory that splits a key into (key_weights, key_noise) pairs."""

    def make_keys(
        seed: int = 0, num_keys: int = _NUM_SAMPLES
    ) -> tuple[jax.Array, jax.Array]:
        keys = jax.random.split(jax.random.key(seed), num=2 * num_keys)
        return keys[:num_keys], keys[num_keys:]

    return make_keys


@pytest.fixture
def model_call_mock():
    """Mock model that returns a Gaussian-head output of shape (batch, 2)."""

    def call(x: jax.Array, rngs: nnx.Rngs) -> jax.Array:
        batch = x.shape[0]
        mu = jnp.mean(x, axis=(-2, -1), keepdims=False).reshape(batch, 1)
        var = jnp.full((batch, 1), 0.01)
        return jnp.concatenate([mu, var], axis=-1)

    return call


def test_mc_sample_output_shape(model_call_mock, x_mock, rng_keys_mock):
    key_weights, key_noise = rng_keys_mock()
    samples = mc_sample(model_call_mock, x_mock, key_weights, key_noise)
    assert samples.shape == (_NUM_SAMPLES, 1), (
        f"Expected shape ({_NUM_SAMPLES}, 1), got {samples.shape}"
    )


def test_mc_sample_output_type(model_call_mock, x_mock, rng_keys_mock):
    key_weights, key_noise = rng_keys_mock()
    samples = mc_sample(model_call_mock, x_mock, key_weights, key_noise)
    assert samples.dtype == jnp.float32, f"Expected dtype float32, got {samples.dtype}"


def test_mc_sample_deterministic_for_same_keys(model_call_mock, x_mock, rng_keys_mock):
    key_weights1, key_noise1 = rng_keys_mock()
    key_weights2, key_noise2 = rng_keys_mock()
    samples1 = mc_sample(model_call_mock, x_mock, key_weights1, key_noise1)
    samples2 = mc_sample(model_call_mock, x_mock, key_weights2, key_noise2)
    assert jnp.allclose(samples1, samples2), (
        "Expected samples to be the same for the same keys"
    )


def test_mc_sample_different_for_different_keys(model_call_mock, x_mock, rng_keys_mock):
    key_weights1, key_noise1 = rng_keys_mock(seed=0)
    key_weights2, key_noise2 = rng_keys_mock(seed=1)
    samples1 = mc_sample(model_call_mock, x_mock, key_weights1, key_noise1)
    samples2 = mc_sample(model_call_mock, x_mock, key_weights2, key_noise2)
    assert samples1.shape == samples2.shape, (
        f"Expected samples to have the same shape, got {samples1.shape} and {samples2.shape}"
    )
    assert not jnp.allclose(samples1, samples2), (
        "Expected samples to be different for different keys"
    )
