"""The battery networks: the MCD posterior network against the mean network."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from hypothesis import given, settings
from hypothesis import strategies as st

from qmodem.battery.methods.mcd import build_mcd_net
from qmodem.battery.train import MeanNetworkHyperparameters, build_mean_net

WINDOW = 10

networks = st.builds(
    MeanNetworkHyperparameters,
    kernel_size=st.integers(1, 5),
    n_filters=st.integers(1, 8),
    activation=st.sampled_from(["gelu", "relu", "silu", "tanh"]),
)
rates = st.floats(0.01, 0.5)


def _tree_structure(module: nnx.Module) -> tuple:
    state = nnx.state(module, nnx.Param)
    return jax.tree.structure(state), [leaf.shape for leaf in jax.tree.leaves(state)]


def _keyed(module: nnx.Module, key: jax.Array) -> nnx.Module:
    """The network in draw mode, its dropout stream reseeded from ``key``."""
    module = nnx.clone(module)
    module.set_attributes(nnx.Dropout, deterministic=False, raise_if_not_found=False)
    nnx.reseed(module, default=key)
    return module


@settings(max_examples=20, deadline=None)
@given(network=networks, rate=rates, seed=st.integers(0, 2**16))
def test_the_mcd_net_has_the_mean_nets_parameter_tree(
    network: MeanNetworkHyperparameters, rate: float, seed: int
) -> None:
    key = jax.random.key(seed)

    assert _tree_structure(build_mcd_net(network, rate, key)) == _tree_structure(
        build_mean_net(network, key)
    )


@settings(max_examples=10, deadline=None)
@given(network=networks, rate=rates, seed=st.integers(0, 2**16))
def test_the_mcd_net_in_inference_mode_is_the_mean_net(
    network: MeanNetworkHyperparameters, rate: float, seed: int
) -> None:
    key = jax.random.key(seed)
    mean_net = build_mean_net(network, key)
    mcd_net = build_mcd_net(network, rate, jax.random.key(seed + 1))
    nnx.update(mcd_net, nnx.state(mean_net, nnx.Param))
    mcd_net.set_attributes(nnx.Dropout, deterministic=True)
    inputs = jax.random.normal(jax.random.key(seed + 2), (4, WINDOW, 2))

    np.testing.assert_array_equal(mcd_net(inputs), mean_net(inputs))


@settings(max_examples=10, deadline=None)
@given(
    rate=st.floats(0.2, 0.5),
    seed=st.integers(0, 2**16),
    split=st.integers(1, 7),
)
def test_a_draw_is_one_function_across_the_batch_and_input_subsets(
    rate: float, seed: int, split: int
) -> None:
    network = MeanNetworkHyperparameters(kernel_size=3, n_filters=4)
    net = _keyed(build_mcd_net(network, rate, jax.random.key(seed)), jax.random.key(1))
    inputs = jax.random.normal(jax.random.key(seed + 1), (8, WINDOW, 2))
    whole = net(inputs)
    # The same input repeated over the batch gets the same prediction in a draw.
    repeated = _keyed(net, jax.random.key(1))(jnp.repeat(inputs[:1], 8, axis=0))
    head = _keyed(net, jax.random.key(1))(inputs[:split])
    tail = _keyed(net, jax.random.key(1))(inputs[split:])

    np.testing.assert_allclose(repeated, jnp.repeat(repeated[:1], 8, axis=0))
    np.testing.assert_allclose(jnp.concatenate([head, tail]), whole, rtol=1e-6)


def test_different_draws_give_different_functions() -> None:
    network = MeanNetworkHyperparameters(kernel_size=3, n_filters=4)
    net = build_mcd_net(network, 0.3, jax.random.key(0))
    inputs = jax.random.normal(jax.random.key(1), (5, WINDOW, 2))

    draws = [_keyed(net, jax.random.key(k))(inputs) for k in range(4)]

    for i in range(len(draws)):
        for j in range(i):
            assert not np.allclose(draws[i], draws[j])
