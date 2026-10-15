"""The battery AVI components: the conv-layer generator and the discriminator."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from hypothesis import given, settings
from hypothesis import strategies as st
from stagereg.jax.posterior import require_inferred_tree

from qmodem.avi import QuantumLatentMap
from qmodem.battery.data_processing import WINDOW_SIZE
from qmodem.battery.models import (
    IN_FEATURES,
    ConvGenerator,
    MeanNet,
    WindowDiscriminator,
)
from qmodem.battery.posterior import INFERRED_SUBNETWORK

N_FILTERS = 4


def _warm_start() -> nnx.State:
    mean_net = MeanNet(3, N_FILTERS, nnx.gelu, rngs=nnx.Rngs(0))
    return nnx.state(mean_net, nnx.All(nnx.Param, INFERRED_SUBNETWORK))


def _generator(seed: int, scale: float = 1e-2) -> ConvGenerator:
    rngs = nnx.Rngs(seed)
    return ConvGenerator(
        QuantumLatentMap(3, 1, rngs=rngs), _warm_start(), scale=scale, rngs=rngs
    )


@settings(max_examples=10, deadline=None)
@given(z=st.floats(0.0, 2 * math.pi), seed=st.integers(0, 2**16))
def test_every_initial_draw_is_close_to_the_warm_start_conv_layer(
    z: float, seed: int
) -> None:
    warm_start = _warm_start()

    draw = _generator(seed)(jnp.asarray(z))

    require_inferred_tree(draw, warm_start)
    for name in ("kernel", "bias"):
        np.testing.assert_allclose(
            draw["conv"][name][...], warm_start["conv"][name][...], atol=0.1
        )


@settings(max_examples=5, deadline=None)
@given(seed=st.integers(0, 2**16))
def test_a_generator_without_scale_outputs_exactly_the_warm_start(seed: int) -> None:
    warm_start = _warm_start()

    draw = _generator(seed, scale=0.0)(jnp.asarray(1.3))

    for name in ("kernel", "bias"):
        np.testing.assert_array_equal(
            draw["conv"][name][...], warm_start["conv"][name][...]
        )


def test_the_generator_trains_the_latent_map_and_the_postprocessor_weights() -> None:
    generator = _generator(0)
    latent = 3
    conv = 3 * IN_FEATURES * N_FILTERS + N_FILTERS

    parameters = nnx.state(generator, nnx.Param)

    assert sum(leaf.size for leaf in jax.tree.leaves(parameters)) == (
        2 * 1 * latent + latent * conv + conv
    )


def test_the_discriminator_scores_each_window_target_pair_with_one_logit() -> None:
    discriminator = WindowDiscriminator(rngs=nnx.Rngs(0))
    inputs = jnp.ones((7, WINDOW_SIZE, IN_FEATURES))
    targets = jnp.zeros((7, 1))

    logits = discriminator(inputs, targets)

    assert logits.shape == (7, 1)
    assert discriminator.hidden == (64, 64)
