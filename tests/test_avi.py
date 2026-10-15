"""The dataset-agnostic AVI inference method and the quantum latent map.

The AVI method is exercised on a tiny linear posterior problem with test-double
generators and discriminators, so these tests depend on no dataset.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable

import jax
import jax.numpy as jnp
import jax.scipy.stats as jstats
import numpy as np
import optax
import pytest
from flax import nnx
from jax.flatten_util import ravel_pytree
from hypothesis import given, settings
from hypothesis import strategies as st
from stagereg.core import Batch
from stagereg.jax import IsotropicGaussianPrior, PosteriorProblem

from qmodem.avi import (
    AVI,
    ClassicalLatentMap,
    LatentMap,
    QuantumLatentMap,
    matched_hidden_width,
)

N_FEATURES = 3
DATASET_SIZE = 64
BATCH_SIZE = 8
LATENT_WIDTH = 3


def _mean_function(parameters, inputs, key):
    del key
    return inputs @ parameters["w"] + parameters["b"]


def _problem() -> PosteriorProblem:
    initial = {
        "w": jnp.asarray([[0.5], [-0.25], [1.0]]),
        "b": jnp.asarray([0.1]),
    }

    def log_likelihood(parameters, batch: Batch, key):
        means = _mean_function(parameters, batch.inputs, key)
        scale = DATASET_SIZE / batch.targets.shape[0]
        return scale * jnp.sum(jstats.norm.logpdf(batch.targets, means, 0.3))

    return PosteriorProblem(
        initial_parameters=initial,
        mean_function=_mean_function,
        log_likelihood=log_likelihood,
        prior=IsotropicGaussianPrior(precision=1.0),
        train_loader=lambda *, split, epoch: [],
        dataset_size=DATASET_SIZE,
        aleatoric_variance=lambda inputs: jnp.full((inputs.shape[0], 1), 0.09),
    )


class _ShiftGenerator(nnx.Module):
    """Test double: the warm start shifted by a linear map of the latent vector."""

    def __init__(self, latent_map: LatentMap, warm_start, *, rngs: nnx.Rngs) -> None:
        flat, self.unravel = ravel_pytree(warm_start)
        self.latent_map = latent_map
        self.offset = nnx.Variable(flat)
        self.scale = nnx.Param(
            0.1 * jax.random.normal(rngs.params(), (latent_map.width,))
        )

    def __call__(self, z):
        shift = self.latent_map(z) @ self.scale[...]
        return self.unravel(self.offset[...] + shift)


class _PairDiscriminator(nnx.Module):
    """Test double: a linear logit of the (input, target) pair."""

    def __init__(self, *, rngs: nnx.Rngs) -> None:
        self.linear = nnx.Linear(N_FEATURES + 1, 1, rngs=rngs)

    def __call__(self, inputs, targets):
        return self.linear(jnp.concatenate([inputs, targets], axis=-1))


type LatentMapFactory = Callable[[nnx.Rngs], LatentMap]

LATENT_MAPS: dict[str, LatentMapFactory] = {
    "quantum": lambda rngs: QuantumLatentMap(LATENT_WIDTH, 1, rngs=rngs),
    "classical": lambda rngs: ClassicalLatentMap(
        LATENT_WIDTH, matched_hidden_width(LATENT_WIDTH, 1), rngs=rngs
    ),
}
"""The AVI properties must hold whichever latent map the generator holds."""


def _avi(
    latent_map: LatentMapFactory,
    *,
    num_draws: int = 2,
    discriminator_steps: int = 2,
    seed: int = 0,
) -> AVI:
    return AVI(
        build_generator=lambda warm_start, key: _ShiftGenerator(
            latent_map(nnx.Rngs(key)),
            warm_start,
            rngs=nnx.Rngs(jax.random.fold_in(key, 1)),
        ),
        build_discriminator=lambda key: _PairDiscriminator(rngs=nnx.Rngs(key)),
        key=jax.random.key(seed),
        generator_optimizer=optax.adam(1e-2),
        discriminator_optimizer=optax.adam(1e-2),
        adversarial_weight=0.1,
        num_draws=num_draws,
        discriminator_steps=discriminator_steps,
    )


def _batch(seed: int = 0) -> Batch:
    key_x, key_y = jax.random.split(jax.random.key(seed))
    inputs = jax.random.normal(key_x, (BATCH_SIZE, N_FEATURES))
    targets = inputs @ jnp.asarray([[0.4], [-0.2], [0.9]]) + 0.3 * jax.random.normal(
        key_y, (BATCH_SIZE, 1)
    )
    return Batch(inputs=inputs, targets=targets)


def _inputs(size: int = 10) -> jax.Array:
    return jax.random.normal(jax.random.key(7), (size, N_FEATURES))


@pytest.fixture(scope="module", params=sorted(LATENT_MAPS))
def build_avi(request: pytest.FixtureRequest) -> Callable[..., AVI]:
    """Builds the test AVI with each latent map in turn."""
    return functools.partial(_avi, LATENT_MAPS[request.param])


@pytest.fixture(scope="module")
def trained_avi(build_avi: Callable[..., AVI]) -> AVI:
    method = build_avi()
    method.init(_problem())
    for step in range(3):
        method.update(_batch(step), jax.random.key(100 + step))
    return method


# --- quantum latent map ----------------------------------------------------------


@settings(max_examples=10, deadline=None)
@given(
    width=st.integers(2, 4),
    layers=st.integers(1, 3),
    z=st.floats(0.0, 2 * math.pi),
    seed=st.integers(0, 2**16),
)
def test_the_quantum_latent_map_is_a_bounded_vector_of_its_width_with_2lq_parameters(
    width: int, layers: int, z: float, seed: int
) -> None:
    latent_map = QuantumLatentMap(width, layers, rngs=nnx.Rngs(seed))

    latent = latent_map(jnp.asarray(z))

    assert isinstance(latent_map, LatentMap)
    assert latent_map.width == width
    assert latent.shape == (width,)
    assert np.all(np.abs(np.asarray(latent)) <= 1.0 + 1e-6)
    assert latent_map.parameter_count == 2 * layers * width
    params = nnx.state(latent_map, nnx.Param)
    assert sum(leaf.size for leaf in jax.tree.leaves(params)) == 2 * layers * width


def test_a_quantum_latent_map_needs_at_least_two_qubits() -> None:
    with pytest.raises(ValueError):
        QuantumLatentMap(1, 1, rngs=nnx.Rngs(0))


def test_the_quantum_latent_map_angles_start_uniform_in_0_2pi() -> None:
    latent_map = QuantumLatentMap(6, 4, rngs=nnx.Rngs(0))
    angles = np.asarray(jax.tree.leaves(nnx.state(latent_map, nnx.Param))[0])

    assert angles.min() >= 0.0
    assert angles.max() <= 2 * math.pi


# --- classical latent map --------------------------------------------------------


def _classical_count(width: int, hidden: int) -> int:
    """Parameters of a ``1 -> hidden -> width`` MLP with biases."""
    return (1 * hidden + hidden) + (hidden * width + width)


@pytest.mark.parametrize(
    ("width", "layers", "hidden"),
    [(5, 1, 1), (3, 1, 1), (6, 4, 5), (4, 3, 3), (3, 4, 4), (2, 1, 1)],
)
def test_the_matched_hidden_width_follows_the_h_formula(
    width: int, layers: int, hidden: int
) -> None:
    assert matched_hidden_width(width, layers) == hidden


@settings(max_examples=50, deadline=None)
@given(width=st.integers(2, 8), layers=st.integers(1, 8))
def test_the_classical_latent_map_is_the_nearest_achievable_match_to_the_circuit(
    width: int, layers: int
) -> None:
    quantum = QuantumLatentMap(width, layers, rngs=nnx.Rngs(0)).parameter_count
    hidden = matched_hidden_width(width, layers)
    classical = ClassicalLatentMap(width, hidden, rngs=nnx.Rngs(0))
    params = nnx.state(classical, nnx.Param)

    assert classical.parameter_count == _classical_count(width, hidden)
    assert sum(leaf.size for leaf in jax.tree.leaves(params)) == _classical_count(
        width, hidden
    )
    best = min(abs(_classical_count(width, h) - quantum) for h in range(1, 4 * quantum))
    assert abs(classical.parameter_count - quantum) == best


@settings(max_examples=10, deadline=None)
@given(
    width=st.integers(1, 6),
    hidden=st.integers(1, 6),
    z=st.floats(0.0, 2 * math.pi),
    seed=st.integers(0, 2**16),
)
def test_the_classical_latent_map_is_a_bounded_vector_of_its_width(
    width: int, hidden: int, z: float, seed: int
) -> None:
    latent_map = ClassicalLatentMap(width, hidden, rngs=nnx.Rngs(seed))

    latent = latent_map(jnp.asarray(z))

    assert isinstance(latent_map, LatentMap)
    assert latent_map.width == width
    assert latent.shape == (width,)
    assert np.all(np.abs(np.asarray(latent)) <= 1.0)


@settings(max_examples=10, deadline=None)
@given(seed=st.integers(0, 2**16))
def test_the_same_key_gives_the_same_classical_initialization_and_others_differ(
    seed: int,
) -> None:
    def parameters(s: int) -> list[np.ndarray]:
        latent_map = ClassicalLatentMap(4, 3, rngs=nnx.Rngs(jax.random.key(s)))
        return [np.asarray(x) for x in jax.tree.leaves(nnx.state(latent_map))]

    first, again, other = parameters(seed), parameters(seed), parameters(seed + 1)

    for a, b in zip(first, again, strict=True):
        np.testing.assert_array_equal(a, b)
    assert any(not np.array_equal(a, b) for a, b in zip(first, other, strict=True))


def test_the_classical_latent_map_uses_flax_default_initialization() -> None:
    latent_map = ClassicalLatentMap(4, 3, rngs=nnx.Rngs(0))
    reference = nnx.Linear(1, 3, rngs=nnx.Rngs(0))

    np.testing.assert_array_equal(latent_map.hidden.kernel[...], reference.kernel[...])
    np.testing.assert_array_equal(latent_map.hidden.bias[...], np.zeros(3))
    np.testing.assert_array_equal(latent_map.output.bias[...], np.zeros(4))


# --- AVI inference method --------------------------------------------------------


def test_avi_early_stops_and_declares_its_diagnostics(
    build_avi: Callable[..., AVI],
) -> None:
    method = build_avi()

    assert method.supports_early_stopping
    assert set(method.diagnostics) == {
        "discriminator_loss",
        "discriminator_accuracy",
        "weight_spread",
    }
    assert not method.has_posterior


def test_an_avi_update_returns_a_finite_loss_and_every_declared_diagnostic(
    build_avi: Callable[..., AVI],
) -> None:
    method = build_avi()
    method.init(_problem())

    outputs = method.update(_batch(), jax.random.key(1))

    assert set(outputs) == {"loss", *method.diagnostics}
    for value in outputs.values():
        assert np.isfinite(float(value))
    assert 0.0 <= float(outputs["discriminator_accuracy"]) <= 1.0
    assert float(outputs["weight_spread"]) >= 0.0


def test_the_discriminator_is_in_the_state_but_not_in_the_posterior_state(
    trained_avi: AVI,
) -> None:
    assert set(trained_avi.state()) == {
        "generator",
        "discriminator",
        "generator_optimizer",
        "discriminator_optimizer",
    }
    assert set(trained_avi.posterior_state()) == {"generator"}


def test_an_update_trains_both_the_generator_and_the_discriminator(
    build_avi: Callable[..., AVI],
) -> None:
    method = build_avi()
    method.init(_problem())
    before = method.state()

    method.update(_batch(), jax.random.key(1))

    after = method.state()
    for part in ("generator", "discriminator"):
        assert any(
            not np.array_equal(a, b)
            for a, b in zip(
                jax.tree.leaves(before[part]), jax.tree.leaves(after[part]), strict=True
            )
        )


@settings(max_examples=10, deadline=None)
@given(seed=st.integers(0, 2**16))
def test_the_same_key_gives_the_same_draws_and_other_keys_other_draws(
    trained_avi: AVI, seed: int
) -> None:
    posterior = trained_avi.posterior()
    inputs = _inputs()

    first = posterior.sample_means(inputs, jax.random.key(seed), 5)
    again = posterior.sample_means(inputs, jax.random.key(seed), 5)
    other = posterior.sample_means(inputs, jax.random.key(seed + 1), 5)

    assert first.shape == (5, 10, 1)
    np.testing.assert_array_equal(first, again)
    assert not np.allclose(first, other)


@settings(max_examples=10, deadline=None)
@given(start=st.integers(0, 8), size=st.integers(1, 2), seed=st.integers(0, 2**16))
def test_avi_draws_are_consistent_across_input_subsets(
    trained_avi: AVI, start: int, size: int, seed: int
) -> None:
    posterior = trained_avi.posterior()
    inputs = _inputs()
    key = jax.random.key(seed)

    whole = posterior.sample_means(inputs, key, 4)
    subset = posterior.sample_means(inputs[start : start + size], key, 4)

    np.testing.assert_allclose(subset, whole[:, start : start + size], rtol=1e-6)


def test_the_posterior_is_independent_of_later_updates(
    build_avi: Callable[..., AVI],
) -> None:
    method = build_avi()
    method.init(_problem())
    posterior = method.posterior()
    key = jax.random.key(3)
    before = posterior.sample_means(_inputs(), key, 3)

    method.update(_batch(), jax.random.key(1))

    np.testing.assert_array_equal(posterior.sample_means(_inputs(), key, 3), before)


def test_a_restored_posterior_gives_identical_draws(
    trained_avi: AVI, build_avi: Callable[..., AVI]
) -> None:
    restored = build_avi(seed=99)
    restored.init(_problem())
    restored.load_posterior(trained_avi.posterior_state())
    key = jax.random.key(4)

    np.testing.assert_array_equal(
        restored.posterior().sample_means(_inputs(), key, 6),
        trained_avi.posterior().sample_means(_inputs(), key, 6),
    )


def test_a_resumed_state_continues_exactly_like_the_original(
    trained_avi: AVI, build_avi: Callable[..., AVI]
) -> None:
    resumed = build_avi(seed=99)
    resumed.init(_problem())
    resumed.load_state(trained_avi.state())
    original = build_avi()
    original.init(_problem())
    original.load_state(trained_avi.state())

    expected = original.update(_batch(9), jax.random.key(9))
    actual = resumed.update(_batch(9), jax.random.key(9))

    for name, value in expected.items():
        np.testing.assert_array_equal(actual[name], value)


def test_a_posterior_state_of_another_tree_is_refused(
    trained_avi: AVI, build_avi: Callable[..., AVI]
) -> None:
    other = build_avi()
    other.init(_problem())
    state = trained_avi.posterior_state()
    broken = {"generator": {"unexpected": jnp.zeros(3)}}

    with pytest.raises(ValueError):
        other.load_posterior(broken)
    other.load_posterior(state)


def test_an_uninitialized_avi_refuses_to_update(build_avi: Callable[..., AVI]) -> None:
    with pytest.raises(ValueError):
        build_avi().update(_batch(), jax.random.key(0))
