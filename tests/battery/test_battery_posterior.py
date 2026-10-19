"""The shared posterior stage: fixed validation keys and the stage's plug-in pieces."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from stagereg.core import Batch
from stagereg.jax import PosteriorProblem

from qmodem.battery.posterior import (
    FixedDrawKeyMethod,
    FixedKeyPredictor,
    PosteriorKeys,
    parameter_count,
)


@dataclasses.dataclass(frozen=True)
class _KeyedPosterior:
    """Draw ``s`` is a constant that depends on the key: ``uniform(fold_in(key,
    s))``."""

    @property
    def num_draws(self) -> int | None:
        return None

    def sample_means(
        self, inputs: jax.Array, key: jax.Array, num_samples: int | None = None
    ) -> jax.Array:
        offsets = jax.vmap(lambda s: jax.random.uniform(jax.random.fold_in(key, s)))(
            jnp.arange(num_samples)
        )
        return offsets[:, None, None] + jnp.zeros((1, inputs.shape[0], 1))


class _StubMethod:
    supports_early_stopping = True
    diagnostics = ("spread",)
    has_posterior = True

    def __init__(self) -> None:
        self.updates: list[Batch] = []

    def init(self, problem: PosteriorProblem) -> None:
        del problem

    def update(self, batch: Batch, key: jax.Array) -> Mapping[str, Any]:
        self.updates.append(batch)
        return {"loss": 1.0, "spread": 2.0}

    def posterior(self) -> _KeyedPosterior:
        return _KeyedPosterior()

    def state(self) -> Any:
        return {"full": 1}

    def load_state(self, state: Any) -> None:
        del state

    def posterior_state(self) -> Any:
        return {"posterior": 1}

    def load_posterior(self, state: Any) -> None:
        del state


INPUTS = jnp.zeros((3, 10, 2))


@settings(max_examples=20, deadline=None)
@given(asked=st.integers(0, 2**31 - 1), fixed=st.integers(0, 2**31 - 1))
def test_a_fixed_draw_key_method_draws_the_same_whatever_key_is_asked(
    asked: int, fixed: int
) -> None:
    method = _StubMethod()
    wrapped = FixedDrawKeyMethod(method, jax.random.key(fixed))

    np.testing.assert_array_equal(
        wrapped.posterior().sample_means(INPUTS, jax.random.key(asked), 5),
        method.posterior().sample_means(INPUTS, jax.random.key(fixed), 5),
    )


def test_a_fixed_draw_key_method_trains_and_checkpoints_like_the_wrapped_one() -> None:
    method = _StubMethod()
    wrapped = FixedDrawKeyMethod(method, jax.random.key(0))
    batch = Batch(inputs=INPUTS)

    assert wrapped.update(batch, jax.random.key(1)) == {"loss": 1.0, "spread": 2.0}
    assert method.updates == [batch]
    assert wrapped.state() == method.state()
    assert wrapped.posterior_state() == method.posterior_state()
    assert wrapped.diagnostics == method.diagnostics
    assert wrapped.supports_early_stopping and wrapped.has_posterior


@given(asked=st.integers(0, 2**31 - 1))
def test_a_fixed_key_predictor_passes_its_own_key(asked: int) -> None:
    received = []
    predictor = FixedKeyPredictor(
        lambda model, batch, requirements, key: received.append(key),
        jax.random.key(7),
    )

    predictor(None, Batch(inputs=INPUTS), None, jax.random.key(asked))

    assert jax.random.key_data(received[0]).tolist() == (
        jax.random.key_data(jax.random.key(7)).tolist()
    )


def _bits(key: jax.Array) -> tuple[int, ...]:
    return tuple(np.asarray(jax.random.key_data(key)).ravel().tolist())


@given(seed=st.integers(0, 2**31 - 1))
def test_posterior_keys_are_reproducible_and_distinct(seed: int) -> None:
    root = jax.random.key(seed)
    keys = PosteriorKeys.of(root)
    fields = [_bits(getattr(keys, f.name)) for f in dataclasses.fields(keys)]

    assert fields == [
        _bits(getattr(PosteriorKeys.of(root), f.name)) for f in dataclasses.fields(keys)
    ]
    assert len({*fields, _bits(root)}) == len(fields) + 1


@given(
    shapes=st.lists(
        st.lists(st.integers(1, 4), min_size=0, max_size=3), min_size=0, max_size=4
    )
)
def test_the_parameter_count_is_the_number_of_entries(shapes: list[list[int]]) -> None:
    tree = {str(i): jnp.zeros(shape) for i, shape in enumerate(shapes)}

    assert parameter_count(tree) == sum(int(np.prod(shape)) for shape in shapes)
