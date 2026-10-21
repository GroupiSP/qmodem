"""The shared posterior stage: its keys and plug-in pieces."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from qmodem.battery.posterior import (
    PosteriorKeys,
    parameter_count,
    posterior_loader,
)


def test_the_posterior_stage_validates_on_the_posterior_validation_set(
    synthetic_battery,
) -> None:
    data = synthetic_battery.data
    loader = posterior_loader(data, batch_size=8, key=jax.random.key(0))

    (batch,) = loader(split="validation", epoch=0)

    np.testing.assert_array_equal(
        np.asarray(batch.inputs), np.asarray(data.posterior_val.features)
    )
    np.testing.assert_array_equal(
        np.asarray(batch.targets), np.asarray(data.posterior_val.targets)
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
