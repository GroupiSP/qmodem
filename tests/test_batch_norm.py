import jax
import jax.numpy as jnp
from flax import nnx


def test_batch_norm_stats():
    """Example test with flax.

    Taken from
    https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/normalization.html
    """
    x = jax.random.normal(jax.random.key(0), (5, 6))

    layer = nnx.BatchNorm(
        num_features=6, momentum=0.9, epsilon=1e-5, dtype=jnp.float32, rngs=nnx.Rngs(0)
    )

    # calculate batch norm on input and update batch statistics
    layer.train()

    layer(x)
    batch_stats1 = nnx.clone(nnx.state(layer, nnx.BatchStat))  # keep a copy
    layer(x)
    batch_stats2 = nnx.state(layer, nnx.BatchStat)

    assert (batch_stats1["mean"][...] != batch_stats2["mean"][...]).all()
    assert (batch_stats1["var"][...] != batch_stats2["var"][...]).all()

    # use stored batch statistics' running average
    layer.eval()
    layer(x)

    batch_stats3 = nnx.state(layer, nnx.BatchStat)
    assert (batch_stats2["mean"][...] == batch_stats3["mean"][...]).all()
    assert (batch_stats2["var"][...] == batch_stats3["var"][...]).all()
