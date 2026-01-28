import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from qmodem.module import HNNV0, MCDNetV0, nll_loss

_rng_key = jax.random.PRNGKey(0)


@pytest.fixture
def mock_hnn_v0() -> HNNV0:
    return HNNV0(dimensions=[1, 10, 10], rngs=nnx.Rngs(0))


@pytest.fixture
def mock_mcdnet_v0() -> MCDNetV0:
    return MCDNetV0(rngs=nnx.Rngs(params=0, dropout=1))


def test_hnn_v0_init(mock_hnn_v0) -> None:
    """Tests the __init__ of the GaussianHeteroscedasticMLP.

    It checks:
    - if the number of linear layers is the expected one.
    """
    num_linear_layers = sum(
        isinstance(layer, nnx.Linear) for _, layer in mock_hnn_v0.iter_modules()
    )
    # The Gaussian layer contains 2 linear layers.
    assert num_linear_layers == mock_hnn_v0.n_hid_layers + 2


@pytest.mark.parametrize(
    "x",
    [
        jax.random.normal(shape=[10, 1], key=_rng_key),
        jax.random.normal(shape=[1, 1], key=_rng_key),
        jnp.zeros(shape=[10, 1]),
    ],
)
def test_hnn_v0_fwd(mock_hnn_v0, x) -> None:
    """Tests the __call__ of the GaussianHeteroscedasticMLP. The test is parametrized
    for different input types. Batch size generic, batch size = 1, all zeros.

    It checks:
    - if the output is a jax array
    - the output shape
    - if the variance output is always non-negative
    """
    preds = mock_hnn_v0(x)
    assert isinstance(preds, jax.Array)
    assert preds.shape == (x.shape[0], 2)
    assert jnp.all(preds[:, 1] >= 0.0)


@pytest.mark.parametrize(
    "batch",
    [
        (
            jax.random.normal(shape=[10, 1], key=_rng_key),
            jax.random.normal(
                shape=[10],
                key=_rng_key,
            ),
        ),
        (
            jax.random.normal(shape=[1, 1], key=_rng_key),
            jax.random.normal(
                shape=[1],
                key=_rng_key,
            ),
        ),
        (
            jnp.zeros(shape=[10, 1]),
            jnp.zeros(shape=[10]),
        ),
    ],
)
def test_nll_loss(mock_hnn_v0, batch) -> None:
    """Tests the negative log-likelihood loss. The test is parametrized for different
    input types. Batch size generic, batch size = 1, all zeros.

    It checks:
    - if the output is a jax array
    - if the output contains one element (shape=(1,))
    """
    loss_value = nll_loss(mock_hnn_v0, batch)
    assert isinstance(loss_value, jax.Array)
    assert jnp.isscalar(loss_value)
