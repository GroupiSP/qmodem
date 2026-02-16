import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from qmodem.module import HNNV0, MCDCNN1D, MCDNetV0, nll_loss, nll_loss_mcd

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


# --- MCDCNN1D tests ---


@pytest.fixture
def mock_mcdcnn1d() -> MCDCNN1D:
    return MCDCNN1D(
        n_filters=4, kernel_size=5, dropout_rate=0.1, rngs=nnx.Rngs(params=0, dropout=1)
    )


def test_mcdcnn1d_init(mock_mcdcnn1d) -> None:
    """Tests the __init__ of MCDCNN1D.

    It checks:
    - model stores expected hyperparameters
    - model contains a Dropout layer
    """
    assert mock_mcdcnn1d.n_filters == 4
    assert mock_mcdcnn1d.kernel_size == 5
    assert mock_mcdcnn1d.dropout_rate == 0.1
    assert isinstance(mock_mcdcnn1d.dropout, nnx.Dropout)


@pytest.mark.parametrize(
    "x",
    [
        jax.random.normal(shape=[10, 1, 30], key=_rng_key),
        jax.random.normal(shape=[1, 1, 30], key=_rng_key),
        jnp.zeros(shape=[5, 1, 30]),
    ],
)
def test_mcdcnn1d_fwd(mock_mcdcnn1d, x) -> None:
    """Tests the __call__ of MCDCNN1D.

    It checks:
    - the output is a jax array
    - the output shape is (batch, 2)
    - the variance output is always non-negative
    """
    rngs = nnx.Rngs(dropout=42)
    preds = mock_mcdcnn1d(x, rngs=rngs)
    assert isinstance(preds, jax.Array)
    assert preds.shape == (x.shape[0], 2)
    assert jnp.all(preds[:, 1] >= 0.0)


def test_mcdcnn1d_dropout_stochasticity(mock_mcdcnn1d) -> None:
    """Tests that MC Dropout produces different outputs across forward passes in train
    mode, and identical outputs in eval mode."""
    x = jax.random.normal(shape=[2, 1, 30], key=_rng_key)
    rng = nnx.Rngs(dropout=0)

    mock_mcdcnn1d.train()
    out1 = mock_mcdcnn1d(x, rngs=rng)
    out2 = mock_mcdcnn1d(x, rngs=rng)
    # Different dropout masks should produce different outputs
    assert not jnp.allclose(out1, out2)

    mock_mcdcnn1d.eval()
    out3 = mock_mcdcnn1d(x, rngs=rng)
    out4 = mock_mcdcnn1d(x, rngs=rng)
    # Eval mode is deterministic
    assert jnp.allclose(out3, out4)


# --- nll_loss_mcd tests ---


@pytest.mark.parametrize(
    "batch",
    [
        (
            jax.random.normal(shape=[10, 1, 30], key=_rng_key),
            jax.random.normal(shape=[10], key=_rng_key),
        ),
        (
            jax.random.normal(shape=[1, 1, 30], key=_rng_key),
            jax.random.normal(shape=[1], key=_rng_key),
        ),
        (
            jnp.zeros(shape=[5, 1, 30]),
            jnp.zeros(shape=[5]),
        ),
    ],
)
def test_nll_loss_mcd(mock_mcdcnn1d, batch) -> None:
    """Tests the NLL loss with MC Dropout rngs forwarding.

    It checks:
    - the output is a jax array
    - the output is scalar
    """
    rngs = nnx.Rngs(dropout=42)
    loss_value = nll_loss_mcd(mock_mcdcnn1d, batch, rngs)
    assert isinstance(loss_value, jax.Array)
    assert jnp.isscalar(loss_value)


def test_nll_loss_mcd_forwards_rngs(mock_mcdcnn1d) -> None:
    """Tests that nll_loss_mcd produces different losses in train mode (dropout active)
    due to rngs being forwarded to the model."""
    batch = (
        jax.random.normal(shape=[10, 1, 30], key=_rng_key),
        jax.random.normal(shape=[10], key=_rng_key),
    )
    rng = nnx.Rngs(dropout=0)

    mock_mcdcnn1d.train()
    loss1 = nll_loss_mcd(mock_mcdcnn1d, batch, rng)
    loss2 = nll_loss_mcd(mock_mcdcnn1d, batch, rng)
    # Different dropout masks should produce different losses
    assert not jnp.allclose(loss1, loss2)
