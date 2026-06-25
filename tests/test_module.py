import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from qmodem.module import (
    GaussianBlock,
    StandardBayesConv1D,
)


@pytest.fixture
def std_bayes_conv1d_layer():
    """Fixture to create a StandardBayesConv1D layer for testing."""
    in_features = 1
    out_features = 4
    kernel_size = 5
    rngs = nnx.Rngs(0)
    return StandardBayesConv1D(in_features, out_features, kernel_size, rngs=rngs)


@pytest.fixture
def X_batch_time_series_1d():
    """Fixture to create a batch of 1D time-series data for testing."""
    batch_size = 16
    time_window_size = 48
    in_features = 1
    return jax.random.normal(
        jax.random.key(0), (batch_size, time_window_size, in_features)
    )


@nnx.vmap(in_axes=(None, 0, 0), out_axes=0)
def _model_fwd(model: nnx.Module, x_i: jax.Array, key: jax.Array) -> jax.Array:
    # NOTE: we need to add a batch dimension to x_i since the model expects a batch of inputs.
    # NOTE: we need to remove the batch dimension from the output since we only want the output for the single input x_i.
    return model(x_i[None], rngs=nnx.Rngs(default=key))[0]


@nnx.vmap(in_axes=(None, 0), out_axes=0)
def _sample_kernel(model: nnx.Module, key: jax.Array) -> jax.Array:
    k1, _ = jax.random.split(key, 2)
    k_sigma = jax.nn.softplus(model.kernel_rho.value)
    eps_k = jax.random.normal(k1, model.kernel_mu.value.shape)
    return model.kernel_mu.value + k_sigma * eps_k


class TestGaussianBlock:
    @pytest.fixture
    def setup(self):
        """Setup common test parameters."""
        self.input_dim = 32
        self.output_dim = 2
        self.batch_size = 10
        self.rngs = nnx.Rngs(0)

    def test_forward_pass_shape(self, setup):
        """Test that forward pass preserves input shape."""
        block = GaussianBlock(self.input_dim, self.output_dim, rngs=self.rngs)
        x = jnp.ones((self.batch_size, self.input_dim))
        output = block(x)
        assert output.shape == (self.batch_size, self.output_dim * 2)

    def test_output_dtype(self, setup):
        """Test that output has correct dtype."""
        block = GaussianBlock(self.input_dim, self.output_dim, rngs=self.rngs)
        x = jnp.ones((self.batch_size, self.input_dim), dtype=jnp.float32)
        output = block(x)
        assert output.dtype == jnp.float32

    def test_forward_pass_values(self, setup):
        """Test that forward pass produces non-negative variance outputs."""
        block = GaussianBlock(self.input_dim, self.output_dim, rngs=self.rngs)
        x = jax.random.normal(jax.random.PRNGKey(0), (self.batch_size, self.input_dim))
        output = block(x)

        var_positive = output[:, self.output_dim :]

        assert jnp.all(var_positive >= 0)  # Ensure variance is non-negative

    def test_distribution_has_zero_covariance(self, setup):
        """Test that the output distribution is close to a multivariate normal with the
        predicted mean and diagonal covariance."""
        block = GaussianBlock(self.input_dim, self.output_dim, rngs=self.rngs)
        x = jax.random.normal(jax.random.PRNGKey(0), (1, self.input_dim))
        output = block(x)

        mu = output[:, : self.output_dim]
        var_positive = output[:, self.output_dim :]

        # Sample from the predicted distribution
        rng = jax.random.PRNGKey(42)
        eps = jax.random.normal(rng, shape=(1000, self.output_dim))
        samples = mu + jnp.sqrt(var_positive) * eps

        # Compute sample mean and covariance
        sample_mean = jnp.mean(samples, axis=0)
        sample_cov = jnp.cov(samples, rowvar=False)

        # Check that sample mean is close to predicted mean
        assert jnp.allclose(sample_mean, jnp.mean(mu, axis=0), atol=0.1)

        # Check that off-diagonal covariance terms are close to zero
        off_diag_cov = sample_cov - jnp.diag(jnp.diag(sample_cov))
        assert jnp.allclose(off_diag_cov, jnp.zeros_like(off_diag_cov), atol=0.1)


def test_standard_bayes_conv1d_weight_correlation(
    std_bayes_conv1d_layer, X_batch_time_series_1d
):
    """Test that the weights sampled from StandardBayesConv1D are not perfectly
    correlated, i.e. that different random keys produce different weight samples."""
    n_samples = len(X_batch_time_series_1d)
    key = jax.random.key(0)
    subkeys = jax.random.split(key, num=n_samples)

    # Shape: (n_samples, kernel_size, in_features, out_features)
    weight_samples = _sample_kernel(std_bayes_conv1d_layer, subkeys)

    # Each row = one flattened kernel sample; rowvar=True gives (n_samples, n_samples)
    W = weight_samples.reshape(n_samples, -1)
    weight_correlation = jnp.corrcoef(W)

    # If broken, all samples would be identical → correlation matrix all-ones
    assert not jnp.allclose(weight_correlation, jnp.ones_like(weight_correlation))
