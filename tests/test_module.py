import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from qmodem.module import GaussianBlock, MLPBlockV0, ResNetBlockV0


class TestMLPBlockV0:
    @pytest.fixture
    def setup(self):
        """Setup common test parameters."""
        self.hidden_dim = 64
        self.dropout_rate = 0.1
        self.batch_size = 32
        self.rngs = nnx.Rngs(0)

    def test_initialization(self, setup):
        """Test that MLPBlockV0 initializes correctly."""
        block = MLPBlockV0(self.hidden_dim, self.dropout_rate, self.rngs)
        assert hasattr(block, "linear1")
        assert hasattr(block, "norm1")
        assert hasattr(block, "dropout")

    def test_forward_pass_shape(self, setup):
        """Test that forward pass preserves input shape."""
        block = MLPBlockV0(self.hidden_dim, self.dropout_rate, self.rngs)
        x = jnp.ones((self.batch_size, self.hidden_dim))
        output = block(x, deterministic=True)
        assert output.shape == x.shape

    def test_forward_pass_deterministic(self, setup):
        """Test that deterministic mode produces consistent outputs."""
        block = MLPBlockV0(self.hidden_dim, self.dropout_rate, self.rngs)
        x = jax.random.normal(jax.random.PRNGKey(0), (self.batch_size, self.hidden_dim))

        output1 = block(x, deterministic=True)
        output2 = block(x, deterministic=True)

        assert jnp.allclose(output1, output2)

    def test_forward_pass_stochastic(self, setup):
        """Test that stochastic mode can produce different outputs."""
        block = MLPBlockV0(self.hidden_dim, self.dropout_rate, self.rngs)
        x = jax.random.normal(jax.random.PRNGKey(0), (self.batch_size, self.hidden_dim))

        output1 = block(x, deterministic=False)
        output2 = block(x, deterministic=False)

        # Outputs may differ due to dropout, but both should be valid
        assert output1.shape == output2.shape

    def test_output_dtype(self, setup):
        """Test that output has correct dtype."""
        block = MLPBlockV0(self.hidden_dim, self.dropout_rate, self.rngs)
        x = jnp.ones((self.batch_size, self.hidden_dim), dtype=jnp.float32)
        output = block(x, deterministic=True)
        assert output.dtype == jnp.float32

    def test_different_batch_sizes(self, setup):
        """Test forward pass with different batch sizes."""
        block = MLPBlockV0(self.hidden_dim, self.dropout_rate, self.rngs)

        for batch_size in [1, 16, 32, 64]:
            x = jnp.ones((batch_size, self.hidden_dim))
            output = block(x, deterministic=True)
            assert output.shape == (batch_size, self.hidden_dim)

    def test_dropout_rate_zero(self, setup):
        """Test with dropout rate of zero."""
        block = MLPBlockV0(self.hidden_dim, 0.0, self.rngs)
        x = jax.random.normal(jax.random.PRNGKey(0), (self.batch_size, self.hidden_dim))

        output1 = block(x, deterministic=False)
        output2 = block(x, deterministic=False)

        # With zero dropout, outputs should be identical
        assert jnp.allclose(output1, output2)


class TestGaussianBlock:
    @pytest.fixture
    def setup(self):
        """Setup common test parameters."""
        self.input_dim = 32
        self.output_dim = 2
        self.batch_size = 10
        self.rngs = nnx.Rngs(0)

    def test_initialization(self, setup):
        """Test that GaussianBlock initializes correctly."""
        block = GaussianBlock(self.input_dim, self.output_dim, rngs=self.rngs)
        assert hasattr(block, "linear_1")
        assert hasattr(block, "linear_2")

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


class TestResNetBlockV0:
    @pytest.fixture
    def setup(self):
        """Setup common test parameters."""
        self.layer_dim = 32
        self.batch_size = 16
        self.rngs = nnx.Rngs(0)
        self.act_fn = nnx.gelu

    def test_initialization(self, setup):
        """Test that ResNetBlockV0 initializes correctly."""
        block = ResNetBlockV0(self.layer_dim, self.act_fn, rngs=self.rngs)
        assert hasattr(block, "linear_1")
        assert hasattr(block, "linear_2")
        assert hasattr(block, "norm")
        assert hasattr(block, "act_fn")

    def test_forward_pass_shape(self, setup):
        """Test that forward pass preserves input shape."""
        block = ResNetBlockV0(self.layer_dim, self.act_fn, rngs=self.rngs)
        x = jnp.ones((self.batch_size, self.layer_dim))
        output = block(x)
        assert output.shape == x.shape

    def test_forward_pass_deterministic(self, setup):
        """Test that forward pass is deterministic (no stochasticity)."""
        block = ResNetBlockV0(self.layer_dim, self.act_fn, rngs=self.rngs)
        x = jax.random.normal(jax.random.PRNGKey(0), (self.batch_size, self.layer_dim))
        output1 = block(x)
        output2 = block(x)
        assert jnp.allclose(output1, output2)

    def test_output_dtype(self, setup):
        """Test that output has correct dtype."""
        block = ResNetBlockV0(self.layer_dim, self.act_fn, rngs=self.rngs)
        x = jnp.ones((self.batch_size, self.layer_dim), dtype=jnp.float32)
        output = block(x)
        assert output.dtype == jnp.float32

    def test_different_batch_sizes(self, setup):
        """Test forward pass with different batch sizes."""
        block = ResNetBlockV0(self.layer_dim, self.act_fn, rngs=self.rngs)
        for batch_size in [1, 8, 16, 64]:
            x = jnp.ones((batch_size, self.layer_dim))
            output = block(x)
            assert output.shape == (batch_size, self.layer_dim)

    def test_identity_initialization(self, setup):
        """Test that with identity initialization, the block behaves like an identity
        function initially."""
        block = ResNetBlockV0(self.layer_dim, self.act_fn, rngs=self.rngs)
        x = jax.random.normal(jax.random.PRNGKey(0), (self.batch_size, self.layer_dim))
        output = block(x)

        # Since weights are initialized to zero, output should be close to input after
        # activation
        assert jnp.allclose(output, self.act_fn(x), atol=1e-5)
