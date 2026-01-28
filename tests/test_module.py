import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from qmodem.module import MLPBlockV0


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
