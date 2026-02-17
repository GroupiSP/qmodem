import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from qmodem.module import (
    BayesCNN1D,
    FlipoutConv1D,
    GaussianBlock,
    HeteroscedasticCNN1D,
    HeteroscedasticCNN1DV1,
    MLPBlockV0,
    ResNetBlockV0,
    ResNetBlockV1,
    SimpleCNN1D,
    StandardBayesConv1D,
)


class TestMLPBlockV0:
    @pytest.fixture
    def setup(self):
        """Setup common test parameters."""
        self.hidden_dim = 64
        self.dropout_rate = 0.1
        self.batch_size = 32
        self.act_fn = nnx.gelu
        self.rngs = nnx.Rngs(0)

    def test_forward_pass_shape(self, setup):
        """Test that forward pass preserves input shape."""
        block = MLPBlockV0(self.hidden_dim, self.dropout_rate, self.act_fn, self.rngs)
        x = jnp.ones((self.batch_size, self.hidden_dim))

        output = block(x, rngs=nnx.Rngs(0))
        assert output.shape == x.shape

    def test_forward_pass_deterministic(self, setup):
        """Test that deterministic mode produces consistent outputs."""
        block = MLPBlockV0(self.hidden_dim, self.dropout_rate, self.act_fn, self.rngs)
        x = jax.random.normal(jax.random.PRNGKey(0), (self.batch_size, self.hidden_dim))

        rngs_dropout = nnx.Rngs(0)

        block.eval()  # deterministic
        output1 = block(x, rngs=rngs_dropout)
        output2 = block(x, rngs=rngs_dropout)

        assert jnp.allclose(output1, output2)

    def test_forward_pass_stochastic(self, setup):
        """Test that stochastic mode can produce different outputs."""
        block = MLPBlockV0(self.hidden_dim, self.dropout_rate, self.act_fn, self.rngs)
        x = jax.random.normal(jax.random.PRNGKey(0), (self.batch_size, self.hidden_dim))

        rngs_dropout = nnx.Rngs(0)

        block.train()  # stochastic
        output1 = block(x, rngs=rngs_dropout)
        output2 = block(x, rngs=rngs_dropout)

        # Outputs may differ due to dropout, but both should be valid
        assert output1.shape == output2.shape

    def test_output_dtype(self, setup):
        """Test that output has correct dtype."""
        block = MLPBlockV0(self.hidden_dim, self.dropout_rate, self.act_fn, self.rngs)
        x = jnp.ones((self.batch_size, self.hidden_dim), dtype=jnp.float32)
        output = block(x, rngs=nnx.Rngs(0))
        assert output.dtype == jnp.float32

    def test_different_batch_sizes(self, setup):
        """Test forward pass with different batch sizes."""
        block = MLPBlockV0(self.hidden_dim, self.dropout_rate, self.act_fn, self.rngs)

        rngs_dropout = nnx.Rngs(0)

        for batch_size in [1, 16, 32, 64]:
            x = jnp.ones((batch_size, self.hidden_dim))
            output = block(x, rngs=rngs_dropout)
            assert output.shape == (batch_size, self.hidden_dim)

    def test_dropout_rate_zero(self, setup):
        """Test with dropout rate of zero."""
        block = MLPBlockV0(self.hidden_dim, 0.0, self.act_fn, self.rngs)
        x = jax.random.normal(jax.random.PRNGKey(0), (self.batch_size, self.hidden_dim))

        rngs_dropout = nnx.Rngs(0)
        block.train()  # stochastic

        output1 = block(x, rngs=rngs_dropout)
        output2 = block(x, rngs=rngs_dropout)

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


class TestResNetBlockV1:
    @pytest.fixture
    def setup(self):
        """Setup common test parameters."""
        self.layer_dim = 32
        self.dropout_rate = 0.1
        self.batch_size = 16
        self.rngs = nnx.Rngs(0)
        self.act_fn = nnx.gelu

    def test_forward_pass_shape(self, setup):
        """Test that forward pass preserves input shape."""
        block = ResNetBlockV1(
            self.layer_dim, self.dropout_rate, self.act_fn, rngs=self.rngs
        )
        x = jnp.ones((self.batch_size, self.layer_dim))

        output = block(x, rngs=nnx.Rngs(0))
        assert output.shape == x.shape

    def test_forward_pass_deterministic(self, setup):
        """Test that deterministic mode produces consistent outputs."""
        block = ResNetBlockV1(
            self.layer_dim, self.dropout_rate, self.act_fn, rngs=self.rngs
        )
        x = jax.random.normal(jax.random.PRNGKey(0), (self.batch_size, self.layer_dim))

        rngs_dropout = nnx.Rngs(0)

        block.eval()  # deterministic
        output1 = block(x, rngs=rngs_dropout)
        output2 = block(x, rngs=rngs_dropout)

        assert jnp.allclose(output1, output2)

    def test_forward_pass_stochastic(self, setup):
        """Test that stochastic mode can produce different outputs due to dropout."""
        block = ResNetBlockV1(
            self.layer_dim, self.dropout_rate, self.act_fn, rngs=self.rngs
        )
        x = jax.random.normal(jax.random.PRNGKey(0), (self.batch_size, self.layer_dim))
        rngs_dropout = nnx.Rngs(0)

        block.train()  # stochastic
        output1 = block(x, rngs=rngs_dropout)
        output2 = block(x, rngs=rngs_dropout)

        # Outputs may differ due to dropout, but both should be valid
        assert output1.shape == output2.shape

    def test_output_dtype(self, setup):
        """Test that output has correct dtype."""
        block = ResNetBlockV1(
            self.layer_dim, self.dropout_rate, self.act_fn, rngs=self.rngs
        )
        x = jnp.ones((self.batch_size, self.layer_dim), dtype=jnp.float32)
        output = block(x, rngs=nnx.Rngs(0))
        assert output.dtype == jnp.float32

    def test_different_batch_sizes(self, setup):
        """Test forward pass with different batch sizes."""
        block = ResNetBlockV1(
            self.layer_dim, self.dropout_rate, self.act_fn, rngs=self.rngs
        )
        rngs_dropout = nnx.Rngs(0)

        for batch_size in [1, 16, 32, 64]:
            x = jnp.ones((batch_size, self.layer_dim))
            output = block(x, rngs=rngs_dropout)
            assert output.shape == (batch_size, self.layer_dim)

    def test_identity_initialization(self, setup):
        """Test that with identity initialization, the block behaves like an identity
        function initially."""
        block = ResNetBlockV1(
            self.layer_dim, self.dropout_rate, self.act_fn, rngs=self.rngs
        )
        x = jax.random.normal(jax.random.PRNGKey(0), (self.batch_size, self.layer_dim))

        block.eval()  # deterministic
        output = block(x, rngs=nnx.Rngs(0))

        # Since weights are initialized to zero, output should be close to input after
        # activation
        assert jnp.allclose(output, self.act_fn(x), atol=1e-5)

    def test_dropout_rate_zero(self, setup):
        """Test with dropout rate of zero."""
        block = ResNetBlockV1(
            self.layer_dim, dropout_rate=0.0, act_fn=self.act_fn, rngs=self.rngs
        )
        x = jax.random.normal(jax.random.PRNGKey(0), (self.batch_size, self.layer_dim))

        rngs_dropout = nnx.Rngs(0)
        block.train()  # stochastic

        output1 = block(x, rngs=rngs_dropout)
        output2 = block(x, rngs=rngs_dropout)
        # With zero dropout, outputs should be identical in stochastic mode
        assert jnp.allclose(output1, output2)


class TestSimpleCNN1D:
    @pytest.fixture
    def setup(self):
        """Setup common test parameters."""
        self.window_size = 48
        self.n_filters = 4
        self.kernel_size = 5
        self.batch_size = 16
        self.rngs = nnx.Rngs(0)

    def test_forward_pass_shape(self, setup):
        """Test that forward pass produces correct output shape."""
        model = SimpleCNN1D(
            self.n_filters,
            self.kernel_size,
            rngs=self.rngs,
        )
        # Input: (batch, 1, window_size)
        x = jnp.ones((self.batch_size, 1, self.window_size))
        output = model(x)
        # Output: (batch,)
        assert output.shape == (self.batch_size,)

    def test_output_dtype(self, setup):
        """Test that output has correct dtype."""
        model = SimpleCNN1D(
            self.n_filters,
            self.kernel_size,
            rngs=self.rngs,
        )
        x = jnp.ones((self.batch_size, 1, self.window_size), dtype=jnp.float32)
        output = model(x)
        assert output.dtype == jnp.float32

    def test_parameter_count(self, setup):
        """Test that parameter count is as expected."""
        model = SimpleCNN1D(
            n_filters=4,
            kernel_size=5,
            rngs=nnx.Rngs(0),
        )
        # Conv: (1 * 5 + 1) * 4 = 24 params
        # Dense: (4 + 1) * 1 = 5 params
        # Total: 29 params
        params = nnx.state(model, nnx.Param)
        total_params = sum(p.size for p in jax.tree.leaves(params))
        assert total_params == 29

    def test_variable_window_sizes(self, setup):
        """Test that a single model handles different window sizes."""
        model = SimpleCNN1D(
            self.n_filters,
            self.kernel_size,
            rngs=nnx.Rngs(0),
        )
        for window_size in [24, 48, 96]:
            x = jnp.ones((self.batch_size, 1, window_size))
            output = model(x)
            assert output.shape == (self.batch_size,)

    def test_different_batch_sizes(self, setup):
        """Test forward pass with different batch sizes."""
        model = SimpleCNN1D(
            self.n_filters,
            self.kernel_size,
            rngs=self.rngs,
        )
        for batch_size in [1, 8, 16, 32]:
            x = jnp.ones((batch_size, 1, self.window_size))
            output = model(x)
            assert output.shape == (batch_size,)

    def test_gradient_computation(self, setup):
        """Test that gradients can be computed through the model."""
        model = SimpleCNN1D(
            self.n_filters,
            self.kernel_size,
            rngs=self.rngs,
        )
        x = jax.random.normal(
            jax.random.PRNGKey(0), (self.batch_size, 1, self.window_size)
        )
        target = jnp.ones((self.batch_size,))

        def loss_fn(model):
            pred = model(x)
            return jnp.mean((pred - target) ** 2)

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        assert jnp.isfinite(loss)
        # Check that gradients exist and are finite
        grad_params = nnx.state(grads, nnx.Param)
        assert all(jnp.all(jnp.isfinite(g)) for g in jax.tree.leaves(grad_params))


class TestHeteroscedasticCNN1D:
    @pytest.fixture
    def setup(self):
        """Setup common test parameters."""
        self.window_size = 48
        self.n_filters = 4
        self.kernel_size = 5
        self.batch_size = 16
        self.rngs = nnx.Rngs(0)

    def test_forward_pass_shape(self, setup):
        """Test that forward pass produces correct output shape."""
        model = HeteroscedasticCNN1D(
            self.n_filters,
            self.kernel_size,
            rngs=self.rngs,
        )
        # Input: (batch, 1, window_size)
        x = jnp.ones((self.batch_size, 1, self.window_size))
        output = model(x)
        # Output: (batch, 2) - concatenated [mu, var_positive]
        assert output.shape == (self.batch_size, 2)

    def test_output_dtype(self, setup):
        """Test that output has correct dtype."""
        model = HeteroscedasticCNN1D(
            self.n_filters,
            self.kernel_size,
            rngs=self.rngs,
        )
        x = jnp.ones((self.batch_size, 1, self.window_size), dtype=jnp.float32)
        output = model(x)
        assert output.dtype == jnp.float32

    def test_split_mu_and_var(self, setup):
        """Test that mu and var_positive can be split correctly."""
        model = HeteroscedasticCNN1D(
            self.n_filters,
            self.kernel_size,
            rngs=self.rngs,
        )
        x = jax.random.normal(
            jax.random.PRNGKey(0), (self.batch_size, 1, self.window_size)
        )
        output = model(x)

        # Split output into mu and var_positive
        mu = output[:, 0]
        var_positive = output[:, 1]

        assert mu.shape == (self.batch_size,)
        assert var_positive.shape == (self.batch_size,)

    def test_variance_is_positive(self, setup):
        """Test that variance output is always positive."""
        model = HeteroscedasticCNN1D(
            self.n_filters,
            self.kernel_size,
            rngs=self.rngs,
        )
        x = jax.random.normal(
            jax.random.PRNGKey(0), (self.batch_size, 1, self.window_size)
        )
        output = model(x)

        var_positive = output[:, 1]
        assert jnp.all(var_positive >= 0)

    def test_parameter_count(self, setup):
        """Test that parameter count is as expected."""
        model = HeteroscedasticCNN1D(
            n_filters=4,
            kernel_size=5,
            rngs=nnx.Rngs(0),
        )
        # Conv: (1 * 5 + 1) * 4 = 24 params
        # GaussianBlock (2 Linear layers): 2 * (4 * 1 + 1) = 10 params
        # Total: 34 params
        params = nnx.state(model, nnx.Param)
        total_params = sum(p.size for p in jax.tree.leaves(params))
        assert total_params == 34

    def test_variable_window_sizes(self, setup):
        """Test that a single model handles different window sizes."""
        model = HeteroscedasticCNN1D(
            self.n_filters,
            self.kernel_size,
            rngs=nnx.Rngs(0),
        )
        for window_size in [24, 48, 96]:
            x = jnp.ones((self.batch_size, 1, window_size))
            output = model(x)
            assert output.shape == (self.batch_size, 2)

    def test_different_batch_sizes(self, setup):
        """Test forward pass with different batch sizes."""
        model = HeteroscedasticCNN1D(
            self.n_filters,
            self.kernel_size,
            rngs=self.rngs,
        )
        for batch_size in [1, 8, 16, 32]:
            x = jnp.ones((batch_size, 1, self.window_size))
            output = model(x)
            assert output.shape == (batch_size, 2)

    def test_gradient_computation(self, setup):
        """Test that gradients can be computed through the model."""
        model = HeteroscedasticCNN1D(
            self.n_filters,
            self.kernel_size,
            rngs=self.rngs,
        )
        x = jax.random.normal(
            jax.random.PRNGKey(0), (self.batch_size, 1, self.window_size)
        )
        target = jnp.ones((self.batch_size,))

        def loss_fn(model):
            output = model(x)
            mu = output[:, 0]
            return jnp.mean((mu - target) ** 2)

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        assert jnp.isfinite(loss)
        # Check that gradients exist and are finite
        grad_params = nnx.state(grads, nnx.Param)
        assert all(jnp.all(jnp.isfinite(g)) for g in jax.tree.leaves(grad_params))


class TestHeteroscedasticCNN1DV1:
    @pytest.fixture
    def setup(self):
        """Setup common test parameters."""
        self.window_size = 48
        self.n_filters = 8
        self.kernel_size = 5
        self.batch_size = 16
        self.rngs = nnx.Rngs(0)

    def test_forward_pass_shape(self, setup):
        """Test that forward pass produces correct output shape."""
        model = HeteroscedasticCNN1DV1(
            self.n_filters,
            self.kernel_size,
            rngs=self.rngs,
        )
        x = jnp.ones((self.batch_size, 1, self.window_size))
        output = model(x)
        assert output.shape == (self.batch_size, 2)

    def test_output_dtype(self, setup):
        """Test that output has correct dtype."""
        model = HeteroscedasticCNN1DV1(
            self.n_filters,
            self.kernel_size,
            rngs=self.rngs,
        )
        x = jnp.ones((self.batch_size, 1, self.window_size), dtype=jnp.float32)
        output = model(x)
        assert output.dtype == jnp.float32

    def test_variance_is_positive(self, setup):
        """Test that variance output is always positive."""
        model = HeteroscedasticCNN1DV1(
            self.n_filters,
            self.kernel_size,
            rngs=self.rngs,
        )
        x = jax.random.normal(
            jax.random.PRNGKey(0), (self.batch_size, 1, self.window_size)
        )
        output = model(x)
        var_positive = output[:, 1]
        assert jnp.all(var_positive >= 0)

    def test_parameter_count(self, setup):
        """Test that parameter count is as expected."""
        model = HeteroscedasticCNN1DV1(
            n_filters=8,
            kernel_size=5,
            rngs=nnx.Rngs(0),
        )
        # Conv1: (1 * 5 + 1) * 8 = 48 params
        # Conv2: (8 * 5 + 1) * 8 = 328 params
        # GaussianBlock: 2 * (8 * 1 + 1) = 18 params
        # Total: 394 params
        params = nnx.state(model, nnx.Param)
        total_params = sum(p.size for p in jax.tree.leaves(params))
        assert total_params == 394

    def test_variable_window_sizes(self, setup):
        """Test that a single model handles different window sizes."""
        model = HeteroscedasticCNN1DV1(
            self.n_filters,
            self.kernel_size,
            rngs=nnx.Rngs(0),
        )
        for window_size in [24, 48, 96]:
            x = jnp.ones((self.batch_size, 1, window_size))
            output = model(x)
            assert output.shape == (self.batch_size, 2)

    def test_different_batch_sizes(self, setup):
        """Test forward pass with different batch sizes."""
        model = HeteroscedasticCNN1DV1(
            self.n_filters,
            self.kernel_size,
            rngs=self.rngs,
        )
        for batch_size in [1, 8, 16, 32]:
            x = jnp.ones((batch_size, 1, self.window_size))
            output = model(x)
            assert output.shape == (batch_size, 2)

    def test_gradient_computation(self, setup):
        """Test that gradients can be computed through the model."""
        model = HeteroscedasticCNN1DV1(
            self.n_filters,
            self.kernel_size,
            rngs=self.rngs,
        )
        x = jax.random.normal(
            jax.random.PRNGKey(0), (self.batch_size, 1, self.window_size)
        )
        target = jnp.ones((self.batch_size,))

        def loss_fn(model):
            output = model(x)
            mu = output[:, 0]
            return jnp.mean((mu - target) ** 2)

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        assert jnp.isfinite(loss)
        grad_params = nnx.state(grads, nnx.Param)
        assert all(jnp.all(jnp.isfinite(g)) for g in jax.tree.leaves(grad_params))


class TestStandardBayesConv1D:
    @pytest.fixture
    def setup(self):
        """Setup common test parameters."""
        self.in_features = 1
        self.out_features = 4
        self.kernel_size = 5
        self.batch_size = 16
        self.length = 48
        self.rngs = nnx.Rngs(0)
        self.key = jax.random.PRNGKey(42)

    def test_forward_pass_shape(self, setup):
        """Test that forward pass produces correct output shape."""
        layer = StandardBayesConv1D(
            self.in_features, self.out_features, self.kernel_size, rngs=self.rngs
        )
        x = jnp.ones((self.batch_size, self.length, self.in_features))
        output = layer(x, key=self.key)
        expected_length = self.length - self.kernel_size + 1
        assert output.shape == (self.batch_size, expected_length, self.out_features)

    def test_output_dtype(self, setup):
        """Test that output has correct dtype."""
        layer = StandardBayesConv1D(
            self.in_features, self.out_features, self.kernel_size, rngs=self.rngs
        )
        x = jnp.ones(
            (self.batch_size, self.length, self.in_features), dtype=jnp.float32
        )
        output = layer(x, key=self.key)
        assert output.dtype == jnp.float32

    def test_stochastic_forward(self, setup):
        """Test that different keys produce different outputs."""
        layer = StandardBayesConv1D(
            self.in_features, self.out_features, self.kernel_size, rngs=self.rngs
        )
        x = jax.random.normal(
            jax.random.PRNGKey(0),
            (self.batch_size, self.length, self.in_features),
        )
        out1 = layer(x, key=jax.random.PRNGKey(1))
        out2 = layer(x, key=jax.random.PRNGKey(2))
        assert not jnp.allclose(out1, out2)

    def test_parameter_count(self, setup):
        """Test that parameter count is 2x a deterministic nnx.Conv."""
        bayes_layer = StandardBayesConv1D(
            self.in_features, self.out_features, self.kernel_size, rngs=self.rngs
        )
        det_layer = nnx.Conv(
            in_features=self.in_features,
            out_features=self.out_features,
            kernel_size=(self.kernel_size,),
            padding="VALID",
            rngs=self.rngs,
        )
        bayes_params = sum(
            p.size for p in jax.tree.leaves(nnx.state(bayes_layer, nnx.Param))
        )
        det_params = sum(
            p.size for p in jax.tree.leaves(nnx.state(det_layer, nnx.Param))
        )
        assert bayes_params == 2 * det_params

    def test_kl_divergence_positive(self, setup):
        """Test that KL divergence is non-negative."""
        layer = StandardBayesConv1D(
            self.in_features, self.out_features, self.kernel_size, rngs=self.rngs
        )
        kl = layer.kl_divergence()
        assert jnp.isfinite(kl)
        assert kl >= 0.0

    def test_gradient_computation(self, setup):
        """Test that gradients can be computed through the layer."""
        layer = StandardBayesConv1D(
            self.in_features, self.out_features, self.kernel_size, rngs=self.rngs
        )
        x = jax.random.normal(
            jax.random.PRNGKey(0),
            (self.batch_size, self.length, self.in_features),
        )

        def loss_fn(layer):
            return jnp.mean(layer(x, key=self.key))

        loss, grads = nnx.value_and_grad(loss_fn)(layer)
        assert jnp.isfinite(loss)
        grad_params = nnx.state(grads, nnx.Param)
        assert all(jnp.all(jnp.isfinite(g)) for g in jax.tree.leaves(grad_params))


class TestFlipoutConv1D:
    @pytest.fixture
    def setup(self):
        """Setup common test parameters."""
        self.in_features = 1
        self.out_features = 4
        self.kernel_size = 5
        self.batch_size = 16
        self.length = 48
        self.rngs = nnx.Rngs(0)
        self.key = jax.random.PRNGKey(42)

    def test_forward_pass_shape(self, setup):
        """Test that forward pass produces correct output shape."""
        layer = FlipoutConv1D(
            self.in_features, self.out_features, self.kernel_size, rngs=self.rngs
        )
        x = jnp.ones((self.batch_size, self.length, self.in_features))
        output = layer(x, key=self.key)
        expected_length = self.length - self.kernel_size + 1
        assert output.shape == (self.batch_size, expected_length, self.out_features)

    def test_output_dtype(self, setup):
        """Test that output has correct dtype."""
        layer = FlipoutConv1D(
            self.in_features, self.out_features, self.kernel_size, rngs=self.rngs
        )
        x = jnp.ones(
            (self.batch_size, self.length, self.in_features), dtype=jnp.float32
        )
        output = layer(x, key=self.key)
        assert output.dtype == jnp.float32

    def test_stochastic_forward(self, setup):
        """Test that different keys produce different outputs."""
        layer = FlipoutConv1D(
            self.in_features, self.out_features, self.kernel_size, rngs=self.rngs
        )
        x = jax.random.normal(
            jax.random.PRNGKey(0),
            (self.batch_size, self.length, self.in_features),
        )
        out1 = layer(x, key=jax.random.PRNGKey(1))
        out2 = layer(x, key=jax.random.PRNGKey(2))
        assert not jnp.allclose(out1, out2)

    def test_parameter_count(self, setup):
        """Test that parameter count is 2x a deterministic nnx.Conv."""
        bayes_layer = FlipoutConv1D(
            self.in_features, self.out_features, self.kernel_size, rngs=self.rngs
        )
        det_layer = nnx.Conv(
            in_features=self.in_features,
            out_features=self.out_features,
            kernel_size=(self.kernel_size,),
            padding="VALID",
            rngs=self.rngs,
        )
        bayes_params = sum(
            p.size for p in jax.tree.leaves(nnx.state(bayes_layer, nnx.Param))
        )
        det_params = sum(
            p.size for p in jax.tree.leaves(nnx.state(det_layer, nnx.Param))
        )
        assert bayes_params == 2 * det_params

    def test_kl_divergence_positive(self, setup):
        """Test that KL divergence is non-negative."""
        layer = FlipoutConv1D(
            self.in_features, self.out_features, self.kernel_size, rngs=self.rngs
        )
        kl = layer.kl_divergence()
        assert jnp.isfinite(kl)
        assert kl >= 0.0

    def test_gradient_computation(self, setup):
        """Test that gradients can be computed through the layer."""
        layer = FlipoutConv1D(
            self.in_features, self.out_features, self.kernel_size, rngs=self.rngs
        )
        x = jax.random.normal(
            jax.random.PRNGKey(0),
            (self.batch_size, self.length, self.in_features),
        )

        def loss_fn(layer):
            return jnp.mean(layer(x, key=self.key))

        loss, grads = nnx.value_and_grad(loss_fn)(layer)
        assert jnp.isfinite(loss)
        grad_params = nnx.state(grads, nnx.Param)
        assert all(jnp.all(jnp.isfinite(g)) for g in jax.tree.leaves(grad_params))


class TestBayesCNN1D:
    @pytest.fixture
    def setup(self):
        """Setup common test parameters."""
        self.window_size = 48
        self.n_filters = 4
        self.kernel_size = 5
        self.batch_size = 16
        self.rngs = nnx.Rngs(0)
        self.key = jax.random.PRNGKey(42)

    @pytest.mark.parametrize("conv_cls", [StandardBayesConv1D, FlipoutConv1D])
    def test_forward_pass_shape(self, setup, conv_cls):
        """Test that forward pass produces correct output shape."""
        model = BayesCNN1D(conv_cls, self.n_filters, self.kernel_size, rngs=self.rngs)
        x = jnp.ones((self.batch_size, 1, self.window_size))
        output = model(x, rngs=nnx.Rngs(params=42))
        assert output.shape == (self.batch_size, 2)

    @pytest.mark.parametrize("conv_cls", [StandardBayesConv1D, FlipoutConv1D])
    def test_output_dtype(self, setup, conv_cls):
        """Test that output has correct dtype."""
        model = BayesCNN1D(conv_cls, self.n_filters, self.kernel_size, rngs=self.rngs)
        x = jnp.ones((self.batch_size, 1, self.window_size), dtype=jnp.float32)
        output = model(x, rngs=nnx.Rngs(params=42))
        assert output.dtype == jnp.float32

    @pytest.mark.parametrize("conv_cls", [StandardBayesConv1D, FlipoutConv1D])
    def test_variance_is_positive(self, setup, conv_cls):
        """Test that variance output is always positive."""
        model = BayesCNN1D(conv_cls, self.n_filters, self.kernel_size, rngs=self.rngs)
        x = jax.random.normal(
            jax.random.PRNGKey(0), (self.batch_size, 1, self.window_size)
        )
        output = model(x, rngs=nnx.Rngs(params=42))
        var_positive = output[:, 1]
        assert jnp.all(var_positive >= 0)

    @pytest.mark.parametrize("conv_cls", [StandardBayesConv1D, FlipoutConv1D])
    def test_parameter_count(self, setup, conv_cls):
        """Test conv params are 2x those of HeteroscedasticCNN1D's conv layer."""
        bayes_model = BayesCNN1D(conv_cls, n_filters=4, kernel_size=5, rngs=nnx.Rngs(0))
        det_model = HeteroscedasticCNN1D(n_filters=4, kernel_size=5, rngs=nnx.Rngs(0))
        bayes_params = sum(
            p.size for p in jax.tree.leaves(nnx.state(bayes_model, nnx.Param))
        )
        det_params = sum(
            p.size for p in jax.tree.leaves(nnx.state(det_model, nnx.Param))
        )
        # GaussianBlock is shared (10 params); conv is doubled (24 → 48)
        det_conv_params = det_params - 10  # 24
        bayes_conv_params = bayes_params - 10  # 48
        assert bayes_conv_params == 2 * det_conv_params

    @pytest.mark.parametrize("conv_cls", [StandardBayesConv1D, FlipoutConv1D])
    def test_stochastic_forward(self, setup, conv_cls):
        """Test that different keys produce different outputs."""
        model = BayesCNN1D(conv_cls, self.n_filters, self.kernel_size, rngs=self.rngs)
        x = jax.random.normal(
            jax.random.PRNGKey(0), (self.batch_size, 1, self.window_size)
        )
        out1 = model(x, rngs=nnx.Rngs(params=1))
        out2 = model(x, rngs=nnx.Rngs(params=2))
        assert not jnp.allclose(out1, out2)

    @pytest.mark.parametrize("conv_cls", [StandardBayesConv1D, FlipoutConv1D])
    def test_kl_divergence_positive(self, setup, conv_cls):
        """Test that KL divergence is non-negative."""
        model = BayesCNN1D(conv_cls, self.n_filters, self.kernel_size, rngs=self.rngs)
        kl = model.kl_divergence()
        assert jnp.isfinite(kl)
        assert kl >= 0.0

    @pytest.mark.parametrize("conv_cls", [StandardBayesConv1D, FlipoutConv1D])
    def test_variable_window_sizes(self, setup, conv_cls):
        """Test that a single model handles different window sizes."""
        model = BayesCNN1D(conv_cls, self.n_filters, self.kernel_size, rngs=nnx.Rngs(0))
        for window_size in [24, 48, 96]:
            x = jnp.ones((self.batch_size, 1, window_size))
            output = model(x, rngs=nnx.Rngs(params=42))
            assert output.shape == (self.batch_size, 2)

    @pytest.mark.parametrize("conv_cls", [StandardBayesConv1D, FlipoutConv1D])
    def test_different_batch_sizes(self, setup, conv_cls):
        """Test forward pass with different batch sizes."""
        model = BayesCNN1D(conv_cls, self.n_filters, self.kernel_size, rngs=self.rngs)
        for batch_size in [1, 8, 16, 32]:
            x = jnp.ones((batch_size, 1, self.window_size))
            output = model(x, rngs=nnx.Rngs(params=42))
            assert output.shape == (batch_size, 2)

    @pytest.mark.parametrize("conv_cls", [StandardBayesConv1D, FlipoutConv1D])
    def test_gradient_computation(self, setup, conv_cls):
        """Test that gradients can be computed through the model."""
        model = BayesCNN1D(conv_cls, self.n_filters, self.kernel_size, rngs=self.rngs)
        x = jax.random.normal(
            jax.random.PRNGKey(0), (self.batch_size, 1, self.window_size)
        )

        def loss_fn(model):
            output = model(x, rngs=nnx.Rngs(params=42))
            mu = output[:, 0]
            return jnp.mean(mu**2) + model.kl_divergence()

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        assert jnp.isfinite(loss)
        grad_params = nnx.state(grads, nnx.Param)
        assert all(jnp.all(jnp.isfinite(g)) for g in jax.tree.leaves(grad_params))
