import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from qmodem.module import (
    GaussianBlock,
    HeteroscedasticCNN1D,
    HeteroscedasticCNN1DV1,
    MLPBlockV0,
    ResNetBlockV0,
    ResNetBlockV1,
    SimpleCNN1D,
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
            self.window_size,
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
            self.window_size,
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
            window_size=48,
            n_filters=4,
            kernel_size=5,
            rngs=nnx.Rngs(0),
        )
        # Conv: (1 * 5 + 1) * 4 = 24 params
        # Dense: (44 * 4 + 1) * 1 = 177 params
        # Total: 201 params
        params = nnx.state(model, nnx.Param)
        total_params = sum(p.size for p in jax.tree.leaves(params))
        assert total_params == 201

    def test_different_window_sizes(self, setup):
        """Test forward pass with different window sizes."""
        for window_size in [24, 48, 96]:
            model = SimpleCNN1D(
                window_size,
                self.n_filters,
                self.kernel_size,
                rngs=nnx.Rngs(0),
            )
            x = jnp.ones((self.batch_size, 1, window_size))
            output = model(x)
            assert output.shape == (self.batch_size,)

    def test_different_batch_sizes(self, setup):
        """Test forward pass with different batch sizes."""
        model = SimpleCNN1D(
            self.window_size,
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
            self.window_size,
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
            self.window_size,
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
            self.window_size,
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
            self.window_size,
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
            self.window_size,
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
            window_size=48,
            n_filters=4,
            kernel_size=5,
            rngs=nnx.Rngs(0),
        )
        # Conv: (1 * 5 + 1) * 4 = 24 params
        # GaussianBlock (2 Linear layers): 2 * (44 * 4 * 1 + 1) = 354 params
        # Total: 378 params
        params = nnx.state(model, nnx.Param)
        total_params = sum(p.size for p in jax.tree.leaves(params))
        assert total_params == 378

    def test_different_window_sizes(self, setup):
        """Test forward pass with different window sizes."""
        for window_size in [24, 48, 96]:
            model = HeteroscedasticCNN1D(
                window_size,
                self.n_filters,
                self.kernel_size,
                rngs=nnx.Rngs(0),
            )
            x = jnp.ones((self.batch_size, 1, window_size))
            output = model(x)
            assert output.shape == (self.batch_size, 2)

    def test_different_batch_sizes(self, setup):
        """Test forward pass with different batch sizes."""
        model = HeteroscedasticCNN1D(
            self.window_size,
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
            self.window_size,
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
            self.window_size,
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
            self.window_size,
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
            self.window_size,
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
            window_size=48,
            n_filters=8,
            kernel_size=5,
            rngs=nnx.Rngs(0),
        )
        # Conv1: (1 * 5 + 1) * 8 = 48 params
        # Conv2: (8 * 5 + 1) * 8 = 328 params
        # GaussianBlock: flatten_size = 8 * 40 = 320; 2 * (320 + 1) = 642 params
        # Total: 1018 params
        params = nnx.state(model, nnx.Param)
        total_params = sum(p.size for p in jax.tree.leaves(params))
        assert total_params == 1018

    def test_different_window_sizes(self, setup):
        """Test forward pass with different window sizes."""
        for window_size in [24, 48, 96]:
            model = HeteroscedasticCNN1DV1(
                window_size,
                self.n_filters,
                self.kernel_size,
                rngs=nnx.Rngs(0),
            )
            x = jnp.ones((self.batch_size, 1, window_size))
            output = model(x)
            assert output.shape == (self.batch_size, 2)

    def test_different_batch_sizes(self, setup):
        """Test forward pass with different batch sizes."""
        model = HeteroscedasticCNN1DV1(
            self.window_size,
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
            self.window_size,
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
