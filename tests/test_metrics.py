import jax
import pytest


class TestCDF:
    @pytest.fixture
    def setup(self):
        self.samples = jax.random.uniform(jax.random.PRNGKey(0), (1000,)) * 1000
