import pytest
from flax import nnx

from qmodem import GaussianHeteroscedasticMLP


@pytest.fixture
def mock_gauss_het_mlp() -> GaussianHeteroscedasticMLP:
    return GaussianHeteroscedasticMLP(dimensions=[1, 10, 10], rngs=nnx.Rngs(0))


def test_gauss_het_mlp_init(mock_gauss_het_mlp) -> None:
    """Tests the __init__ of the GaussianHeteroscedasticMLP.

    It checks:
    - If the number of linear layers is the expected one.
    """
    num_linear_layers = sum(
        isinstance(layer, nnx.Linear) for _, layer in mock_gauss_het_mlp.iter_modules()
    )
    assert num_linear_layers == mock_gauss_het_mlp.n_hid_layers + 1
