from typing import Sequence, SupportsIndex

import jax
import jax.numpy as jnp
import lib_eod_simulation as les
import numpy as np
from flax import nnx


class BatterySimulationSingleTimeSource:
    def __init__(self, simulator: les.SimulatorSimple | les.SimulatorComplete) -> None:
        """Runs and access to battery simulation data.

        Args:
            simulator (les.SimulatorSimple | les.SimulatorComplete): the simulator from
                lib_eod_simulation. It needs to be configured outside of this data
                source. It must have `N_simu=1` (only deterministic case, work in progress
                to extend).
        """
        simulator.simulate()

        self.discharge_voltage = simulator.v_memo.ravel()
        self.times = np.arange(len(self.discharge_voltage)) * simulator.batt.dt
        self.ruls = self.times[-1] - self.times

    def __len__(self) -> int:
        """Number of records in the dataset."""
        return len(self.discharge_voltage)

    def __getitem__(self, record_key: SupportsIndex) -> np.ndarray:
        """Retrieves record for the given record_key."""
        return self.discharge_voltage[record_key], self.ruls[record_key]


class GaussianHeteroscedasticMLP(nnx.Module):
    def __init__(
        self,
        dimensions: Sequence[int],
        act_fn: nnx.Module = nnx.gelu,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """Multi-layer perceptron. The first output is intended as the predicted mean
        and the second one as the predicted variance, turned positive by a softplus
        activation. The uncertainty is heteroscedastic, because the variance is also a
        function of the input vector.

        Args:
            dimensions (Sequence[int]): The dimensions of the layers. The first
                dimension is the number of input features, while the others are
                the number of neurons of the hidden layers. Therefore,
                `N_hidden_layers = len(dimensions) - 1`.
            act_fn (nnx.Module, optional): The activation function. Defaults to nnx.gelu.
            rngs (nnx.Rngs): RNGs for the flax internal modules.
        """
        self.dim_in = dimensions[0]
        self.n_hid_layers = len(dimensions) - 1

        self.layers = nnx.List(
            [
                nnx.Linear(d_i, d_j, rngs=rngs)
                for d_i, d_j in zip(dimensions[:-1], dimensions[1:])
            ]
        )
        self.output_layer = nnx.Linear(dimensions[-1], 2, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        for layer in self.layers:
            x = self.activation(layer(x))

        x = self.output_layer(x)
        return jnp.array([x[:, 0], nnx.softplus(x[:, 1])])


# def nll_loss(params: optax.Params, batch: jax.Array, model: nnx.Module) -> jax.Array:
#     xs, labels = batch
#     outputs = model(xs)
#     means, stds = outputs[:, 0], outputs[:, 1]


def main() -> None:
    print("Hello from qmodem!")
