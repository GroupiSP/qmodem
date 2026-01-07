from typing import Sequence, SupportsIndex

import jax
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


class TwoOutputMLP(nnx.Module):
    def __init__(
        self,
        dimensions: Sequence[int],
        act_fn: nnx.Module = nnx.gelu,
        *,
        rngs: nnx.Rngs,
    ):
        """Multi-layer perceptron with two outputs. If the first output is intended as
        the RUL's mean and the second one as the RUL's std, this MLP can model
        heteroscedastic aleatoric uncertainty.

        Args:
            dimensions (Sequence[int]): The dimensions of the layers. The first
                dimension is the number of input features, while the others are
                the number of neurons of the hidden layers. Therefore,
                `N_hidden_layers = len(dimensions) - 1`.
            act_fn (nnx.Module, optional): The activation function. Defaults to nnx.gelu.
            rngs (nnx.Rngs): RNGs for the flax internal modules.
        """
        self.dim_in = dimensions[0]
        self.n_layers = len(dimensions)

        self.layers = [
            nnx.Linear(d_i, d_j, rngs=rngs)
            for d_i, d_j in zip(dimensions[:-1], dimensions[1:])
        ]
        self.output_layer = nnx.Linear(dimensions[-1], 2, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        for layer in self.layers:
            x = self.activation(layer(x))

        return self.output_layer(x)


def main() -> None:
    print("Hello from qmodem!")
