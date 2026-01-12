import json
from pathlib import Path
from typing import Optional, Sequence, SupportsIndex

import jax
import jax.numpy as jnp
import lib_eod_simulation as les
import numpy as np
from flax import nnx

_SIM_CONFIG_FILE_PATH = (
    Path(__file__).resolve().parent.parent.parent / "battery_sim_config.json"
)


class BatterySimulationSource:
    def __init__(self, simulator: les.SimulatorSimple | les.SimulatorComplete) -> None:
        """Runs and access to battery simulation data.

        Args:
            simulator (les.SimulatorSimple | les.SimulatorComplete): the simulator from
                lib_eod_simulation. It needs to be configured outside of this data
                source. It must have `N_simu=1` (only deterministic case, work in progress
                to extend).
        """
        simulator.simulate()

        # Transpose for convenience. Shape=(N_simu, N_t).
        discharge_voltage_per_sim: np.ndarray = simulator.v_memo.T
        N_t = discharge_voltage_per_sim.shape[1]

        self.discharge_voltage = jnp.array(
            discharge_voltage_per_sim.flatten().reshape(-1, 1)
        )
        ruls = np.empty(shape=(simulator.N_simu * N_t))

        for i in range(simulator.N_simu):
            ruls[i * N_t : (i + 1) * N_t] = np.clip(
                simulator.t_eods[i] - np.arange(N_t) * simulator.batt.dt,
                a_min=0.0,
                a_max=None,
            )  # clipping ensures that the failed particles have RUL=0. after their time of failure

        self.ruls = jnp.array(ruls)

    def __len__(self) -> int:
        """Number of records in the dataset."""
        return len(self.ruls)

    def __getitem__(self, record_key: SupportsIndex) -> tuple[jax.Array, float]:
        """Retrieves record for the given record_key."""
        return self.discharge_voltage[record_key], self.ruls[record_key]


def make_battery_data(N_simu: int = 1) -> BatterySimulationSource:
    """Makes the Grain data source for the battery simulator. Assumes a constant current
    policy.

    Args:
        N_simu (int, optional): Number of MC simulations of the battery discharge. Defaults to 1.

    Returns:
        BatterySimulationSource: the Grain battery data-source.
    """
    with open(_SIM_CONFIG_FILE_PATH) as fp:
        sim_config = json.load(fp)

    I_discharge = les.ConstantCurrentDischarge(sim_config["I_const_discharge"])

    sim = les.SimulatorSimple(
        N_simu,
        sim_config["v_cut"],
        sim_config["SoC"],
        I_discharge,
        sim_config["model_config"],
    )

    return sim, BatterySimulationSource(sim)


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
        self.act_fn = act_fn

        self.layers = nnx.List(
            [
                nnx.Linear(d_i, d_j, rngs=rngs)
                for d_i, d_j in zip(dimensions[:-1], dimensions[1:])
            ]
        )

        # the output layer
        self.layers.append(nnx.Linear(dimensions[-1], 2, rngs=rngs))

    def __call__(self, x: jax.Array, rngs: Optional[nnx.Rngs] = None) -> jax.Array:
        for layer in self.layers[:-1]:
            x = self.act_fn(layer(x))

        # apply the output layer w/o activation function
        x = self.layers[-1](x)
        return jnp.stack([x[:, 0], nnx.softplus(x[:, 1])], axis=-1)


def nll_loss(
    model: nnx.Module, batch: jax.Array, rngs: Optional[nnx.Rngs] = None
) -> jax.Array:
    """Negative log-liklihood loss, based on a Gaussian predictive distribution of the model.
    It implements Equation (31) in https://doi.org/10.1016/j.ymssp.2023.110796.

    Args:
        model (nnx.Module): Gaussian neural network with 2 outputs (mean and variance).
        batch (jax.Array): batched input data.
        rngs (nnx.Rngs): passed to the forward method of the model.

    Returns:
        jax.Array: loss value for the batch.
    """

    xs, labels = batch
    outputs = model(xs, rngs=rngs)
    means, variances = outputs[:, 0], outputs[:, 1]
    losses = 0.5 * jnp.log(variances) + 0.5 * jnp.square(labels - means) / variances

    return jnp.mean(losses)


def main() -> None:
    print("Hello from qmodem!")
