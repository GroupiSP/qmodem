import json
from pathlib import Path

import jax
import jax.numpy as jnp
import lib_eod_simulation as les
import matplotlib.pyplot as plt
import orbax.checkpoint as ocp
from flax import nnx

from qmodem import SIM_CONFIG_FILE_PATH, HeteroscedasticResNet


def main() -> None:
    N_SIMU = 500  # used to produce the RUL statistics

    # Load checkpoint
    ckpt_dir = Path().cwd() / "checkpoints/"
    checkpointer = ocp.StandardCheckpointer()

    abstract_model = nnx.eval_shape(lambda: HeteroscedasticResNet(rngs=nnx.Rngs(0)))
    graphdef, abstract_state = nnx.split(abstract_model)

    state_restored = checkpointer.restore(ckpt_dir / "trained_state", abstract_state)

    model = nnx.merge(graphdef, state_restored)

    # Create a battery simulator.
    with open(SIM_CONFIG_FILE_PATH) as fp:
        sim_config = json.load(fp)

    I_discharge = les.ConstantCurrentDischarge(sim_config["I_const_discharge"])

    sim = les.SimulatorSimple(
        N_SIMU,
        sim_config["v_cut"],
        sim_config["SoC"],
        I_discharge,
        sim_config["model_config"],
    )

    # Run simulations.
    sim.simulate()

    # Get the predicted distribution statistics.
    v_0 = jax.random.choice(key=jax.random.key(0), a=sim.v_memo[0, :], shape=(1, 1))
    preds = model(v_0)
    mu_0, var_0 = preds[0, 0], preds[0, 1]
    sigma_0 = jnp.sqrt(var_0)

    # Plot the simulator and predicted distributions.
    color = plt.cm.rainbow([0, 0.9])

    ts = jnp.arange(len(sim.p_rul)) * sim.batt.dt

    plt.fill_between(
        ts,
        sim.p_rul / sim.batt.dt,
        color=color[0],
        label="True distribution",
        alpha=0.4,
    )

    # ts = jnp.linspace(mu_0 - 4 * sigma_0, mu_0 + 4 * sigma_0, 100)
    dist_pred = (1 / (jnp.sqrt(2 * jnp.pi) * sigma_0)) * jnp.exp(
        -0.5 * ((ts - mu_0) / sigma_0) ** 2
    )
    plt.plot(ts, dist_pred, color=color[1], label="Predicted distribution")

    plt.legend()
    plt.grid()
    plt.xlabel("Time [s]")
    plt.ylabel("P(RUL)")
    plt.title("RUL distributions for SoC=1.")
    plt.show()


if __name__ == "__main__":
    main()
