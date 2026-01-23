import json
from pathlib import Path

import jax
import jax.numpy as jnp
import lib_eod_simulation as les
import matplotlib.pyplot as plt
import orbax.checkpoint as ocp
from flax import nnx

from qmodem import BATT_CONFIG_PATH, HeteroscedasticResNet


def main() -> None:
    # Battery simulator parameters.
    N_SIMU = 500  # used to produce the RUL statistics
    CURRENT_AMPLITUDE = -2.8 * 0.75
    V_CUT = 2.5
    SOC_0 = 1.0
    DT = 10.0
    OMEGA_STD = 1e-3
    ETA_STD = 1e-2

    # Create battery model.
    with open(BATT_CONFIG_PATH) as fp:
        battery_config = json.load(fp)

    battery = les.BatteryModel(battery_config)

    # Create a current discharge policy.
    discharge_policy = les.ConstantCurrentDischarge(CURRENT_AMPLITUDE)

    # Create the battery simulator.
    simulator_config = {
        "N_simu": N_SIMU,
        "v_cut": V_CUT,
        "SoC_0": SOC_0,
        "dt": DT,
        "omega_std": OMEGA_STD,
        "eta_std": ETA_STD,
        "I": discharge_policy,
        "battery": battery,
    }

    sim = les.SimulatorSimple(simulator_config)

    # Load (trained) model checkpoint.
    ckpt_dir = Path().cwd() / "checkpoints/"
    checkpointer = ocp.StandardCheckpointer()

    abstract_model = nnx.eval_shape(lambda: HeteroscedasticResNet(rngs=nnx.Rngs(0)))
    graphdef, abstract_state = nnx.split(abstract_model)

    state_restored = checkpointer.restore(ckpt_dir / "trained_state", abstract_state)

    model = nnx.merge(graphdef, state_restored)

    # Run simulations.
    sim.simulate()

    # Get the predicted distribution statistics.
    v_0 = jax.random.choice(key=jax.random.key(0), a=sim.v_memo[0, :], shape=(1, 1))
    preds = model(v_0)
    mu_0, var_0 = preds[0, 0], preds[0, 1]
    sigma_0 = jnp.sqrt(var_0)

    # Plot the simulator and predicted distributions.
    color = plt.cm.rainbow([0, 0.9])

    ts = jnp.arange(len(sim.p_rul)) * sim.dt

    plt.fill_between(
        ts,
        sim.p_rul / sim.dt,
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
