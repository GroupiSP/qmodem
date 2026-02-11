import jax
import jax.numpy as jnp
import lib_eod_simulation as les
import matplotlib.pyplot as plt
import numpy as np
from flax import nnx

from _shared import (
    create_battery_and_policy,
    get_run_dirs,
    make_simulator_config,
    read_json,
    restore_model_state,
)
from qmodem import MCDNetV0


def main() -> None:
    # Directories
    _root_dir, CHECKPOINT_DIR, METADATA_DIR = get_run_dirs("MLPV0")

    # Battery simulator parameters.
    N_SIMU = 100
    CURRENT_AMPLITUDE = -2.8 * 0.75
    V_CUT = 2.5
    OMEGA_STD = 1e-3
    ETA_STD = 1e-2
    # Number of forward passes for the Monte Carlo Dropout.
    ENSEMBLE_SIZE = 20

    # For uncscaling we need the max(RUL) found during training, saved as metadata.
    meta = read_json(METADATA_DIR / "meta.json")
    battery, discharge_policy = create_battery_and_policy(CURRENT_AMPLITUDE)

    # Create a deterministic battery simulator to obtain SoCs for determining
    # the RUL later and discharge voltages to feed the model.
    simulator_det_config = make_simulator_config(
        n_simu=1,
        v_cut=V_CUT,
        soc_0=1.0,
        dt=100.0,
        omega_std=0.0,
        eta_std=0.0,
        discharge_policy=discharge_policy,
        battery=battery,
    )

    sim_det = les.SimulatorSimple(simulator_det_config)

    # Simulate and get results.
    sim_det.simulate()

    vs, socs_initial = sim_det.v_memo, sim_det.soc_memo

    # Simulate the battery at different SoCs.
    rul_mean = []
    rul_lower = []  # 95% confidence intervals.
    rul_upper = []

    for soc_0 in socs_initial.flatten():
        sim_config = make_simulator_config(
            n_simu=N_SIMU,
            v_cut=V_CUT,
            soc_0=soc_0,
            dt=10.0,
            omega_std=OMEGA_STD,
            eta_std=ETA_STD,
            discharge_policy=discharge_policy,
            battery=battery,
        )
        sim = les.SimulatorSimple(sim_config)

        sim.simulate()

        m = les.expected_RUL(sim)
        rul_var = les.variance_RUL(sim)

        rul_mean.append(m)
        rul_lower.append(m - 1.96 * np.sqrt(rul_var))
        rul_upper.append(m + 1.96 * np.sqrt(rul_var))

    # Load (trained) model checkpoint.
    model = MCDNetV0(rngs=nnx.Rngs(0))
    restore_model_state(CHECKPOINT_DIR / "trained_state", model)

    rng_dropout = nnx.Rngs(1)

    @nnx.jit
    def predict_step(model: MCDNetV0, x: jax.Array, rngs: nnx.Rngs) -> jax.Array:
        return model(x, rngs=rngs)

    rul_mean_nn = jnp.empty(shape=len(vs))
    rul_std_nn = jnp.empty(shape=len(vs))

    model.train()  # for MCD
    for i, v in enumerate(vs):
        predictions = jnp.array(
            [
                predict_step(model, v, rng_dropout).squeeze() * meta["y_max"]
                for _ in range(ENSEMBLE_SIZE)
            ]
        )
        rul_mean_nn = rul_mean_nn.at[i].set(jnp.mean(predictions).squeeze())
        rul_std_nn = rul_std_nn.at[i].set(jnp.std(predictions).squeeze())

    # print(rul_mean_nn)
    # print(rul_std_nn)

    rul_pred_lower = rul_mean_nn - 1.96 * rul_std_nn
    rul_pred_upper = rul_mean_nn + 1.96 * rul_std_nn

    # Plot the simulated RULs and 95% confidence intervals.
    plt.figure()

    color = plt.cm.rainbow([0.0, 0.9])

    ts = np.arange(len(socs_initial)) * sim_det.dt
    plt.plot(ts, rul_mean, color=color[0], alpha=0.4, label="True RUL")
    plt.fill_between(ts, rul_upper, rul_lower, color=color[0], alpha=0.2)

    plt.plot(
        ts,
        rul_mean_nn.flatten(),
        color=color[1],
        alpha=0.4,
        label="Predicted RUL (MCD)",
    )
    plt.fill_between(
        ts,
        rul_pred_upper.flatten(),
        rul_pred_lower.flatten(),
        color=color[1],
        alpha=0.2,
    )

    plt.legend()
    plt.grid()
    plt.xlabel("Time [s]")
    plt.ylabel("P(RUL)")
    plt.title("RUL mean and confidence interval.")
    plt.show()


if __name__ == "__main__":
    main()
