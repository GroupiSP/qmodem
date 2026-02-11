from pathlib import Path

import lib_eod_simulation as les
import matplotlib.pyplot as plt
import numpy as np
from flax import nnx

from _shared import (
    create_battery_and_policy,
    make_simulator_config,
    restore_model_from_checkpoint,
)
from qmodem import HNNV1


def main() -> None:
    # Battery simulator parameters.
    N_SIMU = 100
    CURRENT_AMPLITUDE = -2.8 * 0.75
    V_CUT = 2.5
    OMEGA_STD = 1e-3
    ETA_STD = 1e-2

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
    ckpt_dir = Path().cwd() / "checkpoints/"
    model = restore_model_from_checkpoint(
        ckpt_dir / "trained_state",
        lambda: HNNV1(rngs=nnx.Rngs(0)),
    )

    # Get the predicted distribution statistics.
    preds = model(vs)

    rul_pred_mean, rul_pred_var = preds[:, 0], preds[:, 1]
    rul_pred_lower = rul_pred_mean - 1.96 * np.sqrt(rul_pred_var)
    rul_pred_upper = rul_pred_mean + 1.96 * np.sqrt(rul_pred_var)

    # Plot the simulated RULs and 95% confidence intervals.
    plt.figure()

    color = plt.cm.rainbow([0.0, 0.9])

    ts = np.arange(len(socs_initial)) * sim_det.dt
    plt.plot(ts, rul_mean, color=color[0], alpha=0.4, label="True RUL")
    plt.fill_between(ts, rul_upper, rul_lower, color=color[0], alpha=0.2)

    plt.plot(ts, rul_pred_mean, color=color[1], alpha=0.4, label="Predicted RUL (HNN)")
    plt.fill_between(ts, rul_pred_upper, rul_pred_lower, color=color[1], alpha=0.2)

    plt.legend()
    plt.grid()
    plt.xlabel("Time [s]")
    plt.ylabel("P(RUL)")
    plt.title("RUL mean and confidence interval.")
    plt.show()


if __name__ == "__main__":
    main()
