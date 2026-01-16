import json
from pathlib import Path

import lib_eod_simulation as les
import matplotlib.pyplot as plt
import numpy as np
import orbax.checkpoint as ocp
from flax import nnx

from qmodem import BATT_CONFIG_PATH, HeteroscedasticResNet


def main() -> None:
    # Battery simulator parameters.
    N_SIMU = 100
    CURRENT_AMPLITUDE = -2.8 * 0.75
    V_CUT = 2.5
    OMEGA_STD = 1e-3
    ETA_STD = 1e-2

    # Create battery model.
    with open(BATT_CONFIG_PATH) as fp:
        battery_config = json.load(fp)

    battery = les.BatteryModel(battery_config)

    # Create a current discharge policy.
    discharge_policy = les.ConstantCurrentDischarge(CURRENT_AMPLITUDE)

    # Create a deterministic battery simulator to obtain SoCs for determining
    # the RUL later and discharge voltages to feed the model.
    simulator_det_config = {
        "N_simu": 1,
        "v_cut": V_CUT,
        "SoC_0": 1.0,
        "dt": 100.0,
        "omega_std": 0.0,
        "eta_std": 0.0,
        "I": discharge_policy,
        "battery": battery,
    }

    sim_det = les.SimulatorSimple(simulator_det_config)

    # Simulate and get results.
    sim_det.simulate()

    vs, socs_initial = sim_det.v_memo, sim_det.soc_memo

    # Simulate the battery at different SoCs.
    rul_mean = []
    rul_lower = []  # 95% confidence intervals.
    rul_upper = []

    for soc_0 in socs_initial.flatten():
        sim_config = {
            "N_simu": N_SIMU,
            "v_cut": V_CUT,
            "SoC_0": soc_0,
            "dt": 10.0,
            "omega_std": OMEGA_STD,
            "eta_std": ETA_STD,
            "I": discharge_policy,
            "battery": battery,
        }
        sim = les.SimulatorSimple(sim_config)

        sim.simulate()

        m = les.expected_RUL(sim)
        rul_var = les.variance_RUL(sim)

        rul_mean.append(m)
        rul_lower.append(m - 1.96 * np.sqrt(rul_var))
        rul_upper.append(m + 1.96 * np.sqrt(rul_var))

    # Load (trained) model checkpoint.
    ckpt_dir = Path().cwd() / "checkpoints/"
    checkpointer = ocp.StandardCheckpointer()

    abstract_model = nnx.eval_shape(lambda: HeteroscedasticResNet(rngs=nnx.Rngs(0)))
    graphdef, abstract_state = nnx.split(abstract_model)

    state_restored = checkpointer.restore(ckpt_dir / "trained_state", abstract_state)

    model = nnx.merge(graphdef, state_restored)

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

    plt.plot(ts, rul_pred_mean, color=color[1], alpha=0.4, label="Predicted RUL")
    plt.fill_between(ts, rul_pred_upper, rul_pred_lower, color=color[1], alpha=0.2)

    plt.legend()
    plt.grid()
    plt.xlabel("Time [s]")
    plt.ylabel("P(RUL)")
    plt.title("RUL mean and confidence interval.")
    plt.show()


if __name__ == "__main__":
    main()
