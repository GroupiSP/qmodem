import json
from pathlib import Path

import jax
import jax.numpy as jnp
import lib_eod_simulation as les
import matplotlib.pyplot as plt
import orbax.checkpoint as ocp
from flax import nnx

from qmodem import BATT_CONFIG_PATH, MCDNetV0
from qmodem.metrics import cdf, crps


def main() -> None:
    # Directories
    ROOT_DIR = Path().cwd() / "saved" / "MLPV0"
    CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
    METADATA_DIR = ROOT_DIR / "metadata"

    # Battery simulator parameters.
    N_SIMU = 1000
    CURRENT_AMPLITUDE = -2.8 * 0.75
    V_CUT = 2.5
    OMEGA_STD = 1e-3
    ETA_STD = 1e-2
    # Number of forward passes for the Monte Carlo Dropout.
    # ENSEMBLE_SIZE = 20

    # For uncscaling we need the max(RUL) found during training, saved as metadata.
    with open(METADATA_DIR / "meta.json", "r") as fp:
        meta = json.load(fp)

    # Create battery model.
    with open(BATT_CONFIG_PATH) as fp:
        battery_config = json.load(fp)

    battery = les.BatteryModel(battery_config)

    # Create a current discharge policy.
    discharge_policy = les.ConstantCurrentDischarge(CURRENT_AMPLITUDE)

    sim_config = {
        "N_simu": N_SIMU,
        "v_cut": V_CUT,
        "SoC_0": 1.0,
        "dt": 10.0,
        "omega_std": OMEGA_STD,
        "eta_std": ETA_STD,
        "I": discharge_policy,
        "battery": battery,
    }
    sim = les.SimulatorSimple(sim_config)

    sim.simulate()

    # Results from the simulator.
    # Get the samples from the reference RUL distribution (t_eods).
    rul_ref_samples = sim.t_eods
    print(f"RUL reference samples shape: {rul_ref_samples.shape}")

    # Load (trained) model checkpoint.
    checkpointer = ocp.StandardCheckpointer()

    model = MCDNetV0(rngs=nnx.Rngs(0))
    target_state = nnx.state(model, nnx.Param)

    state_restored = checkpointer.restore(
        CHECKPOINT_DIR / "trained_state", target=target_state
    )

    nnx.update(model, state_restored)

    # Run one more simulation to get the test data for RUL prediction.
    sim = les.SimulatorSimple(sim_config)
    sim.simulate()
    v0 = sim.v_memo[0, 0]  # voltage at SoC=1.0

    rng_dropout = nnx.Rngs(1)
    forked_rngs_dropout = rng_dropout.fork(
        split=N_SIMU
    )  # same number of predictions as sims

    @nnx.vmap(in_axes=(None, None, 0), out_axes=0)
    def predict_step(model: MCDNetV0, x: jax.Array, rngs: nnx.Rngs) -> jax.Array:
        return model(x, rngs=rngs)

    model.train()  # for MCD
    rul_pred_samples = (
        predict_step(model, jnp.array(v0).reshape(-1, 1), forked_rngs_dropout).ravel()
        * meta["y_max"]
    )
    print(f"RUL predicted samples shape: {rul_pred_samples.shape}")
    print(f"RUL predicted samples (first 10): {rul_pred_samples[:10]}")

    # Plot the histograms of the reference and predicted RUL distributions.
    plt.figure(figsize=(10, 6))
    plt.hist(
        rul_ref_samples,
        bins=100,
        density=True,
        alpha=0.5,
        label="Reference RUL Distribution",
    )
    plt.hist(
        rul_pred_samples,
        bins=100,
        density=True,
        alpha=0.5,
        label="Predicted RUL Distribution",
    )
    plt.xlabel("RUL")
    plt.ylabel("Density")
    plt.title("RUL Distributions")
    plt.legend()

    # Compute and plot the CDFs over a grid.
    x_grid = jnp.linspace(0, max(rul_ref_samples.max(), rul_pred_samples.max()), 1000)
    cdf_ref = jax.vmap(cdf, in_axes=(0, None), out_axes=0)(x_grid, rul_ref_samples)
    cdf_pred = jax.vmap(cdf, in_axes=(0, None), out_axes=0)(x_grid, rul_pred_samples)

    plt.figure(figsize=(10, 6))
    plt.plot(x_grid, cdf_ref, label="Reference RUL CDF")
    plt.plot(x_grid, cdf_pred, label="Predicted RUL CDF")
    plt.xlabel("RUL")
    plt.ylabel("CDF")
    plt.title("RUL CDFs")
    plt.legend()

    # Compute the CRPS between the two distributions.
    crps_value = crps(rul_ref_samples, rul_pred_samples, x_grid)
    print(f"CRPS between reference and predicted RUL distributions: {crps_value}")

    plt.show()


if __name__ == "__main__":
    main()
