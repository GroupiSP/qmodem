import jax
import jax.numpy as jnp
import lib_eod_simulation as les
import matplotlib.pyplot as plt
from flax import nnx

from _shared import (
    create_battery_and_policy,
    get_run_dirs,
    make_simulator_config,
    read_json,
    restore_model_state,
)
from qmodem import MCDNetV0
from qmodem.metrics import cdf, crps


def main() -> None:
    # Directories
    _root_dir, CHECKPOINT_DIR, METADATA_DIR = get_run_dirs("MLPV0")

    # Battery simulator parameters.
    N_SIMU = 1000
    CURRENT_AMPLITUDE = -2.8 * 0.75
    V_CUT = 2.5
    OMEGA_STD = 1e-3
    ETA_STD = 1e-2
    # Number of forward passes for the Monte Carlo Dropout.
    # ENSEMBLE_SIZE = 20

    # For uncscaling we need the max(RUL) found during training, saved as metadata.
    meta = read_json(METADATA_DIR / "meta.json")
    battery, discharge_policy = create_battery_and_policy(CURRENT_AMPLITUDE)

    sim_config = make_simulator_config(
        n_simu=N_SIMU,
        v_cut=V_CUT,
        soc_0=1.0,
        dt=10.0,
        omega_std=OMEGA_STD,
        eta_std=ETA_STD,
        discharge_policy=discharge_policy,
        battery=battery,
    )
    sim = les.SimulatorSimple(sim_config)

    sim.simulate()

    # Results from the simulator.
    # Get the samples from the reference RUL distribution (t_eods).
    rul_ref_samples = sim.t_eods
    print(f"RUL reference samples shape: {rul_ref_samples.shape}")

    # Load (trained) model checkpoint.
    model = MCDNetV0(rngs=nnx.Rngs(0))
    restore_model_state(CHECKPOINT_DIR / "trained_state", model)

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
