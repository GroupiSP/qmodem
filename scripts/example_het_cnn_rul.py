"""Compare heteroscedastic CNN predictions with simulator predictions for battery RUL.

This script:
- Runs a deterministic simulation to get voltage trajectory
- Runs stochastic simulations from SOCs after the first time window
- Uses trained HeteroscedasticCNN1DV1 to predict RUL with uncertainty
- Compares predictions using 95% CI plots and CRPS metric
- Plots CDFs of simulator and CNN predictions
"""

from pathlib import Path

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
    restore_model_from_checkpoint,
)
from qmodem import HeteroscedasticCNN1DV1
from qmodem.metrics import cdf, crps


def main() -> None:
    # Directories
    root_dir, _, metadata_dir = get_run_dirs("het_cnn_time_window", create=False)
    ckpt_dir = root_dir / "checkpoints"

    # Create output directory for plots
    output_dir = Path("saved/het_cnn_rul")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata
    metadata = read_json(metadata_dir / "meta.json")
    y_max = metadata["y_max"]
    WINDOW_SIZE = metadata["window_size"]

    # Battery simulator parameters
    N_SIMU = 100  # Number of stochastic simulations
    CURRENT_AMPLITUDE = -2.8 * 0.75
    V_CUT = 2.5
    OMEGA_STD = 1e-3
    ETA_STD = 1e-2
    DT = 10.0  # Same as training

    battery, discharge_policy = create_battery_and_policy(CURRENT_AMPLITUDE)

    print("=" * 70)
    print("Heteroscedastic CNN RUL Prediction with Uncertainty")
    print("=" * 70)
    print(f"Window size: {WINDOW_SIZE}")
    print(f"dt: {DT}")
    print(f"Number of stochastic simulations: {N_SIMU}")
    print()

    # Create a deterministic battery simulator to obtain SOCs and voltages
    print("Running deterministic simulation...")
    simulator_det_config = make_simulator_config(
        n_simu=1,
        v_cut=V_CUT,
        soc_0=1.0,
        dt=DT,
        omega_std=0.0,
        eta_std=0.0,
        discharge_policy=discharge_policy,
        battery=battery,
    )

    sim_det = les.SimulatorSimple(simulator_det_config)
    sim_det.simulate()

    vs_det = sim_det.v_memo.flatten()  # Voltage trajectory
    socs_det = sim_det.soc_memo.flatten()  # SOC trajectory

    print(f"Deterministic simulation length: {len(vs_det)} time steps")
    print(f"First window endpoint index: {WINDOW_SIZE - 1}")
    print(f"SOC after first window: {socs_det[WINDOW_SIZE]:.4f}")
    print()

    # Load trained model
    print("Loading trained heteroscedastic CNN model...")
    model = restore_model_from_checkpoint(
        ckpt_dir / "trained_state",
        lambda: HeteroscedasticCNN1DV1(
            window_size=WINDOW_SIZE, n_filters=8, kernel_size=5, rngs=nnx.Rngs(0)
        ),
    )
    model.eval()
    print("Model loaded successfully")
    print()

    # Get CNN predictions for all windows
    print("Computing CNN predictions for all time windows...")
    cnn_means = []
    cnn_vars = []
    cnn_prediction_indices = []  # Track which time index each prediction is for

    # Extract non-overlapping windows from deterministic voltages
    # Each window [start, start+WINDOW_SIZE-1] predicts RUL at index (start+WINDOW_SIZE)
    num_windows = (len(vs_det) - WINDOW_SIZE) // WINDOW_SIZE + 1
    for i in range(num_windows):
        start_idx = i * WINDOW_SIZE
        end_idx = start_idx + WINDOW_SIZE
        target_idx = end_idx  # The time step the CNN will predict RUL for

        # Only make prediction if target_idx is within the simulation bounds
        if end_idx > len(vs_det) or target_idx >= len(vs_det):
            break

        # Extract window
        window = vs_det[start_idx:end_idx]
        window_input = jnp.expand_dims(
            jnp.expand_dims(window, 0), 0
        )  # (1, 1, window_size)

        # Get prediction: (mu, var)
        pred = model(window_input)[0]  # Shape: (2,)
        cnn_means.append(pred[0])
        cnn_vars.append(pred[1])
        cnn_prediction_indices.append(target_idx)

    # Add final prediction using a window extended backwards to cover the end-of-discharge
    last_target_idx = len(vs_det) - 1
    if last_target_idx > cnn_prediction_indices[-1]:
        start_idx = last_target_idx - WINDOW_SIZE
        window = vs_det[start_idx:last_target_idx]
        window_input = jnp.expand_dims(jnp.expand_dims(window, 0), 0)
        pred = model(window_input)[0]
        cnn_means.append(pred[0])
        cnn_vars.append(pred[1])
        cnn_prediction_indices.append(last_target_idx)

    cnn_means = jnp.array(cnn_means)
    cnn_vars = jnp.array(cnn_vars)

    # Unscale
    cnn_means_unscaled = cnn_means * y_max
    cnn_vars_unscaled = cnn_vars * (y_max**2)
    cnn_stds_unscaled = jnp.sqrt(cnn_vars_unscaled)

    # Compute 95% CI from Gaussian parameters
    cnn_lower = cnn_means_unscaled - 1.96 * cnn_stds_unscaled
    cnn_upper = cnn_means_unscaled + 1.96 * cnn_stds_unscaled

    print(f"Computed {len(cnn_means)} CNN predictions")
    print(f"CNN prediction indices: {cnn_prediction_indices}")
    print(
        f"Last CNN prediction at index: {cnn_prediction_indices[-1] if cnn_prediction_indices else 'N/A'}"
    )
    print()

    # Run stochastic simulations from SOCs after the first window
    print("Running stochastic simulations from each SOC...")
    sim_means = []
    sim_lower = []
    sim_upper = []
    sim_samples_all = []  # Store all samples for CRPS

    # Simulate from SOCs at window boundaries + the final deterministic EoD point
    sim_soc_indices = list(cnn_prediction_indices)
    last_det_idx = len(socs_det) - 1
    if last_det_idx not in sim_soc_indices:
        sim_soc_indices.append(last_det_idx)

    for idx, soc_idx in enumerate(sim_soc_indices):
        soc_0 = socs_det[soc_idx]

        sim_config = make_simulator_config(
            n_simu=N_SIMU,
            v_cut=V_CUT,
            soc_0=soc_0,
            dt=DT,
            omega_std=OMEGA_STD,
            eta_std=ETA_STD,
            discharge_policy=discharge_policy,
            battery=battery,
        )
        sim = les.SimulatorSimple(sim_config)
        sim.simulate()

        # Get RUL statistics
        rul_samples = sim.t_eods  # Individual RUL samples for each simulation
        m = np.mean(rul_samples)
        std = np.std(rul_samples)

        sim_means.append(m)
        sim_lower.append(m - 1.96 * std)
        sim_upper.append(m + 1.96 * std)
        sim_samples_all.append(rul_samples)

        if (idx + 1) % 5 == 0 or idx == len(sim_soc_indices) - 1:
            print(f"  Completed {idx + 1}/{len(sim_soc_indices)} SOCs")

    sim_means = np.array(sim_means)
    sim_lower = np.array(sim_lower)
    sim_upper = np.array(sim_upper)

    print(f"Completed stochastic simulations for {len(sim_means)} SOCs")
    print()

    # Compute CRPS at the first window endpoint (index 1 in our arrays)
    print("Computing CRPS at SOC after first window...")
    if len(sim_samples_all) > 0 and len(cnn_means_unscaled) > 0:
        # Simulator samples at first prediction point
        sim_samples_first = sim_samples_all[0]

        # Generate CNN samples from predicted Gaussian
        cnn_mean_first = float(cnn_means_unscaled[0])
        cnn_std_first = float(cnn_stds_unscaled[0])

        # Generate samples
        key = jax.random.PRNGKey(42)
        cnn_samples_first = (
            jax.random.normal(key, shape=(N_SIMU,)) * cnn_std_first + cnn_mean_first
        )

        # Create x_grid for CRPS computation
        all_samples = jnp.concatenate([sim_samples_first, cnn_samples_first])
        x_min = float(jnp.min(all_samples))
        x_max = float(jnp.max(all_samples))
        x_grid = jnp.linspace(x_min, x_max, 1000)

        # Compute CRPS
        crps_value = crps(sim_samples_first, cnn_samples_first, x_grid)

        print(f"CRPS at first window endpoint: {crps_value:.4f}")
        print()

    # Plot 1: RUL trajectory with 95% CI
    print("Creating RUL trajectory plot...")
    fig1, ax1 = plt.subplots(figsize=(10, 6))

    # Time points for plotting (separate axes for simulator and CNN)
    time_points_sim = np.array([idx * DT for idx in sim_soc_indices])
    time_points_cnn = np.array([idx * DT for idx in cnn_prediction_indices])

    colors = plt.cm.rainbow([0.0, 0.9])

    # Simulator
    ax1.plot(
        time_points_sim,
        sim_means,
        color=colors[0],
        alpha=0.8,
        label="Simulator RUL",
        linewidth=2,
    )
    ax1.fill_between(
        time_points_sim,
        sim_lower,
        sim_upper,
        color=colors[0],
        alpha=0.2,
        label="Simulator 95% CI",
    )

    # CNN
    ax1.plot(
        time_points_cnn,
        cnn_means_unscaled,
        color=colors[1],
        alpha=0.8,
        label="CNN RUL",
        linewidth=2,
    )
    ax1.fill_between(
        time_points_cnn,
        cnn_lower,
        cnn_upper,
        color=colors[1],
        alpha=0.2,
        label="CNN 95% CI",
    )

    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("RUL [s]")
    ax1.set_title("RUL Prediction: Simulator vs Heteroscedastic CNN")

    # Save and show
    fig1_path = output_dir / "rul_trajectory.png"
    fig1.savefig(fig1_path, dpi=150, bbox_inches="tight")
    print(f"Saved RUL trajectory plot to {fig1_path}")

    # Plot 2: CDF comparison at first window endpoint
    print("Creating CDF comparison plot...")
    if len(sim_samples_all) > 0 and len(cnn_means_unscaled) > 0:
        fig2, ax2 = plt.subplots(figsize=(10, 6))

        # Compute CDFs
        sim_cdf_values = jax.vmap(cdf, in_axes=(0, None), out_axes=0)(
            x_grid, sim_samples_first
        )
        cnn_cdf_values = jax.vmap(cdf, in_axes=(0, None), out_axes=0)(
            x_grid, cnn_samples_first
        )

        # Plot CDFs
        ax2.plot(
            x_grid,
            sim_cdf_values,
            color=colors[0],
            linewidth=2,
            label="Simulator CDF",
            alpha=0.8,
        )
        ax2.plot(
            x_grid,
            cnn_cdf_values,
            color=colors[1],
            linewidth=2,
            label="CNN CDF",
            alpha=0.8,
        )

        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlabel("RUL [s]")
        ax2.set_ylabel("Cumulative Probability")
        ax2.set_title(
            f"CDF Comparison at SOC after First Window (CRPS={crps_value:.4f})"
        )
        ax2.set_ylim([0, 1])

        # Save and show
        fig2_path = output_dir / "cdf_comparison.png"
        fig2.savefig(fig2_path, dpi=150, bbox_inches="tight")
        print(f"Saved CDF comparison plot to {fig2_path}")

    print()
    print("Showing plots...")
    plt.show()
    print()
    print("Done!")


if __name__ == "__main__":
    main()
