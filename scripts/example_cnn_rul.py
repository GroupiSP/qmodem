"""Plot the deterministic true RUL vs CNN-predicted RUL for a test battery discharge.

Requires a trained CNN checkpoint from `example_cnn_time_window.py`.
"""

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
from qmodem import SimpleCNN1D
from qmodem.data import _back_calculate_rul_linear


def main() -> None:
    # Directories
    _root_dir, CHECKPOINT_DIR, METADATA_DIR = get_run_dirs("cnn_time_window")

    # Must match training parameters
    CURRENT_AMPLITUDE = -2.8 * 0.75
    V_CUT = 2.5

    # Load training y_max for unscaling predictions
    meta = read_json(METADATA_DIR / "y_max.json")
    y_max = meta["y_max"]
    window_size = meta["window_size"]

    battery, discharge_policy = create_battery_and_policy(CURRENT_AMPLITUDE)

    # Run a deterministic simulation (no noise) for the true RUL curve
    sim_config = make_simulator_config(
        n_simu=1,
        v_cut=V_CUT,
        soc_0=1.0,
        dt=10.0,
        omega_std=0.0,
        eta_std=0.0,
        discharge_policy=discharge_policy,
        battery=battery,
    )

    sim_det = les.SimulatorSimple(sim_config)
    sim_det.simulate()

    discharge_voltage = sim_det.v_memo.T[0]  # Shape: (N_t,)
    N_t = len(discharge_voltage)
    t_eod = sim_det.t_eods[0]

    # True RUL: linearly decreasing from t_eod to 0
    true_ruls = np.array(_back_calculate_rul_linear(t_eod=t_eod, N_t=N_t))
    ts = np.arange(N_t) * sim_det.dt

    print(f"Deterministic simulation: N_t={N_t}, t_eod={t_eod:.2f}")
    print(f"Training y_max: {y_max:.2f}")
    print()

    # Load trained CNN model
    print("Loading trained model...")
    model = SimpleCNN1D(
        window_size=window_size, n_filters=4, kernel_size=5, rngs=nnx.Rngs(0)
    )
    restore_model_state(CHECKPOINT_DIR / "trained_state", model)
    print("Model loaded successfully!")

    # Predict RUL for each window position (stride=1 for smooth curve)
    num_windows = N_t - window_size + 1
    predicted_ruls = np.zeros(num_windows)

    for i in range(num_windows):
        window = discharge_voltage[i : i + window_size].reshape(1, 1, -1)
        window_jax = jnp.array(window)
        pred_normalized = model(window_jax)[0]
        predicted_ruls[i] = float(pred_normalized * y_max)

    # Time axis for predictions: each prediction corresponds to the time step
    # at the end of the window
    pred_ts = ts[window_size:]  # Time at index `end` for each window
    # Last window's prediction corresponds to t_eod (RUL=0)
    pred_ts = np.append(pred_ts, ts[N_t - 1])

    print(f"Generated {num_windows} predictions")
    print()

    # Plot
    plt.figure(figsize=(10, 6))

    color = plt.cm.tab10([0, 1])

    plt.plot(ts, true_ruls, color=color[0], linewidth=2, label="True RUL")
    plt.plot(
        pred_ts,
        predicted_ruls,
        color=color[1],
        linewidth=2,
        alpha=0.8,
        label="CNN Predicted RUL",
    )

    plt.xlabel("Time [s]")
    plt.ylabel("RUL [s]")
    plt.title("True vs CNN-Predicted RUL")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
