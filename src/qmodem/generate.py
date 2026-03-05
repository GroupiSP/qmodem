"""Data generation utilities for battery discharge datasets.

This module provides functions to generate and persist train and test datasets so that
all models see the same data and generation cost is paid only once.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import lib_eod_simulation as les
import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _run_discharge(config: dict[str, Any], soc_0: float) -> tuple[np.ndarray, float]:
    """Run a single discharge simulation and return (voltage, t_eod)."""
    sim_config = config.copy()
    sim_config["N_simu"] = 1
    sim_config["SoC_0"] = soc_0
    sim = les.SimulatorSimple(sim_config)
    sim.simulate()
    return sim.v_memo.flatten(), float(sim.t_eods[0])


def _run_stochastic_sims(
    config: dict[str, Any], soc_0: float, n_simu: int
) -> np.ndarray:
    """Run *n_simu* stochastic simulations from *soc_0* and return t_eods."""
    sim_config = config.copy()
    sim_config["SoC_0"] = soc_0
    sim_config["N_simu"] = n_simu
    sim = les.SimulatorSimple(sim_config)
    sim.simulate()
    return np.array(sim.t_eods)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_train_data(
    simulator_config: dict[str, Any],
    n_histories: int,
    soc_range: tuple[float, float] = (0.05, 1.0),
    seed: int = 42,
    output_path: Path | str = "data/train.npz",
) -> Path:
    """Generate training data and save to an ``.npz`` file.

    Each history is a single discharge simulation starting from a random SoC₀
    sampled uniformly from *soc_range*.

    The saved archive contains:
    - ``voltages``: list of 1-D voltage arrays (object array, variable length)
    - ``t_eods``: 1-D array of end-of-discharge times, shape ``(n_histories,)``
    - ``soc_0s``: 1-D array of initial SoC values, shape ``(n_histories,)``
    - ``config_json``: JSON string of *simulator_config* (for provenance)

    Args:
        simulator_config: Base simulator configuration dict.
        n_histories: Number of discharge histories to generate.
        soc_range: ``(low, high)`` bounds for uniform SoC₀ sampling.
        seed: Random seed for reproducibility.
        output_path: Destination file path.

    Returns:
        The resolved output path.
    """
    rng = np.random.default_rng(seed)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    voltages: list[np.ndarray] = []
    t_eods: list[float] = []
    soc_0s: list[float] = []

    for _ in range(n_histories):
        soc_0 = float(rng.uniform(*soc_range))
        soc_0s.append(soc_0)
        voltage, t_eod = _run_discharge(simulator_config, soc_0)
        voltages.append(voltage)
        t_eods.append(t_eod)

    # Store variable-length arrays as an object array
    voltages_obj = np.empty(n_histories, dtype=object)
    for i, v in enumerate(voltages):
        voltages_obj[i] = v

    np.savez(
        output_path,
        voltages=voltages_obj,
        t_eods=np.array(t_eods),
        soc_0s=np.array(soc_0s),
        config_json=json.dumps(
            {k: v for k, v in simulator_config.items() if _is_json_serialisable(k, v)}
        ),
    )
    return output_path.resolve()


def generate_test_data(
    simulator_config: dict[str, Any],
    n_test_cases: int = 1,
    n_simu: int = 500,
    n_intermediate_socs: int = 200,
    seed: int = 123,
    output_dir: Path | str = "data",
) -> list[Path]:
    """Generate test data and save each test case to a separate ``.npz`` file.

    For every test case:
    1. A single stochastic discharge trajectory is simulated.
    2. At *n_intermediate_socs* evenly-spaced time indices, *n_simu*
       stochastic simulations are run from the SoC at that point to produce
       reference RUL distributions.

    Each saved archive contains:
    - ``voltage``: 1-D voltage history, shape ``(N_t,)``
    - ``soc_history``: 1-D SoC history, shape ``(N_t,)``
    - ``t_eod``: scalar end-of-discharge time
    - ``dt``: scalar time step
    - ``eval_indices``: 1-D int array of evaluation time indices, shape ``(n_intermediate_socs,)``
    - ``ref_t_eods``: 2-D array of reference t_eods, shape ``(n_intermediate_socs, n_simu)``
    - ``config_json``: JSON string of *simulator_config*

    Args:
        simulator_config: Base simulator configuration dict.
        n_test_cases: Number of independent test cases (discharges).
        n_simu: Stochastic simulations per intermediate SoC for reference
            RUL distributions.
        n_intermediate_socs: Number of evenly-spaced evaluation points along
            each trajectory.
        seed: Random seed for reproducibility.
        output_dir: Directory for output files.

    Returns:
        List of resolved output paths.
    """
    rng = np.random.default_rng(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    dt = simulator_config.get("dt", 1.0)

    for case_idx in range(n_test_cases):
        # Run single stochastic trajectory
        sim_config = simulator_config.copy()
        sim_config["N_simu"] = 1
        case_seed = int(rng.integers(0, 2**31))
        np.random.seed(case_seed)  # seed for lib_eod_simulation
        sim = les.SimulatorSimple(sim_config)
        sim.simulate()

        voltage = sim.v_memo.flatten()
        soc_history = sim.soc_memo.flatten()
        t_eod = float(sim.t_eods[0])
        n_t = len(voltage)

        # Evaluation indices (evenly spaced along trajectory)
        eval_indices = np.linspace(
            0, n_t, n_intermediate_socs, endpoint=False, dtype=np.int32
        )

        # Reference RUL distributions at each eval point
        ref_t_eods = np.empty((n_intermediate_socs, n_simu))
        for k, idx in enumerate(eval_indices):
            soc_0 = float(soc_history[idx])
            ref_t_eods[k] = _run_stochastic_sims(simulator_config, soc_0, n_simu)

        out_path = output_dir / f"test_case_{case_idx}.npz"
        np.savez(
            out_path,
            voltage=voltage,
            soc_history=soc_history,
            t_eod=np.float64(t_eod),
            dt=np.float64(dt),
            eval_indices=eval_indices,
            ref_t_eods=ref_t_eods,
            config_json=json.dumps(
                {
                    k: v
                    for k, v in simulator_config.items()
                    if _is_json_serialisable(k, v)
                }
            ),
        )
        paths.append(out_path.resolve())

    return paths


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _is_json_serialisable(_key: str, value: Any) -> bool:
    """Return True if *value* can be JSON-serialised (skip non-serialisable objects like
    battery models and discharge policies)."""
    try:
        json.dumps(value)
        return True
    except (TypeError, ValueError):
        return False
