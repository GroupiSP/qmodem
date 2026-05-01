from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import lib_eod_simulation as les
import orbax.checkpoint as ocp
from flax import nnx
from jax.typing import ArrayLike

# ---------------------------------------------------------------------------
# Shared paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
BATT_CONFIG_PATH = ROOT_DIR / "battery_config.json"
CMAPSS_DIR_PATH = ROOT_DIR / "data" / "raw" / "CMAPSSData"
PROCESSED_DATA_DIR_PATH = ROOT_DIR / "data" / "processed"

# ---------------------------------------------------------------------------
# Shared seeds
# ---------------------------------------------------------------------------
TRAIN_SEED: int = 42
TEST_SEED: int = 123

# ---------------------------------------------------------------------------
# Shared default parameters
# ---------------------------------------------------------------------------
SHARED_PARAMS: dict[str, Any] = {
    "training": {
        "lr": 1e-2,
        "n_epochs": 500,
        "batch_size": 32,
        "patience": 50,
        "print_every": 10,
    },
    "model": {
        "n_filters": 4,
        "kernel_size": 5,
    },
    "data": {
        "n_histories_train": 100,
        "n_histories_val": 20,
        "window_size": 20,
        "stride": 10,
        "normalize": True,
    },
    "simulation": {
        "current_amplitude": -2.8 * 0.75,
        "v_cut": 2.5,
        "dt": 20.0,
        "omega_std": 3e-3,
        "eta_std": 0.0,
        "soc_range": (0.05, 1.0),
    },
    "mcd_cnn": {
        "dropout_rate": 0.1,
    },
    "qavi_cnn": {
        "n_qubits": 6,
        "n_pqc_layers": 1,
    },
    "generate": {
        "n_test_cases": 1,
        "n_simu": 500,
        "n_intermediate_socs": 200,
    },
    "test": {
        "n_samples": 500,
    },
}


# ---------------------------------------------------------------------------
# Filesystem utilities
# ---------------------------------------------------------------------------


def mkdir_if_not_existent(paths: list[str | Path]) -> None:
    """Iterates through a list of paths and creates the directories if they do not
    already exist.

    Args:
        paths: A list of directory paths. Accepts both strings and Path objects.
    """
    for path in paths:
        try:
            dir_path = Path(path)
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"Checked/Created: {dir_path.resolve()}")
        except OSError as e:
            print(f"Error creating directory '{path}': {e}")
            raise


def read_json(path: Path) -> dict[str, Any]:
    """Read JSON data from a file."""
    with open(path, "r") as fp:
        return json.load(fp)


def write_json(path: Path, data: Mapping[str, Any]) -> None:
    """Write JSON data to a file."""
    with open(path, "w") as fp:
        json.dump(data, fp)


# ---------------------------------------------------------------------------
# Run directory management
# ---------------------------------------------------------------------------


def get_run_dirs(
    run_name: str, create: bool = False, base_dir: str = "saved"
) -> tuple[Path, Path, Path]:
    """Return root, checkpoint, and metadata directories for a run.

    Args:
        run_name: Name of the run (e.g. ``"het_cnn/train"``).
        create: If ``True``, create directories if they don't exist.
        base_dir: Base directory under CWD for run artefacts. Defaults to
            ``"saved"``.

    Returns:
        Tuple of ``(root_dir, checkpoint_dir, metadata_dir)``.
    """
    root_dir = Path().cwd() / base_dir / run_name
    checkpoint_dir = root_dir / "checkpoints"
    metadata_dir = root_dir / "metadata"
    if create:
        mkdir_if_not_existent([checkpoint_dir, metadata_dir])
    return root_dir, checkpoint_dir, metadata_dir


# ---------------------------------------------------------------------------
# Battery / simulation helpers
# ---------------------------------------------------------------------------


def load_battery_config() -> dict[str, Any]:
    """Load the battery configuration file."""
    return read_json(BATT_CONFIG_PATH)


def create_battery_and_policy(
    current_amplitude: float,
) -> tuple[les.BatteryModel, les.ConstantCurrentDischarge]:
    """Create a battery model and constant-current discharge policy."""
    battery = les.BatteryModel(load_battery_config())
    discharge_policy = les.ConstantCurrentDischarge(current_amplitude)
    return battery, discharge_policy


def make_simulator_config(
    *,
    n_simu: int,
    v_cut: float,
    soc_0: float,
    dt: float,
    omega_std: float,
    eta_std: float,
    discharge_policy: les.ConstantCurrentDischarge,
    battery: les.BatteryModel,
) -> dict[str, Any]:
    """Build a simulator configuration dictionary."""
    return {
        "N_simu": n_simu,
        "v_cut": v_cut,
        "SoC_0": soc_0,
        "dt": dt,
        "omega_std": omega_std,
        "eta_std": eta_std,
        "I": discharge_policy,
        "battery": battery,
    }


# ---------------------------------------------------------------------------
# Model checkpoint utilities
# ---------------------------------------------------------------------------


def restore_model_state(checkpoint_path: Path, model: nnx.Module) -> None:
    """Restore a model's parameters from a checkpoint path.

    Restores only ``nnx.Param`` leaves — suitable for models containing
    non-serialisable state (e.g. ``nnx.Dropout`` with PRNGKey).

    Args:
        checkpoint_path: Path to the Orbax checkpoint directory.
        model: An initialised model whose parameters will be updated in-place.
    """
    checkpointer = ocp.StandardCheckpointer()
    target_state = nnx.state(model, nnx.Param)
    state_restored = checkpointer.restore(checkpoint_path, target=target_state)
    nnx.update(model, state_restored)


def restore_model_from_checkpoint(
    checkpoint_path: Path, model_factory: Callable[[], nnx.Module]
) -> nnx.Module:
    """Restore a model using eval-shape and a checkpoint path.

    Uses ``nnx.eval_shape`` to create an abstract model without allocating
    arrays, then restores full state from disk.

    Args:
        checkpoint_path: Path to the Orbax checkpoint directory.
        model_factory: Zero-argument callable that returns a new model instance.

    Returns:
        The restored model with checkpoint weights.
    """
    checkpointer = ocp.StandardCheckpointer()
    abstract_model = nnx.eval_shape(model_factory)
    graphdef, abstract_state = nnx.split(abstract_model)
    state_restored = checkpointer.restore(checkpoint_path, abstract_state)
    return nnx.merge(graphdef, state_restored)


# ---------------------------------------------------------------------------
# JAX (AI) utilities
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Statistics:
    mean: jax.Array
    std: jax.Array
    p_025: jax.Array
    p_975: jax.Array
    count: int


def get_statistics(arr: jax.Array, dim: int) -> Statistics:
    """Compute mean, std, 2.5th and 97.5th percentiles along a specified dimension."""
    mean = jnp.mean(arr, axis=dim)
    std = jnp.std(arr, axis=dim)
    p_025 = jnp.percentile(arr, 2.5, axis=dim)
    p_975 = jnp.percentile(arr, 97.5, axis=dim)
    count = arr.shape[dim]
    return Statistics(mean=mean, std=std, p_025=p_025, p_975=p_975, count=count)


def states_equal(s1: ArrayLike, s2: ArrayLike) -> bool:
    """Checks if two JAX pytrees (eg model parameters) are equal.

    Args:
        s1 (ArrayLike): First pytree to compare.
        s2 (ArrayLike): Second pytree to compare.

    Returns:
        bool: True if the pytrees are equal, False otherwise.
    """
    leaves_equal = jax.tree.leaves(jax.tree.map(jnp.array_equal, s1, s2))
    return all(leaves_equal)
