from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping

import lib_eod_simulation as les
import orbax.checkpoint as ocp
from flax import nnx

from qmodem import BATT_CONFIG_PATH
from qmodem.utils import mkdir_if_not_existent

# ---------------------------------------------------------------------------
# Shared seeds
# ---------------------------------------------------------------------------
TRAIN_SEED: int = 42
TEST_SEED: int = 123

# ---------------------------------------------------------------------------
# Shared parameters
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
}


def read_json(path: Path) -> dict[str, Any]:
    """Read JSON data from a file."""
    with open(path, "r") as fp:
        return json.load(fp)


def write_json(path: Path, data: Mapping[str, Any]) -> None:
    """Write JSON data to a file."""
    with open(path, "w") as fp:
        json.dump(data, fp)


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


# TODO - remove
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


def get_run_dirs(run_name: str, create: bool = False) -> tuple[Path, Path, Path]:
    """Return root, checkpoint, and metadata directories for a run."""
    root_dir = Path().cwd() / "saved" / run_name
    checkpoint_dir = root_dir / "checkpoints"
    metadata_dir = root_dir / "metadata"
    if create:
        mkdir_if_not_existent([checkpoint_dir, metadata_dir])
    return root_dir, checkpoint_dir, metadata_dir


def restore_model_state(checkpoint_path: Path, model: nnx.Module) -> None:
    """Restore a model's parameters from a checkpoint path."""
    checkpointer = ocp.StandardCheckpointer()
    target_state = nnx.state(model, nnx.Param)
    state_restored = checkpointer.restore(checkpoint_path, target=target_state)
    nnx.update(model, state_restored)


def restore_model_from_checkpoint(
    checkpoint_path: Path, model_factory: Callable[[], nnx.Module]
) -> nnx.Module:
    """Restore a model using eval-shape and a checkpoint path."""
    checkpointer = ocp.StandardCheckpointer()
    abstract_model = nnx.eval_shape(model_factory)
    graphdef, abstract_state = nnx.split(abstract_model)
    state_restored = checkpointer.restore(checkpoint_path, abstract_state)
    return nnx.merge(graphdef, state_restored)
