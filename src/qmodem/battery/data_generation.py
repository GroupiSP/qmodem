from __future__ import annotations

import copy
import dataclasses
import functools
import pathlib
import pickle
import tempfile
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum, auto
from typing import Any, Protocol

import jaxtyping as jtp
import mlflow
import numpy as np
import pandas as pd
import simbat as sb


class VOCModel(StrEnum):
    BUSTOS_BAEZA = auto()


class ECMModel(StrEnum):
    THEVENIN_ZERO_ORDER = auto()


type TestCaseRULSamples = jtp.Float[np.ndarray, "n_checkpoints n_mc_samples"]
# Keys follow  the pattern "test_case_{test_case_id}"
type TestRULSamples = dict[str, TestCaseRULSamples]


class EventFunction(Protocol):
    def __call__(self, t0: float, soc0: float) -> bool: ...


class SimulatorUpdater(Protocol):
    def __call__(
        self,
        base_config: sb.SimulationConfig,
        idx: int,
        soc0: float,
        t0: float,
        tc_data: pd.DataFrame,
        event_fn: EventFunction,
    ) -> sb.SimulationConfig: ...


class ConfigFactory(Protocol):
    """Builds a simulation config whose noise callables draw from ``rng``."""

    def __call__(self, rng: np.random.Generator) -> sb.SimulationConfig: ...


TRAIN_STREAM_SPLIT_NAMES: tuple[str, str, str] = (
    "train",
    "backbone_val",
    "posterior_val",
)
"""Splits of ``train.csv``, in ``run_id`` order."""

TEST_SPLIT_NAME = "test"
"""The split of ``test.csv``."""


def n_histories_param(split: str) -> str:
    """Name of a split's history count, as a :class:`Hyperparameters` field and param.

    Args:
        split: A split name, e.g. ``"backbone_val"``.

    Returns:
        ``"n_histories_<split>"``.
    """
    return f"n_histories_{split}"


def n_policy_1_param(split: str) -> str:
    """Name of a split's policy-1 count, as a :class:`Hyperparameters` field and param.

    Args:
        split: A split name, e.g. ``"backbone_val"``.

    Returns:
        ``"n_policy_1_<split>"``.
    """
    return f"n_policy_1_{split}"


@dataclass(frozen=True)
class Hyperparameters:
    voc_model: VOCModel = VOCModel.BUSTOS_BAEZA
    ecm_model: ECMModel = ECMModel.THEVENIN_ZERO_ORDER
    ecm_model_params: dict[str, float] = dataclasses.field(
        default_factory=lambda: {"r0": 0.1},
    )
    battery_nominal_capacity: float = 10080.0  # in Coulombs
    dt: float = 5.0  # in seconds
    v_cutoff: float = 2.5  # in Volts
    n_histories_train: int = 10
    n_policy_1_train: int = 3
    n_histories_backbone_val: int = 20
    n_policy_1_backbone_val: int = 6
    n_histories_posterior_val: int = 20
    n_policy_1_posterior_val: int = 6
    n_histories_test: int = 20
    n_policy_1_test: int = 6
    process_noise_std: float = 2e-3
    measurement_noise_std: float = 5e-3
    train_seed: int = 42
    test_seed: int = 123

    def __post_init__(self) -> None:
        """Validate the hyperparameters on construction.

        Raises:
            ValueError: If a split's counts are invalid (see :meth:`_validate`).
        """
        self._validate()

    def _validate(self) -> None:
        """Check that every split's counts form a valid :class:`SplitSpec`.

        Raises:
            ValueError: If a count is negative or a split's policy-1 count exceeds
                its size.
        """
        for split in (*TRAIN_STREAM_SPLIT_NAMES, TEST_SPLIT_NAME):
            self.split(split)

    def split(self, name: str) -> SplitSpec:
        """Size and policy mix of one named split.

        Args:
            name: A split name (see :data:`TRAIN_STREAM_SPLIT_NAMES`,
                :data:`TEST_SPLIT_NAME`).

        Returns:
            The split.

        Raises:
            AttributeError: If ``name`` is not a split.
            ValueError: If the split's counts are invalid.
        """
        return SplitSpec(
            getattr(self, n_histories_param(name)),
            getattr(self, n_policy_1_param(name)),
        )

    def train_stream_splits(self) -> tuple[SplitSpec, ...]:
        """Splits of ``train.csv``, in ``run_id`` order (seeded by ``train_seed``).

        Returns:
            The train, backbone validation set and posterior validation set splits.
        """
        return tuple(self.split(name) for name in TRAIN_STREAM_SPLIT_NAMES)

    def test_split(self) -> SplitSpec:
        """Split of ``test.csv`` (seeded by ``test_seed``).

        Returns:
            The test split.
        """
        return self.split(TEST_SPLIT_NAME)


def gaussian_noise(*, rng: np.random.Generator, noise_std: float) -> float:
    """Draw a zero-mean Gaussian noise sample.

    Module-level (picklable) replacement for lambda-based noise distributions.
    """
    return rng.normal(loc=0.0, scale=noise_std)


def always_first_policy() -> int:
    """Deterministically select the first current policy.

    Picklable replacement for the
    default lambda of ``SimulationConfig.policy_choice_distribution``.
    """
    return 0


def bernoulli_policy_choice(*, rng: np.random.Generator, p: tuple[float, float]) -> int:
    """Stochastically select between the first two current policies.

    Picklable replacement for lambda-based policy choice.
    """
    return int(rng.choice([0, 1], p=p))


def log_simulation_config(
    config: sb.SimulationConfig, artifact_file: str = "simulation_config.pkl"
) -> None:
    """Pickle a :class:`simbat.SimulationConfig` and log it as an MLflow artifact.

    The config must be free of lambdas/closures so that it is picklable with the
    standard library ``pickle`` (all callables should be module-level functions,
    ``functools.partial`` of module-level functions, or picklable class instances).
    """
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / artifact_file
        path.write_bytes(pickle.dumps(config))
        mlflow.log_artifact(str(path))


def load_simulation_config(
    run_id: str, artifact_file: str = "simulation_config.pkl"
) -> sb.SimulationConfig:
    """Download and unpickle a :class:`simbat.SimulationConfig` logged with
    :func:`log_simulation_config` from the given MLflow run."""
    path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path=artifact_file
    )
    return pickle.loads(pathlib.Path(path).read_bytes())


def _modify_dataframe(df: pd.DataFrame, run_id: int) -> None:
    """Modifies the datafram in order to result in the following schema:
    run_id, policy_id, time, load, soc, voltage"""
    df.drop(
        columns=["rul_probability", "eod_reached_sim_0"], inplace=True
    )  # Drop the RUL probability column
    df.rename(
        columns={
            "policy_id_0": "policy_id",
            "time": "time",
            "load_sim_0": "load",
            "soc_sim_0": "soc",
            "voltage_sim_0": "voltage",
        },
        inplace=True,
    )
    df.insert(0, "run_id", run_id)  # Add a run_id column for tracking


@dataclass(frozen=True)
class SplitSpec:
    """Size and current-policy mix of one split of generated histories.

    Attributes:
        n_histories: Histories in the split.
        n_policy_1: Histories that follow policy 1; the other
            ``n_histories - n_policy_1`` follow policy 0.
    """

    n_histories: int
    n_policy_1: int

    def __post_init__(self) -> None:
        """Validate the counts.

        Raises:
            ValueError: If a count is negative or ``n_policy_1 > n_histories``.
        """
        if self.n_histories < 0 or self.n_policy_1 < 0:
            raise ValueError(f"Split counts must be non-negative, got {self}.")
        if self.n_policy_1 > self.n_histories:
            raise ValueError(
                f"n_policy_1={self.n_policy_1} exceeds n_histories={self.n_histories}."
            )


def fixed_policy_choice(*, policy_id: int) -> int:
    """Deterministically select the given current policy.

    Picklable policy choice used to simulate a history with an assigned policy.

    Args:
        policy_id: Index of the current policy to select.

    Returns:
        ``policy_id``.
    """
    return policy_id


def assign_policies(split: SplitSpec, rng: np.random.Generator) -> np.ndarray:
    """Assign exactly the split's policy counts to its histories, in a random order.

    Args:
        split: Size and policy-1 count of the split.
        rng: Generator that permutes the assignment.

    Returns:
        ``(n_histories,)`` integer array with ``n_policy_1`` ones and zeros elsewhere.
    """
    policies = np.zeros(split.n_histories, dtype=np.int64)
    policies[: split.n_policy_1] = 1
    return rng.permutation(policies)


def write_split_histories(
    make_config: ConfigFactory,
    splits: Sequence[SplitSpec],
    seed: int,
) -> pd.DataFrame:
    """Simulate the histories of consecutive splits with fixed policy counts.

    Split ``i`` draws from its own generator, seeded by the ``i``-th child of
    ``SeedSequence(seed)``: it permutes the split's policies (see
    :func:`assign_policies`) and feeds the noise of the split's config, built by
    ``make_config``. A split's histories therefore depend only on ``seed``, its
    position and its own counts, not on the other splits' counts. Every history is
    simulated with its assigned policy; the config's policy choice is not used. The
    splits occupy contiguous ``run_id`` blocks in the given order, starting at 0.

    Args:
        make_config: Builds a split's base simulation config from its generator.
        splits: Splits to generate, in ``run_id`` order.
        seed: Root seed of the splits' generators.

    Returns:
        Histories with columns ``run_id, time, policy_id, load, soc, voltage``.
    """
    frames = []
    run_id = 0
    children = np.random.SeedSequence(seed).spawn(len(splits))
    for split, child in zip(splits, children, strict=True):
        rng = np.random.default_rng(child)
        config = make_config(rng)
        for policy_id in assign_policies(split, rng):
            history_config = dataclasses.replace(
                config,
                policy_choice_distribution=functools.partial(
                    fixed_policy_choice, policy_id=int(policy_id)
                ),
            )
            df = sb.simulate_constant_capacity_simple(
                n_sim=1, config=history_config
            ).to_dataframe()
            _modify_dataframe(df, run_id)
            frames.append(df)
            run_id += 1
    if not frames:
        return pd.DataFrame(
            columns=["run_id", "time", "policy_id", "load", "soc", "voltage"]
        )
    return pd.concat(frames, ignore_index=True)


def add_rul_series(df: pd.DataFrame) -> pd.DataFrame:
    """Add a RUL series to the dataframe.

    Args:
        df: A dataframe with columns "run_id" and "time".

    Returns:
        A new dataframe with an additional column "rul" containing the remaining
        useful life for each time step.
    """
    df = df.copy()
    df["rul"] = 0.0  # Initialize the RUL column

    for run_id, group in df.groupby("run_id"):
        max_time = group["time"].max()
        df.loc[group.index, "rul"] = max_time - group["time"]

    return df


def run_discharges_from_intermediate_socs(
    config: sb.SimulationConfig,
    overrides: Iterable[Mapping[str, Any]],
    n_sim: int = 100,
) -> Iterator[sb.SimulationResult]:
    """Run Monte Carlo discharge simulations from a set of intermediate starting states.

    Args:
        config: Base simulation config (e.g. loaded from the data-generation run).
        overrides: One mapping of ``SimulationConfig`` field overrides per starting
            point, applied via :func:`dataclasses.replace` (typically ``soc_0`` and
            ``t_0``).
        n_sim: Number of Monte Carlo particles per discharge.

    Yields:
        One :class:`simbat.SimulationResult` per override.
    """
    for override in overrides:
        cfg = dataclasses.replace(config, **override)
        yield sb.simulate_constant_capacity_simple(n_sim=n_sim, config=cfg)


def sim_updater_two_scenarios(
    base_config: sb.SimulationConfig,
    idx: int,
    soc0: float,
    t0: float,
    tc_data: pd.DataFrame,
    event_fn: EventFunction,
) -> sb.SimulationConfig:
    _to_update = ["soc_0", "t_0", "current_policies", "policy_choice_distribution"]

    def dirac_policy_choice_distribution() -> int:
        return 0

    # Get the policy at the given index
    policy_id = tc_data.loc[idx, "policy_id"]

    # Update to a deterministic policy if the event has happened
    if event_fn(t0, soc0):
        policy_choice_distribution = dirac_policy_choice_distribution
        current_policies = [base_config.current_policies[policy_id]]
    else:
        policy_choice_distribution = base_config.policy_choice_distribution
        current_policies = base_config.current_policies

    return sb.SimulationConfig(
        policy_choice_distribution=policy_choice_distribution,
        current_policies=current_policies,
        soc_0=soc0,
        t_0=t0,
        **{k: v for k, v in base_config.__dict__.items() if k not in _to_update},
    )


def reconstruct_true_rul_distribution(
    test_data: pd.DataFrame,
    simulator_base_config: sb.SimulationConfig,
    n_soc0s: int,
    n_mc_samples: int,
    simulator_updater: SimulatorUpdater,
    event_fn: EventFunction,
) -> TestRULSamples:
    """Simulate reference RUL samples at evenly spaced timestamps of every test case.

    The samples are a pure function of the arguments: the simulator's random
    generators (held by the config's noise and policy-choice callables) are copied
    before simulating, so every call, and so every method run's evaluation, is scored
    against the same reference samples, and ``simulator_base_config`` is left
    unchanged.

    Args:
        test_data: Test histories with ``run_id``, ``time``, ``soc`` and
            ``policy_id`` columns.
        simulator_base_config: The data-generation run's test simulation config.
        n_soc0s: Timestamps per test case (``t0`` included), evenly spaced in rows.
        n_mc_samples: Reference samples per timestamp.
        simulator_updater: Builds the config of one timestamp.
        event_fn: Whether the policy is known at ``(t0, soc0)``.

    Returns:
        ``(n_soc0s, n_mc_samples)`` RUL samples keyed ``test_case_<run_id>``.
    """
    simulator_base_config = copy.deepcopy(simulator_base_config)
    # Count the number of unique test cases; create the output data structure
    tc_run_ids = test_data["run_id"].unique()
    out = dict.fromkeys([f"test_case_{i}" for i in tc_run_ids], None)

    # Loop over the test cases
    for tc_i in tc_run_ids:
        # Define an empty array to hold the RUL samples for this test case
        rul_samples = np.empty((n_soc0s, n_mc_samples), dtype=np.float32)

        # Extract the test case data and sort it by time
        tc_data = test_data[test_data["run_id"] == tc_i]
        tc_data = tc_data.sort_values(by="time", ascending=True).reset_index(drop=True)

        # Define the interemediate checkpoints and corresponding SOCs and times
        checkpoints = np.linspace(0, len(tc_data) - 1, num=n_soc0s, dtype=np.int32)
        soc0s = tc_data.loc[checkpoints, "soc"].to_numpy()
        t0s = tc_data.loc[checkpoints, "time"].to_numpy()

        # Loop over the checkpoints
        for i, (checkpoint, soc0, t0) in enumerate(zip(checkpoints, soc0s, t0s)):
            # Update the config with SOC, t_0, policies and policy choice distribution
            updated_config = simulator_updater(
                base_config=simulator_base_config,
                idx=checkpoint,
                soc0=soc0,
                t0=t0,
                tc_data=tc_data,
                event_fn=event_fn,
            )

            # Simulate
            result = sb.simulate_constant_capacity_simple(
                n_sim=n_mc_samples, config=updated_config
            )

            # Update the checkpoint array
            rul_samples[i, :] = result.times_eod - t0

        # Update test case entry
        out[f"test_case_{tc_i}"] = rul_samples

    return out
