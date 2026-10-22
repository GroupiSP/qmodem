from __future__ import annotations

import dataclasses
import enum
import functools
import pathlib
from collections.abc import Callable, Iterable, Mapping

import jax
import jax.numpy as jnp
import jaxtyping
import mlflow
import numpy as np
import pandas as pd
import sklearn.preprocessing as skpp
from stagereg.core import Batch, LoaderFactory

from qmodem.battery.data_generation import (
    TRAIN_STREAM_SPLIT_NAMES,
    n_histories_param,
)
from qmodem.data import (
    ArrayDataSource,
    DataPipeline,
    DataScaler,
    ScalingMode,
    ScalingStep,
    add_feature_dimension_to_y,
    get_time_windows_and_join,
    to_jax,
)

WINDOW_SIZE = 10
"""Window size shared by every method, so that N is the same for every posterior."""

FEATURES: tuple[str, ...] = ("load", "voltage")
"""Input channels of every window."""

VALIDATION_TIMESTAMP_COUNT = 50
"""K: windows per history of both validation sets (ADR 0003).

Each validation history contributes ``min(K, n)`` of its ``n`` stride-1 windows,
evenly spaced and including the first and last. Changing it changes every
validation score and supersedes ADR 0003.
"""

TRAIN_SPLIT = "train"
VALIDATION_SPLIT = "validation"


class ValidationSet(enum.StrEnum):
    """Which validation set a loader's ``"validation"`` split serves.

    The backbone validation set selects the mean and variance stages (and their HPO
    studies); the posterior validation set selects the posterior stage (and its HPO
    study). The two are disjoint.
    """

    BACKBONE = "backbone"
    POSTERIOR = "posterior"


@dataclasses.dataclass
class PreparedData:
    """Windowed, scaled train data and both validation sets.

    Attributes:
        train: Training windows and targets (stride 1).
        backbone_val: Backbone validation set windows and targets, at its
            validation timestamps.
        posterior_val: Posterior validation set windows and targets, at its
            validation timestamps.
        validation_timestamps: K, the validation timestamps per history.
    """

    train: ArrayDataSource
    backbone_val: ArrayDataSource
    posterior_val: ArrayDataSource
    validation_timestamps: int = VALIDATION_TIMESTAMP_COUNT

    def validation(self, which: ValidationSet) -> ArrayDataSource:
        """Return the windows of one validation set.

        Args:
            which: The validation set.

        Returns:
            Its windows and targets.
        """
        return {
            ValidationSet.BACKBONE: self.backbone_val,
            ValidationSet.POSTERIOR: self.posterior_val,
        }[which]

    def window_counts(self) -> dict[str, int]:
        """Return the window count of each split and K, keyed for MLflow params.

        Returns:
            ``{"windows_train": ..., "windows_backbone_val": ...,
            "windows_posterior_val": ..., "validation_timestamps": K}``.
        """
        return {
            "windows_train": len(self.train),
            "windows_backbone_val": len(self.backbone_val),
            "windows_posterior_val": len(self.posterior_val),
            "validation_timestamps": self.validation_timestamps,
        }


@dataclasses.dataclass(frozen=True)
class SplitSizes:
    """History counts of the contiguous ``run_id`` blocks of ``train.csv``.

    Attributes:
        n_train: Training histories (the first block).
        n_backbone_val: Backbone validation set histories (the second block).
        n_posterior_val: Posterior validation set histories (the third block).
    """

    n_train: int
    n_backbone_val: int
    n_posterior_val: int

    @classmethod
    def from_params(cls, params: Mapping[str, str]) -> SplitSizes:
        """Read the sizes from a data-generation run's MLflow params.

        Args:
            params: The data-generation run's params.

        Returns:
            The split sizes.

        Raises:
            KeyError: If a size param is missing.
        """
        n_train, n_backbone_val, n_posterior_val = (
            int(params[n_histories_param(split)]) for split in TRAIN_STREAM_SPLIT_NAMES
        )
        return cls(
            n_train=n_train,
            n_backbone_val=n_backbone_val,
            n_posterior_val=n_posterior_val,
        )


@dataclasses.dataclass(frozen=True)
class HistorySplits:
    """The histories of ``train.csv``, split into disjoint ``run_id`` blocks.

    Attributes:
        train: Training histories.
        backbone_val: Backbone validation set histories.
        posterior_val: Posterior validation set histories.
    """

    train: pd.DataFrame
    backbone_val: pd.DataFrame
    posterior_val: pd.DataFrame


def split_histories(histories: pd.DataFrame, sizes: SplitSizes) -> HistorySplits:
    """Slice the histories of ``train.csv`` into train and both validation sets.

    ``run_id`` ``[0, n_train)`` is train, the next ``n_backbone_val`` the backbone
    validation set and the next ``n_posterior_val`` the posterior validation set.

    Args:
        histories: The histories, with a ``run_id`` column.
        sizes: The block sizes logged on the data-generation run.

    Returns:
        The three disjoint splits, together covering every history.

    Raises:
        ValueError: If the ``run_id``\\ s are not exactly ``0 .. total - 1``.
    """
    run_ids = histories["run_id"]
    total = sizes.n_train + sizes.n_backbone_val + sizes.n_posterior_val
    if set(run_ids.unique().tolist()) != set(range(total)):
        raise ValueError(
            f"Expected run_id 0..{total - 1} ({sizes}), got "
            f"{run_ids.nunique()} distinct run_id values."
        )
    first_posterior = sizes.n_train + sizes.n_backbone_val
    return HistorySplits(
        train=histories[run_ids < sizes.n_train],
        backbone_val=histories[
            (run_ids >= sizes.n_train) & (run_ids < first_posterior)
        ],
        posterior_val=histories[run_ids >= first_posterior],
    )


def _to_batch(features: jaxtyping.ArrayLike, targets: jaxtyping.ArrayLike) -> Batch:
    """Move one batch of windows and targets to the device.

    Args:
        features: The batch's windows.
        targets: The batch's targets.

    Returns:
        The stagereg batch.
    """
    return Batch(inputs=jnp.asarray(features), targets=jnp.asarray(targets))


def _shuffled_batches(
    windows: ArrayDataSource, batch_size: int, seed: int
) -> list[Batch]:
    """Split one epoch's shuffled training windows into batches.

    Args:
        windows: Training windows and targets, backed by NumPy arrays so that
            fancy indexing stays on the host.
        batch_size: Batch size (the last batch may be smaller).
        seed: Seed of this epoch's shuffle.

    Returns:
        The batches, every window appearing exactly once.
    """
    order = np.random.default_rng(seed).permutation(len(windows))
    return [
        _to_batch(*windows[idx])
        for idx in np.split(order, range(batch_size, len(order), batch_size))
    ]


def make_data_pipeline(
    window_size: int = WINDOW_SIZE,
    x_scaler: DataScaler | None = None,
    y_scaler: DataScaler | None = None,
) -> DataPipeline:
    """Build the multiple-scenarios data pipeline (scaling, then windowing).

    Args:
        window_size: Number of time steps per window (stride 1).
        x_scaler: Scaler of the input features. Defaults to a ``StandardScaler``.
        y_scaler: Scaler of the RUL target. Defaults to a ``MinMaxScaler`` on
            ``(0, 1)``.

    Returns:
        A pipeline mapping a histories dataframe to ``(X (N, W, C), y (N, 1))``.
    """
    return DataPipeline(
        [
            ScalingStep(
                scaler=x_scaler if x_scaler is not None else skpp.StandardScaler(),
                features=list(FEATURES),
            ),
            ScalingStep(
                scaler=(
                    y_scaler
                    if y_scaler is not None
                    else skpp.MinMaxScaler(feature_range=(0, 1))
                ),
                features=["rul"],
            ),
            functools.partial(
                get_time_windows_and_join,
                window_size=window_size,
                stride=1,
                features=list(FEATURES),
            ),
            add_feature_dimension_to_y,
            to_jax,
        ]
    )


def validation_timestamp_indices(n_windows: int, k: int) -> np.ndarray:
    """Return the indices of one history's validation timestamps.

    The same ``linspace`` rule as a test case's evaluation timestamps
    (``evaluate.evaluation_indices``): ``min(k, n_windows)`` evenly spaced
    indices, the first and last window included.

    Args:
        n_windows: The history's number of stride-1 windows.
        k: K, the validation timestamps per history.

    Returns:
        The increasing, distinct window indices ``(min(k, n_windows),)``.

    Raises:
        ValueError: If ``k < 2`` (the last window would be dropped).
    """
    if k < 2:
        raise ValueError(f"K must be at least 2 to keep the last window, got {k}.")
    return np.linspace(0, n_windows - 1, num=min(k, n_windows), dtype=np.int64)


def _validation_windows(
    histories: pd.DataFrame, pipeline: DataPipeline, k: int
) -> ArrayDataSource:
    """Window one validation set at its validation timestamps.

    Args:
        histories: The validation set's histories, with a ``run_id`` column.
        pipeline: The data pipeline, in ``TRANSFORM`` mode.
        k: K, the validation timestamps per history.

    Returns:
        Per history, in ``run_id`` order of appearance, its selected windows.
    """
    features, targets = [], []
    for _, history in histories.groupby("run_id", sort=False):
        X, y = pipeline(history)
        keep = validation_timestamp_indices(len(y), k)
        features.append(np.asarray(X)[keep])
        targets.append(np.asarray(y)[keep])
    if not features:
        X, y = pipeline(histories)
        return ArrayDataSource(features=X, targets=y)
    return ArrayDataSource(
        features=jnp.asarray(np.concatenate(features)),
        targets=jnp.asarray(np.concatenate(targets)),
    )


def prepare_windows(
    splits: HistorySplits,
    pipeline: DataPipeline,
    *,
    fit_scalers: bool = True,
    validation_timestamps: int = VALIDATION_TIMESTAMP_COUNT,
) -> PreparedData:
    """Window the train and both validation sets with train-fitted scalers.

    Training windows are every stride-1 window. Both validation sets keep only
    their validation timestamps (``validation_timestamp_indices``, ADR 0003).

    Args:
        splits: The train, backbone validation set and posterior validation set
            histories.
        pipeline: The data pipeline.
        fit_scalers: Fit the pipeline's scalers on the training histories (a
            backbone run). ``False`` uses already fitted scalers as they are
            (runs consuming a backbone, see ``backbone_data_pipeline``).
        validation_timestamps: K, the validation timestamps per history.

    Returns:
        The prepared windows of every split.

    Raises:
        ValueError: If ``validation_timestamps < 2``.
    """
    pipeline.set_mode(
        ScalingMode.FIT_TRANSFORM if fit_scalers else ScalingMode.TRANSFORM
    )
    X_train, y_train = pipeline(splits.train)

    pipeline.set_mode(ScalingMode.TRANSFORM)
    return PreparedData(
        train=ArrayDataSource(features=X_train, targets=y_train),
        backbone_val=_validation_windows(
            splits.backbone_val, pipeline, validation_timestamps
        ),
        posterior_val=_validation_windows(
            splits.posterior_val, pipeline, validation_timestamps
        ),
        validation_timestamps=validation_timestamps,
    )


def posterior_dataset_size(data: PreparedData) -> int:
    """Return the posterior's dataset size N: the number of training windows.

    Args:
        data: The prepared windows.

    Returns:
        The number of training windows (stride 1).
    """
    return len(data.train)


def _shuffle_seed(key: jax.Array, epoch: int) -> int:
    """Derive the integer shuffle seed of one epoch from the loader key.

    Args:
        key: The loader's typed PRNG key.
        epoch: The epoch index.

    Returns:
        A non-negative integer seed, the same for the same ``(key, epoch)``.
    """
    epoch_key = jax.random.fold_in(key, epoch)
    return int(jax.random.randint(epoch_key, (), 0, jnp.iinfo(jnp.int32).max))


def make_loader(
    data: PreparedData,
    batch_size: int,
    key: jax.Array,
    *,
    validation: ValidationSet,
) -> LoaderFactory:
    """Build stagereg's ``LoaderFactory`` over the prepared windows.

    The training split is reshuffled every epoch with a seed derived from
    ``key`` and the epoch, so the same epoch always gives the same batch order.
    The validation split is one batch holding every window of the ``validation``
    set, in order.

    Args:
        data: The prepared windows.
        batch_size: Training batch size (the last batch may be smaller).
        key: Typed PRNG key the training shuffle derives from.
        validation: The validation set the ``"validation"`` split serves.

    Returns:
        A loader ``loader(*, split, epoch) -> list[Batch]``.
    """
    # Converted once: indexing NumPy is far cheaper than indexing device arrays.
    train = ArrayDataSource(
        features=np.asarray(data.train.features),
        targets=np.asarray(data.train.targets),
    )
    served = data.validation(validation)
    val_batch = _to_batch(served.features, served.targets)
    batches_of: dict[str, Callable[[int], list[Batch]]] = {
        TRAIN_SPLIT: lambda epoch: _shuffled_batches(
            train, batch_size, _shuffle_seed(key, epoch)
        ),
        VALIDATION_SPLIT: lambda epoch: [val_batch],
    }

    def loader(*, split: str, epoch: int) -> list[Batch]:
        """Return the batches of one split and epoch.

        Args:
            split: ``"train"`` or ``"validation"``.
            epoch: The epoch index.

        Returns:
            The batches.

        Raises:
            ValueError: If the split is unknown.
        """
        try:
            build = batches_of[split]
        except KeyError:
            raise ValueError(
                f"Unknown split {split!r}; expected one of {sorted(batches_of)}."
            ) from None
        return build(epoch)

    return loader


def restore_only_loader(*, split: str, epoch: int) -> Iterable[Batch]:
    """The loader of a stage that is only restored, never trained.

    Args:
        split: The requested split (never served).
        epoch: The requested epoch (never served).

    Raises:
        RuntimeError: Always; a restored stage must not read data.
    """
    raise RuntimeError(
        f"A restored stage must not read data (asked for {split!r}, epoch {epoch})."
    )


def prepare_data(
    raw_data_dir: pathlib.Path,
    data_gen_run_id: str,
    pipeline: DataPipeline,
    *,
    fit_scalers: bool = True,
) -> PreparedData:
    """Prepare the multiple-scenarios windows from the raw training histories.

    ``train.csv`` is sliced into train, backbone validation set and posterior
    validation set by ``run_id`` range (``split_histories``), with the sizes
    logged on the data-generation run (``SplitSizes.from_params``).

    Args:
        raw_data_dir: Directory holding ``train.csv``.
        data_gen_run_id: MLflow run ID of the data-generation run. The MLflow
            tracking URI must already be set.
        pipeline: The data pipeline (see ``make_data_pipeline``).
        fit_scalers: Fit the pipeline's scalers on the training histories;
            ``False`` keeps already fitted ones (see ``prepare_windows``).

    Returns:
        The prepared windows of every split.

    Raises:
        ValueError: If ``train.csv`` does not match the logged split sizes.
    """
    data_gen_run = mlflow.get_run(data_gen_run_id)
    splits = split_histories(
        pd.read_csv(raw_data_dir / "train.csv"),
        SplitSizes.from_params(data_gen_run.data.params),
    )
    return prepare_windows(splits, pipeline, fit_scalers=fit_scalers)
