"""Per-test-case evaluation driver of the comparison protocol (§6).

The pure core maps per-timestamp predictive samples and simulator reference samples
of each test case to a per-test-case metric table (``case_metrics``,
``metric_table``). Aggregation over test cases is a separate function
(``aggregate``). Every metric is a stagereg metric function; qmodem has no metric
code of its own.

The first timestamp of a test case (t0) has no prediction. It is dropped from every
metric and serves only as the time origin of WSU.

Around the core, ``evaluate_predictive_model`` builds the samples from a predictive
model (M posterior draws, each with one aleatoric noise sample), and
``log_evaluation`` logs the result, with one RUL-over-time figure per test case,
into a method run through an ``ExperimentTracker``.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from stagereg.core import (
    ExperimentTracker,
    coverage,
    expected_sample_crps,
    rmse,
    wsu,
)
from stagereg.jax import PosteriorPredictive

from .data_processing import FEATURES, WINDOW_SIZE
from .rul_plot import central_band, plot_rul_over_time

PRIMARY_COVERAGE_LEVEL = 0.95
"""The protocol's primary coverage level, also the band WSU integrates."""

RELIABILITY_LEVELS: tuple[float, ...] = (
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    PRIMARY_COVERAGE_LEVEL,
)
"""Default coverage levels: the reliability curve (F1), primary level last."""

EVALUATION_PREFIX = "evaluation/test"
"""Prefix of every evaluation metric and artifact of a method run."""


@dataclasses.dataclass(frozen=True)
class CaseSamples:
    """Predictive and reference samples of one test case, one row per timestamp.

    Row 0 is t0. Its target, reference and predictive samples are never used, so a
    missing prediction may be given as NaN.

    Attributes:
        name: Test-case name, e.g. ``"case_3"``.
        policy: Current-policy ID of the test case.
        times: Strictly increasing timestamps ``(T,)``, t0 first.
        targets: Realized RUL at each timestamp ``(T,)``.
        reference_samples: Simulator reference RUL samples ``(T, R)``.
        predictive_samples: Predictive RUL samples ``(T, M)``.
    """

    name: str
    policy: int
    times: np.ndarray
    targets: np.ndarray
    reference_samples: np.ndarray
    predictive_samples: np.ndarray


def coverage_key(level: float) -> str:
    """Return the metric name of the coverage at ``level``.

    Args:
        level: Nominal coverage level in ``(0, 1)``.

    Returns:
        The metric name, e.g. ``"coverage_95"`` for 0.95.
    """
    return f"coverage_{100 * level:g}"


def case_metrics(
    case: CaseSamples,
    coverage_levels: Sequence[float] = (PRIMARY_COVERAGE_LEVEL,),
    wsu_level: float = PRIMARY_COVERAGE_LEVEL,
) -> dict[str, float]:
    """Compute the protocol metrics of one test case, t0 excluded.

    - ``crps``: mean over timestamps of the expected CRPS against the reference
      samples (``expected_sample_crps``).
    - ``rmse``: of the predictive sample means against the realized targets.
    - ``coverage_<level>``: share of targets inside the empirical central band.
    - ``wsu``: stagereg's per-trajectory WSU of the ``wsu_level`` band, with t0 as
      the time origin.

    Args:
        case: The test case's samples.
        coverage_levels: Nominal levels of the reported coverages.
        wsu_level: Nominal level of the band WSU integrates.

    Returns:
        The metric values by name.

    Raises:
        ValueError: If a sample used by a metric is not finite, or the case has
            fewer than two evaluated timestamps (see ``stagereg.core.wsu``).
    """
    predictive = case.predictive_samples[1:]
    targets = case.targets[1:]
    metrics = {
        "crps": float(
            expected_sample_crps(predictive, case.reference_samples[1:]).mean()
        ),
        "rmse": rmse(targets, predictive.mean(axis=1)),
    }
    for level in coverage_levels:
        metrics[coverage_key(level)] = coverage(
            targets, *central_band(predictive, level)
        )
    lower, upper = central_band(predictive, wsu_level)
    # stagereg's wsu takes the origin from x[0] and never reads lower[0]/upper[0].
    metrics["wsu"] = wsu(
        np.concatenate([lower[:1], lower]),
        np.concatenate([upper[:1], upper]),
        case.times,
    )
    return metrics


def metric_table(
    cases: Sequence[CaseSamples],
    coverage_levels: Sequence[float] = (PRIMARY_COVERAGE_LEVEL,),
    wsu_level: float = PRIMARY_COVERAGE_LEVEL,
) -> dict[str, dict[str, float]]:
    """Compute the metrics of every test case.

    Args:
        cases: The test cases' samples.
        coverage_levels: Nominal levels of the reported coverages.
        wsu_level: Nominal level of the band WSU integrates.

    Returns:
        The metrics of each case (see ``case_metrics``), keyed by case name in input
        order.
    """
    return {case.name: case_metrics(case, coverage_levels, wsu_level) for case in cases}


@dataclasses.dataclass(frozen=True)
class AggregateMetrics:
    """Metrics averaged over test cases.

    Attributes:
        overall: Unweighted mean over every test case (primary).
        per_policy: Unweighted mean over the cases of each current policy
            (secondary).
    """

    overall: dict[str, float]
    per_policy: dict[int, dict[str, float]]


def _mean_rows(rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
    """Average metric rows key by key.

    Args:
        rows: Metric rows sharing the same keys.

    Returns:
        The unweighted mean of each metric.
    """
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def aggregate(
    table: Mapping[str, Mapping[str, float]], policies: Mapping[str, int]
) -> AggregateMetrics:
    """Average a metric table over test cases, overall and per current policy.

    Args:
        table: Metrics of each test case, keyed by case name.
        policies: Current-policy ID of each test case, keyed by case name.

    Returns:
        The unweighted means over all cases and over the cases of each policy.

    Raises:
        ValueError: If ``table`` is empty.
    """
    if not table:
        raise ValueError("Cannot aggregate an empty metric table.")
    by_policy: dict[int, list[Mapping[str, float]]] = {}
    for name, row in table.items():
        by_policy.setdefault(policies[name], []).append(row)
    return AggregateMetrics(
        overall=_mean_rows(list(table.values())),
        per_policy={
            policy: _mean_rows(rows) for policy, rows in sorted(by_policy.items())
        },
    )


# --- Driver: sampling a predictive model on the test histories -------------------


class FeatureScaler(Protocol):
    """Scales input features fitted on the training histories."""

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Scale features ``(N, C)``.

        Args:
            x: Raw features.

        Returns:
            The scaled features.
        """
        ...


class TargetScaler(Protocol):
    """Maps scaled RUL back to seconds.

    Assumed affine (e.g. min-max).
    """

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        """Unscale targets ``(N, 1)``.

        Args:
            x: Scaled targets.

        Returns:
            The targets in RUL units.
        """
        ...


class PredictiveModel(Protocol):
    """A model with a posterior predictive, such as a ``MethodRun``."""

    def predictive(
        self, inputs: jax.Array, key: jax.Array, num_draws: int
    ) -> PosteriorPredictive:
        """Return the posterior predictive at ``inputs``.

        Args:
            inputs: Windows ``(B, W, C)``.
            key: Key of the posterior draws.
            num_draws: Number of draws M.

        Returns:
            Draw means ``(M, B, 1)`` and the aleatoric variance ``(B, 1)``.
        """
        ...


class EvaluationParameters(BaseModel):
    """The protocol's evaluation parameters (§6).

    Attributes:
        num_samples: Predictive samples per timestamp M (draws, one noise each).
        n_timestamps: Evaluation timestamps per test case, t0 included.
        coverage_levels: Nominal levels of the reported coverages.
        wsu_level: Nominal level of the band WSU integrates.
        window_size: Length of each input window.
        features: Input channels of each window.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    num_samples: int = Field(default=100, ge=1)
    n_timestamps: int = Field(default=20, ge=3)
    coverage_levels: tuple[float, ...] = RELIABILITY_LEVELS
    wsu_level: float = Field(default=PRIMARY_COVERAGE_LEVEL, gt=0.0, lt=1.0)
    window_size: int = Field(default=WINDOW_SIZE, ge=1)
    features: tuple[str, ...] = FEATURES


@dataclasses.dataclass(frozen=True)
class CasePrediction:
    """The predictive samples of one test case and their variance split.

    Attributes:
        samples: The case's samples (t0 prediction row is NaN).
        epistemic_variance: Variance of the draw means, per evaluated timestamp
            ``(T - 1,)``, in RUL units.
        aleatoric_variance: Mean aleatoric variance, per evaluated timestamp
            ``(T - 1,)``, in RUL units.
    """

    samples: CaseSamples
    epistemic_variance: np.ndarray
    aleatoric_variance: np.ndarray

    def to_artifact(self) -> dict[str, Any]:
        """Return the case as a JSON-serializable mapping.

        Prediction rows (``predictive_samples`` and the variances) start at the
        first evaluated timestamp; ``times``, ``targets`` and ``reference_samples``
        include t0.

        Returns:
            The artifact payload.
        """
        samples = self.samples
        return {
            "name": samples.name,
            "policy": samples.policy,
            "times": samples.times.tolist(),
            "targets": samples.targets.tolist(),
            "reference_samples": samples.reference_samples.tolist(),
            "predictive_samples": samples.predictive_samples[1:].tolist(),
            "epistemic_variance": self.epistemic_variance.tolist(),
            "aleatoric_variance": self.aleatoric_variance.tolist(),
        }


@dataclasses.dataclass(frozen=True)
class Evaluation:
    """The evaluation of one predictive model on the test cases.

    Attributes:
        predictions: Samples and variance split of each test case.
        table: Metrics of each test case.
        aggregate: Means over test cases, overall and per policy.
    """

    predictions: list[CasePrediction]
    table: dict[str, dict[str, float]]
    aggregate: AggregateMetrics


def evaluation_indices(n_rows: int, n_timestamps: int) -> np.ndarray:
    """Return the row indices of the evaluation timestamps of one test case.

    They match the checkpoints of ``reconstruct_true_rul_distribution``.

    Args:
        n_rows: Number of rows of the time-sorted test case.
        n_timestamps: Number of timestamps, t0 included.

    Returns:
        The row index of each timestamp ``(n_timestamps,)``.
    """
    return np.linspace(0, n_rows - 1, num=n_timestamps, dtype=np.int32)


def _windows(features: np.ndarray, ends: np.ndarray, window_size: int) -> np.ndarray:
    """Cut the window ending one step before each index.

    A window that would start before the first row is left-padded with the first
    row, as in training.

    Args:
        features: Scaled features ``(N, C)``.
        ends: Exclusive end index of each window.
        window_size: Window length W.

    Returns:
        Windows ``(len(ends), W, C)``.
    """
    windows = []
    for end in ends:
        window = features[max(0, end - window_size) : end]
        windows.append(
            np.pad(window, ((window_size - len(window), 0), (0, 0)), mode="edge")
        )
    return np.stack(windows)


def _unscale(y_scaler: TargetScaler, values: np.ndarray) -> np.ndarray:
    """Apply the inverse target scaling elementwise.

    Args:
        y_scaler: The target scaler.
        values: Scaled targets of any shape.

    Returns:
        The targets in RUL units, shaped like ``values``.
    """
    flat = np.asarray(values, dtype=np.float64).reshape(-1, 1)
    return np.asarray(y_scaler.inverse_transform(flat)).reshape(values.shape)


def predict_case(
    model: PredictiveModel,
    history: pd.DataFrame,
    reference_samples: np.ndarray,
    *,
    name: str,
    x_scaler: FeatureScaler,
    y_scaler: TargetScaler,
    parameters: EvaluationParameters,
    key: jax.Array,
) -> CasePrediction:
    """Draw the predictive samples of one test case.

    At each evaluated timestamp, M posterior draws are taken and each gets one
    aleatoric noise sample. t0 gets no prediction (a NaN row).

    Args:
        model: The predictive model.
        history: The test case's rows (any order).
        reference_samples: Simulator reference samples ``(T, R)``.
        name: Test-case name.
        x_scaler: Input feature scaler.
        y_scaler: Target scaler (affine).
        parameters: Evaluation parameters.
        key: Key of this case's draws and noise.

    Returns:
        The case's samples and variance split, in RUL units.

    Raises:
        ValueError: If the case has several policies, or the reference samples
            do not have one row per timestamp.
    """
    history = history.sort_values("time").reset_index(drop=True)
    policy = _case_policy(history, reference_samples, name=name, parameters=parameters)
    indices = evaluation_indices(len(history), parameters.n_timestamps)
    features = x_scaler.transform(
        history[list(parameters.features)].to_numpy(dtype=np.float64)
    )
    windows = _windows(np.asarray(features), indices[1:], parameters.window_size)
    sampled = _sample_predictive(
        model, windows, y_scaler=y_scaler, num_samples=parameters.num_samples, key=key
    )
    # A finite posterior (pSGLD) yields its retained draws, not num_samples.
    no_prediction = np.full((1, sampled.samples.shape[0]), np.nan)
    return CasePrediction(
        samples=CaseSamples(
            name=name,
            policy=policy,
            times=history["time"].to_numpy(dtype=np.float64)[indices],
            targets=history["rul"].to_numpy(dtype=np.float64)[indices],
            reference_samples=np.asarray(reference_samples, dtype=np.float64),
            predictive_samples=np.vstack([no_prediction, sampled.samples.T]),
        ),
        epistemic_variance=sampled.means.var(axis=0),
        aleatoric_variance=(sampled.noise_scale**2).mean(axis=0),
    )


def _case_policy(
    history: pd.DataFrame,
    reference_samples: np.ndarray,
    *,
    name: str,
    parameters: EvaluationParameters,
) -> int:
    """Validate a test case and return its policy.

    Args:
        history: The test case's rows.
        reference_samples: Its simulator reference samples ``(T, R)``.
        name: Test-case name.
        parameters: Evaluation parameters.

    Returns:
        The case's single policy.

    Raises:
        ValueError: If the case has several policies, or the reference samples
            do not have one row per timestamp.
    """
    policies = history["policy_id"].unique()
    if len(policies) != 1:
        raise ValueError(f"Test case {name!r} has several policies: {policies}.")
    if len(reference_samples) != parameters.n_timestamps:
        raise ValueError(
            f"Test case {name!r} has {len(reference_samples)} reference rows; "
            f"expected {parameters.n_timestamps}."
        )
    return int(policies[0])


@dataclasses.dataclass(frozen=True)
class _PredictiveSamples:
    """Predictive samples at the evaluated windows, in RUL units.

    Attributes:
        means: The draws' means ``(M, T-1)``.
        noise_scale: The aleatoric standard deviation per draw ``(M, T-1)``.
        samples: One noisy sample per draw ``(M, T-1)``.
    """

    means: np.ndarray
    noise_scale: np.ndarray
    samples: np.ndarray


def _sample_predictive(
    model: PredictiveModel,
    windows: np.ndarray,
    *,
    y_scaler: TargetScaler,
    num_samples: int,
    key: jax.Array,
) -> _PredictiveSamples:
    """Draw M posterior draws and one aleatoric noise sample per draw.

    Args:
        model: The predictive model.
        windows: Scaled input windows ``(T-1, W, C)``.
        y_scaler: Target scaler (affine).
        num_samples: Draws M of an unlimited posterior.
        key: Split into the draw key and the noise key.

    Returns:
        The draws' means, noise scales and samples, unscaled.
    """
    draw_key, noise_key = jax.random.split(key)
    predictive = model.predictive(
        jnp.asarray(windows, dtype=jnp.float32), draw_key, num_samples
    )
    draws = np.asarray(predictive.draws, dtype=np.float64)[..., 0]  # (M, T-1)
    std = np.sqrt(
        np.broadcast_to(np.asarray(predictive.aleatoric_variance)[..., 0], draws.shape)
    )
    noise = np.asarray(jax.random.normal(noise_key, draws.shape), dtype=np.float64)
    means = _unscale(y_scaler, draws)
    noise_scale = _unscale(y_scaler, draws + std) - means  # affine: a * std
    return _PredictiveSamples(
        means=means, noise_scale=noise_scale, samples=means + noise_scale * noise
    )


def evaluate_predictive_model(
    model: PredictiveModel,
    *,
    test_histories: pd.DataFrame,
    reference_samples: Mapping[str, np.ndarray],
    x_scaler: FeatureScaler,
    y_scaler: TargetScaler,
    parameters: EvaluationParameters,
    key: jax.Array,
) -> Evaluation:
    """Evaluate a predictive model on every test case.

    Test case ``i`` (``run_id`` i) is named ``case_<i>`` and draws with
    ``fold_in(key, i)``.

    Args:
        model: The predictive model.
        test_histories: The test histories (``run_id``, ``policy_id``, ``time``,
            the features and ``rul``).
        reference_samples: Simulator reference samples ``(T, R)`` keyed
            ``test_case_<i>``, as built by ``reconstruct_true_rul_distribution``.
        x_scaler: Input feature scaler.
        y_scaler: Target scaler (affine).
        parameters: Evaluation parameters.
        key: The evaluation key of the method run.

    Returns:
        The samples, metric table and aggregates.
    """
    predictions = [
        predict_case(
            model,
            history,
            reference_samples[f"test_case_{case_id}"],
            name=f"case_{case_id}",
            x_scaler=x_scaler,
            y_scaler=y_scaler,
            parameters=parameters,
            key=jax.random.fold_in(key, int(case_id)),
        )
        for case_id, history in test_histories.groupby("run_id", sort=True)
    ]
    cases = [prediction.samples for prediction in predictions]
    table = metric_table(cases, parameters.coverage_levels, parameters.wsu_level)
    return Evaluation(
        predictions=predictions,
        table=table,
        aggregate=aggregate(table, {case.name: case.policy for case in cases}),
    )


def log_evaluation(
    evaluation: Evaluation,
    tracker: ExperimentTracker,
    *,
    band_level: float = PRIMARY_COVERAGE_LEVEL,
) -> None:
    """Log an evaluation into a method run.

    Metrics go to ``evaluation/test/<case>/<metric>`` (per case),
    ``evaluation/test/<metric>`` (mean over cases) and
    ``evaluation/test/policy_<p>/<metric>`` (mean over a policy's cases), at step
    0. Each case's samples and variance split go to the artifact
    ``evaluation/test/<case>/predictive`` (JSON), and its RUL-over-time figure to
    ``evaluation/test/<case>/rul_over_time`` (PNG).

    Args:
        evaluation: The evaluation.
        tracker: The method run's tracker.
        band_level: Nominal level of the bands in the RUL-over-time figures.
    """
    metrics = {
        f"{EVALUATION_PREFIX}/{case}/{metric}": value
        for case, row in evaluation.table.items()
        for metric, value in row.items()
    }
    metrics |= {
        f"{EVALUATION_PREFIX}/{metric}": value
        for metric, value in evaluation.aggregate.overall.items()
    }
    metrics |= {
        f"{EVALUATION_PREFIX}/policy_{policy}/{metric}": value
        for policy, row in evaluation.aggregate.per_policy.items()
        for metric, value in row.items()
    }
    tracker.log_metrics(metrics, step=0)
    for prediction in evaluation.predictions:
        case_prefix = f"{EVALUATION_PREFIX}/{prediction.samples.name}"
        tracker.log_artifact(f"{case_prefix}/predictive", prediction.to_artifact())
        tracker.log_artifact(
            f"{case_prefix}/rul_over_time",
            plot_rul_over_time(prediction.samples, level=band_level),
        )
