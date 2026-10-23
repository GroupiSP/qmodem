"""Per-test-case evaluation driver of the comparison protocol (§6).

The pure core maps the exact posterior predictive (``CasePredictive``) and the
simulator reference samples of each test case to a per-test-case metric table
(``case_metrics``, ``metric_table``). Aggregation over test cases is a separate
function (``aggregate``). Every metric is a stagereg metric function; qmodem has no
metric code of its own.

The first timestamp of a test case (t0) has no prediction. It is dropped from every
metric and serves only as the time origin of WSU.

Around the core, ``evaluate_predictive_model`` builds the posterior predictive of
each case from a predictive model (M posterior draws; no aleatoric noise is
sampled), and ``log_evaluation`` logs the result, with one RUL-over-time figure per
test case, into a method run through an ``ExperimentTracker``.
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
from stagereg.core import ExperimentTracker, coverage, rmse, wsu
from stagereg.jax import PosteriorPredictive

from .data_processing import FEATURES, WINDOW_SIZE, evenly_spaced_indices
from .forecasts import CasePredictive, ReferenceSamples
from .rul_plot import plot_rul_over_time

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
class EvaluatedCase:
    """One evaluated test case: its timeline, reference and posterior predictive.

    Row 0 of ``times``, ``targets`` and the reference is t0. Its target and
    reference are never scored, and the predictive has no t0 row.

    Attributes:
        name: Test-case name, e.g. ``"case_3"``.
        policy: Current-policy ID of the test case.
        times: Strictly increasing timestamps ``(T,)``, t0 first.
        targets: Realized RUL at each timestamp ``(T,)``.
        reference: Simulator reference RUL samples, t0 included.
        predictive: Posterior predictive in RUL units, t0 excluded.
    """

    name: str
    policy: int
    times: np.ndarray
    targets: np.ndarray
    reference: ReferenceSamples
    predictive: CasePredictive

    def to_artifact(self) -> dict[str, Any]:
        """Return the case as a JSON-serializable mapping.

        ``times``, ``targets`` and ``reference_samples`` include t0;
        ``draw_means`` ``(T - 1, M)`` and ``aleatoric_variance`` ``(T - 1,)`` start
        at the first evaluated timestamp.

        Returns:
            The artifact payload, the inverse of ``from_artifact``.
        """
        return {
            "name": self.name,
            "policy": self.policy,
            "times": self.times.tolist(),
            "targets": self.targets.tolist(),
            "reference_samples": self.reference.samples.tolist(),
            "draw_means": self.predictive.draw_means.tolist(),
            "aleatoric_variance": self.predictive.aleatoric_variance.tolist(),
        }

    @classmethod
    def from_artifact(cls, artifact: Mapping[str, Any]) -> EvaluatedCase:
        """Rebuild a case from its artifact payload.

        Args:
            artifact: A payload of ``to_artifact``, e.g. read back from JSON.

        Returns:
            The case, scoring exactly as the logged one.

        Raises:
            KeyError: If a field of the artifact format is missing.
        """

        def array(key: str) -> np.ndarray:
            return np.asarray(artifact[key], dtype=np.float64)

        return cls(
            name=str(artifact["name"]),
            policy=int(artifact["policy"]),
            times=array("times"),
            targets=array("targets"),
            reference=ReferenceSamples(array("reference_samples")),
            predictive=CasePredictive(
                draw_means=array("draw_means"),
                aleatoric_variance=array("aleatoric_variance"),
            ),
        )


def coverage_key(level: float) -> str:
    """Return the metric name of the coverage at ``level``.

    Args:
        level: Nominal coverage level in ``(0, 1)``.

    Returns:
        The metric name, e.g. ``"coverage_95"`` for 0.95.
    """
    return f"coverage_{100 * level:g}"


def case_metrics(
    case: EvaluatedCase,
    coverage_levels: Sequence[float] = (PRIMARY_COVERAGE_LEVEL,),
    wsu_level: float = PRIMARY_COVERAGE_LEVEL,
) -> dict[str, float]:
    """Compute the protocol metrics of one test case, t0 excluded.

    Every metric is exact given the posterior draws:

    - ``crps``: mean over timestamps of the closed-form expected CRPS against the
      reference samples (``CasePredictive.crps``).
    - ``rmse``: of the mixture mean against the realized targets.
    - ``coverage_<level>``: share of targets inside the central mixture interval.
    - ``wsu``: stagereg's per-trajectory WSU of the ``wsu_level`` mixture interval,
      with t0 as the time origin.

    Args:
        case: The evaluated test case.
        coverage_levels: Nominal levels of the reported coverages.
        wsu_level: Nominal level of the band WSU integrates.

    Returns:
        The metric values by name.

    Raises:
        ValueError: If a value used by a metric is not finite, or the case has
            fewer than two evaluated timestamps (see ``stagereg.core.wsu``).
    """
    predictive = case.predictive
    targets = case.targets[1:]
    metrics = {
        "crps": float(predictive.crps(case.reference).mean()),
        "rmse": rmse(targets, predictive.mean()),
    }
    for level in coverage_levels:
        metrics[coverage_key(level)] = coverage(targets, *predictive.interval(level))
    lower, upper = predictive.interval(wsu_level)
    # stagereg's wsu takes the origin from x[0] and never reads lower[0]/upper[0].
    metrics["wsu"] = wsu(
        np.concatenate([lower[:1], lower]),
        np.concatenate([upper[:1], upper]),
        case.times,
    )
    return metrics


def metric_table(
    cases: Sequence[EvaluatedCase],
    coverage_levels: Sequence[float] = (PRIMARY_COVERAGE_LEVEL,),
    wsu_level: float = PRIMARY_COVERAGE_LEVEL,
) -> dict[str, dict[str, float]]:
    """Compute the metrics of every test case.

    Args:
        cases: The evaluated test cases.
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


# --- Driver: the posterior predictive of a model on the test histories ----------


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
        num_draws: Posterior draws M per timestamp (an unlimited posterior's).
        n_timestamps: Evaluation timestamps per test case, t0 included.
        coverage_levels: Nominal levels of the reported coverages.
        wsu_level: Nominal level of the band WSU integrates.
        window_size: Length of each input window.
        features: Input channels of each window.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    num_draws: int = Field(default=100, ge=1)
    n_timestamps: int = Field(default=20, ge=3)
    coverage_levels: tuple[float, ...] = RELIABILITY_LEVELS
    wsu_level: float = Field(default=PRIMARY_COVERAGE_LEVEL, gt=0.0, lt=1.0)
    window_size: int = Field(default=WINDOW_SIZE, ge=1)
    features: tuple[str, ...] = FEATURES


@dataclasses.dataclass(frozen=True)
class Evaluation:
    """The evaluation of one predictive model on the test cases.

    Attributes:
        cases: Each evaluated test case, with its posterior predictive.
        table: Metrics of each test case.
        aggregate: Means over test cases, overall and per policy.
    """

    cases: list[EvaluatedCase]
    table: dict[str, dict[str, float]]
    aggregate: AggregateMetrics


def evaluation_indices(n_rows: int, n_timestamps: int) -> np.ndarray:
    """Return the row indices of the evaluation timestamps of one test case.

    They match the checkpoints of ``reconstruct_true_rul_distribution``
    (``data_processing.evenly_spaced_indices``).

    Args:
        n_rows: Number of rows of the time-sorted test case.
        n_timestamps: Number of timestamps, t0 included.

    Returns:
        The row index of each timestamp ``(n_timestamps,)``.
    """
    return evenly_spaced_indices(n_rows, n_timestamps, dtype=np.int32)


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


def _unscale_affine(y_scaler: TargetScaler) -> tuple[float, float]:
    """Return the scale ``a`` and shift ``b`` of the affine inverse scaling.

    Args:
        y_scaler: The target scaler (affine).

    Returns:
        ``(a, b)`` with ``inverse_transform(y) == a * y + b``.
    """
    zero, one = np.asarray(
        y_scaler.inverse_transform(np.array([[0.0], [1.0]])), dtype=np.float64
    ).ravel()
    return float(one - zero), float(zero)


def unscale_predictive(
    predictive: PosteriorPredictive, y_scaler: TargetScaler
) -> CasePredictive:
    """Map a model's posterior predictive at the evaluated windows to RUL units.

    Draws a mask excludes are dropped. Draw means map to ``a * mu_m + b`` and the
    shared variance to ``a**2 * s**2`` (``y_scaler`` is affine).

    Args:
        predictive: Draw means ``(M, T - 1, 1)`` and the aleatoric variance,
            broadcastable against ``(T - 1, 1)``, in scaled units.
        y_scaler: The target scaler (affine).

    Returns:
        The case's posterior predictive in RUL units.
    """
    draws = np.asarray(predictive.draws, dtype=np.float64)[..., 0]  # (M, T-1)
    if predictive.draw_mask is not None:
        draws = draws[np.asarray(predictive.draw_mask, dtype=bool)]
    variance = np.broadcast_to(
        np.asarray(predictive.aleatoric_variance, dtype=np.float64),
        (draws.shape[1], 1),
    )[:, 0]
    # Contiguous, like an array read back from the artifact: same summation order.
    scaled = CasePredictive(
        draw_means=np.ascontiguousarray(draws.T), aleatoric_variance=variance.copy()
    )
    return scaled.affine(*_unscale_affine(y_scaler))


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
) -> EvaluatedCase:
    """Build the posterior predictive of one test case.

    At each evaluated timestamp, M posterior draws are taken; no aleatoric noise is
    sampled. A finite posterior (pSGLD) yields all its retained draws instead of
    ``num_draws``. t0 gets no prediction.

    Args:
        model: The predictive model.
        history: The test case's rows (any order).
        reference_samples: Simulator reference samples ``(T, R)``.
        name: Test-case name.
        x_scaler: Input feature scaler.
        y_scaler: Target scaler (affine).
        parameters: Evaluation parameters.
        key: Key of this case's posterior draws.

    Returns:
        The evaluated case, in RUL units.

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
    predictive = model.predictive(
        jnp.asarray(windows, dtype=jnp.float32), key, parameters.num_draws
    )
    return EvaluatedCase(
        name=name,
        policy=policy,
        times=history["time"].to_numpy(dtype=np.float64)[indices],
        targets=history["rul"].to_numpy(dtype=np.float64)[indices],
        reference=ReferenceSamples(np.asarray(reference_samples, dtype=np.float64)),
        predictive=unscale_predictive(predictive, y_scaler),
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
        The evaluated cases, metric table and aggregates.
    """
    cases = [
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
    table = metric_table(cases, parameters.coverage_levels, parameters.wsu_level)
    return Evaluation(
        cases=cases,
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
    0. Each case's posterior predictive goes to the artifact
    ``evaluation/test/<case>/predictive`` (JSON, see ``EvaluatedCase.to_artifact``),
    and its RUL-over-time figure to
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
    for case in evaluation.cases:
        case_prefix = f"{EVALUATION_PREFIX}/{case.name}"
        tracker.log_artifact(f"{case_prefix}/predictive", case.to_artifact())
        tracker.log_artifact(
            f"{case_prefix}/rul_over_time", plot_rul_over_time(case, level=band_level)
        )
