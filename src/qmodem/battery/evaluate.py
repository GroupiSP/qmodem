"""Per-test-case evaluation driver of the comparison protocol (§6).

The pure core maps per-timestamp predictive samples and simulator reference samples
of each test case to a per-test-case metric table (``case_metrics``,
``metric_table``). Aggregation over test cases is a separate function
(``aggregate``). Every metric is a stagereg metric function; qmodem has no metric
code of its own.

The first timestamp of a test case (t0) has no prediction. It is dropped from every
metric and serves only as the time origin of WSU.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence

import numpy as np
from stagereg.core import coverage, expected_sample_crps, rmse, wsu

PRIMARY_COVERAGE_LEVEL = 0.95
"""The protocol's primary coverage level, also the band WSU integrates."""


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


def _band(samples: np.ndarray, level: float) -> tuple[np.ndarray, np.ndarray]:
    """Return the empirical central band of each row of ``samples``.

    Args:
        samples: Samples ``(U, M)``.
        level: Nominal coverage level in ``(0, 1)``.

    Returns:
        The lower and upper empirical percentiles ``(U,)`` each.
    """
    tail = 100 * (1 - level) / 2
    lower, upper = np.percentile(samples, [tail, 100 - tail], axis=1)
    return lower, upper


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
        metrics[coverage_key(level)] = coverage(targets, *_band(predictive, level))
    lower, upper = _band(predictive, wsu_level)
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
