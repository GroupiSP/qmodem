from __future__ import annotations

import dataclasses
from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np

from qmodem.metrics import crps as _crps


@dataclasses.dataclass(frozen=True)
class EvalTimeStamp:
    time: float
    target: np.ndarray
    samples_true: np.ndarray
    samples_pred: np.ndarray

    @property
    def average_pred(self) -> float:
        return np.mean(self.samples_pred)

    @property
    def ci_95_true(self) -> np.ndarray:
        """NOTE: could be private, but it is kept here for symmetry with the predicted CI."""
        return np.percentile(self.samples_true, [2.5, 97.5])

    @property
    def ci_95_pred(self) -> np.ndarray:
        return np.percentile(self.samples_pred, [2.5, 97.5])

    @property
    def squared_error(self) -> float:
        return (self.target - self.average_pred) ** 2

    @property
    def is_covered(self) -> bool:
        lower_bound_pred, upper_bound_pred = self.ci_95_pred
        return lower_bound_pred <= self.target <= upper_bound_pred

    def crps(self, x_grid: np.ndarray) -> float:
        return _crps(
            samples_true=self.samples_true,
            samples_pred=self.samples_pred,
            x_grid=x_grid,
        )


@dataclasses.dataclass
class TestCaseResults:
    id: int
    eval_time_stamps: list[EvalTimeStamp]

    def __post_init__(self) -> None:
        self._times: np.ndarray = np.array([ets.time for ets in self.eval_time_stamps])
        self._period: float = self._times[-1] - self._times[0]

    @property
    def squared_errors(self) -> np.ndarray:
        return np.array([ets.squared_error for ets in self.eval_time_stamps[1:]])

    @property
    def coverage(self) -> float:
        return np.mean([ets.is_covered for ets in self.eval_time_stamps[1:]])

    @property
    def wsu(self) -> float:
        return (
            np.dot(
                (
                    np.array(
                        [
                            (ets_t.ci_95_pred[1] + ets_t1.ci_95_pred[1]) / 2
                            for (ets_t, ets_t1) in zip(
                                self.eval_time_stamps[2:], self.eval_time_stamps[1:-1]
                            )
                        ]
                    )
                    - np.array(
                        [
                            (ets_t.ci_95_pred[0] + ets_t1.ci_95_pred[0]) / 2
                            for (ets_t, ets_t1) in zip(
                                self.eval_time_stamps[2:], self.eval_time_stamps[1:-1]
                            )
                        ]
                    )
                ),
                self._times[1:-1] - self._times[0],
            )
            / self._period**2
        )

    @property
    def rmse(self) -> float:
        return np.sqrt(np.mean(self.squared_errors))

    def average_crps(self, x_grid: np.ndarray) -> float:
        return np.mean([ets.crps(x_grid=x_grid) for ets in self.eval_time_stamps])

    def plot_rul_over_time(self, ax: plt.Axes, legend: bool = True) -> None:
        ax.plot(
            self._times,
            [rt for rt in [ets.target for ets in self.eval_time_stamps]],
            label="True RUL",
        )
        ax.fill_between(
            self._times,
            [ets.ci_95_true[0] for ets in self.eval_time_stamps],
            [ets.ci_95_true[1] for ets in self.eval_time_stamps],
            alpha=0.3,
            label="True RUL CI",
        )
        ax.plot(
            self._times[1:],
            [ets.average_pred for ets in self.eval_time_stamps[1:]],
            "-o",
            label="Predicted RUL",
        )
        ax.fill_between(
            self._times[1:],
            [ets.ci_95_pred[0] for ets in self.eval_time_stamps[1:]],
            [ets.ci_95_pred[1] for ets in self.eval_time_stamps[1:]],
            alpha=0.3,
            label="Predicted RUL CI",
        )
        ax.set_title(f"Test Case {self.id}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("RUL (s)")
        ax.set_ylim(bottom=0)
        ax.grid()

        if legend:
            ax.legend()


def bar_plot_metrics_per_test_case(
    axes: Iterable[plt.Axes],
    test_case_results: list[TestCaseResults],
    rul_grid_crps: np.ndarray,
) -> None:
    """`axes` is expected to contain 4 subplot axes."""
    test_case_ids = [tcr.id for tcr in test_case_results]
    rmses = [tcr.rmse for tcr in test_case_results]
    coverages = [tcr.coverage for tcr in test_case_results]
    wsus = [tcr.wsu for tcr in test_case_results]
    average_crpss = [
        tcr.average_crps(x_grid=rul_grid_crps) for tcr in test_case_results
    ]

    metrics = {
        "RMSE": rmses,
        "Coverage": coverages,
        "WSU": wsus,
        "CRPS": average_crpss,
    }

    x = np.arange(len(test_case_ids))
    for ax, metric in zip(axes, metrics.keys()):
        ax.bar(x, metrics[metric])
        ax.set_xticks(x)
        ax.set_xticklabels(test_case_ids, rotation=45)
        ax.set_xlabel("Test Case ID")
        ax.set_title(metric)
        ax.grid()
