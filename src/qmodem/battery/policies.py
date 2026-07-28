from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import simbat as sb


class ConstantDischargeCurrentPolicy:
    def __init__(self, current_value: float) -> None:
        self.current_value = current_value

    def __call__(self, soc: float, t: float) -> float:
        """Returns the constant current value for the given SoC values."""
        return self.current_value


class VariableDischargeCurrentPolicy:
    def __init__(self, current_values: list[float], time_values: list[float]) -> None:
        self.current_values = current_values
        self.time_values = time_values

    def __call__(self, soc: float, t: float) -> float:
        """Returns the current values at time `t` for the given SoC values."""
        for i in range(len(self.time_values) - 1):
            if self.time_values[i] <= t < self.time_values[i + 1]:
                return self.current_values[i]
        return self.current_values[
            -1
        ]  # Return the last current value if t exceeds the last time value


def plot_current_profile(
    ax: plt.Axes,
    policy: sb.simulate.DischargePolicyTemplate,
    t_grid: np.ndarray = np.linspace(0, 10_000, 100),
) -> None:
    """Plots the current profile for a given discharge policy.

    Args:
        ax: The matplotlib Axes object to plot on.
        policy: The discharge policy to visualize.
        t_grid: The time grid over which to evaluate the policy.
    """
    current_values = [policy(soc=None, t=t) for t in t_grid]
    ax.plot(t_grid, current_values)
    ax.set_xlabel("Time")
    ax.set_ylabel("Current")
    ax.grid()
