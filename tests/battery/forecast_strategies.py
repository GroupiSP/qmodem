"""Shared hypothesis strategies: posterior predictives and evaluated test cases.

``evaluated_cases`` builds on ``case_predictives``, so every test module draws its
cases from the same distribution family.
"""

from __future__ import annotations

import numpy as np
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from qmodem.battery.evaluate import EvaluatedCase
from qmodem.battery.forecasts import CasePredictive, ReferenceSamples

FINITE = st.floats(-50.0, 50.0, allow_nan=False, width=32)
"""Finite RUL values, exactly representable in float32."""

LEVELS = st.floats(0.01, 0.99)
"""Nominal coverage levels."""


@st.composite
def case_predictives(
    draw: st.DrawFn,
    *,
    n_times: int | None = None,
    elements: st.SearchStrategy[float] = FINITE,
) -> CasePredictive:
    """Draw a posterior predictive with 1-8 draws.

    Args:
        draw: Hypothesis' draw function.
        n_times: Number of evaluated timestamps; 1-5 if ``None``.
        elements: Strategy of the draw means.

    Returns:
        The posterior predictive.
    """
    rows = draw(st.integers(1, 5)) if n_times is None else n_times
    n_draws = draw(st.integers(1, 8))
    means = draw(arrays(np.float64, (rows, n_draws), elements=elements))
    variance = draw(
        arrays(np.float64, (rows,), elements=st.floats(0.01, 25.0, allow_nan=False))
    )
    return CasePredictive(draw_means=means, aleatoric_variance=variance)


@st.composite
def evaluated_cases(
    draw: st.DrawFn, *, name: str = "case_0", policy: int = 0
) -> EvaluatedCase:
    """Draw a test case with t0 plus at least two evaluated timestamps.

    Args:
        draw: Hypothesis' draw function.
        name: Test-case name.
        policy: Current-policy ID.

    Returns:
        The evaluated case.
    """
    n_times = draw(st.integers(3, 7))
    n_ref = draw(st.integers(2, 12))
    steps = draw(st.lists(st.floats(0.5, 20.0), min_size=n_times, max_size=n_times))
    return EvaluatedCase(
        name=name,
        policy=policy,
        times=np.cumsum(steps),
        targets=draw(arrays(np.float64, (n_times,), elements=FINITE)),
        reference=ReferenceSamples(
            draw(arrays(np.float64, (n_times, n_ref), elements=FINITE))
        ),
        predictive=draw(case_predictives(n_times=n_times - 1)),
    )
