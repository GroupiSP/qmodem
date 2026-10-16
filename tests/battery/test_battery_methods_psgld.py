"""PSGLD's chain schedule: burn-in and thinning in update steps, from epochs."""

from __future__ import annotations

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from qmodem.battery.methods.psgld import ChainSchedule, chain_schedule


def _retained_steps(schedule: ChainSchedule, total_steps: int) -> list[int]:
    """The steps whose positions stagereg's pSGLD retains (steps numbered from 1)."""
    return [
        step
        for step in range(1, total_steps + 1)
        if step > schedule.burn_in
        and (step - schedule.burn_in) % schedule.thinning == 0
    ]


def test_the_protocol_schedule_at_the_battery_size() -> None:
    # N = 7197 windows at batch 32 -> 225 batches per epoch.
    schedule = chain_schedule(
        epochs=300, burn_in_epochs=100, num_samples=100, steps_per_epoch=225
    )

    assert schedule == ChainSchedule(burn_in=22_500, thinning=450)


schedules = st.tuples(
    st.integers(2, 60),  # epochs
    st.integers(1, 300),  # steps per epoch
    st.integers(1, 120),  # samples
    st.data(),
)


@given(schedules)
def test_the_chain_retains_exactly_the_requested_samples(arguments) -> None:
    epochs, steps_per_epoch, num_samples, data = arguments
    burn_in_epochs = data.draw(st.integers(0, epochs - 1))
    assume((epochs - burn_in_epochs) * steps_per_epoch >= num_samples)
    total = epochs * steps_per_epoch

    schedule = chain_schedule(
        epochs=epochs,
        burn_in_epochs=burn_in_epochs,
        num_samples=num_samples,
        steps_per_epoch=steps_per_epoch,
    )
    retained = _retained_steps(schedule, total)

    assert len(retained) == num_samples
    assert retained[-1] == total  # the last epoch's last position closes the chain
    assert schedule.burn_in >= burn_in_epochs * steps_per_epoch


@given(st.integers(1, 50), st.integers(1, 50))
def test_a_chain_too_short_for_its_samples_is_refused(steps, extra) -> None:
    with pytest.raises(ValueError, match="samples"):
        chain_schedule(
            epochs=2,
            burn_in_epochs=1,
            num_samples=steps + extra,
            steps_per_epoch=steps,
        )
