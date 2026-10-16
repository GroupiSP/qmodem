"""pSGLD: stagereg's ``PreconditionedSGLD`` on the conv layer.

The protocol fixes the chain in epochs (300 epochs, 100 of them burn-in) and its
posterior in draws (M = 100 retained samples). stagereg's pSGLD counts burn-in and
thinning in update steps (training batches), so the schedule is derived from the
number of batches per epoch of the run's training loader (``chain_schedule``).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any, ClassVar

import jax
from pydantic import Field, model_validator
from stagereg.core import Batch
from stagereg.jax import (
    InferenceMethod,
    PosteriorProblem,
    PreconditionedSGLD,
    RetainedSamplesPosterior,
)

from ..posterior import PosteriorSpec, PosteriorStageHyperparameters, PyTree
from ..train import Method
from .base import MethodConfigBase

DECAY = 0.99
"""RMSprop decay of the preconditioner (fixed by the protocol)."""

STABILITY = 1e-5
"""Added to the preconditioner's square root (fixed by the protocol)."""


@dataclasses.dataclass(frozen=True)
class ChainSchedule:
    """A pSGLD chain's burn-in and thinning, in update steps.

    Attributes:
        burn_in: Update steps whose positions are discarded.
        thinning: Retain one position every ``thinning`` steps after burn-in.
    """

    burn_in: int
    thinning: int


def chain_schedule(
    *, epochs: int, burn_in_epochs: int, num_samples: int, steps_per_epoch: int
) -> ChainSchedule:
    """Derive the step schedule that retains exactly ``num_samples`` positions.

    The thinning spreads the samples evenly over the epochs after burn-in. The
    steps left over by the integer thinning are added to the burn-in, so the
    chain's last position (the last epoch's last step) is always retained and
    exactly ``num_samples`` positions are, for any number of steps per epoch.

    Args:
        epochs: The chain's total number of epochs.
        burn_in_epochs: The epochs whose positions are at least discarded.
        num_samples: The number of positions to retain (M).
        steps_per_epoch: Training batches per epoch.

    Returns:
        The schedule: ``burn_in >= burn_in_epochs * steps_per_epoch`` and
        ``epochs * steps_per_epoch - burn_in == num_samples * thinning``.

    Raises:
        ValueError: If the epochs after burn-in have fewer steps than samples,
            or the arguments are out of range.
    """
    if not 0 <= burn_in_epochs < epochs:
        raise ValueError("burn_in_epochs must be in [0, epochs).")
    if num_samples < 1 or steps_per_epoch < 1:
        raise ValueError("num_samples and steps_per_epoch must be positive.")
    sampling_steps = (epochs - burn_in_epochs) * steps_per_epoch
    thinning = sampling_steps // num_samples
    if thinning < 1:
        raise ValueError(
            f"{sampling_steps} steps after burn-in cannot retain {num_samples} samples."
        )
    return ChainSchedule(
        burn_in=epochs * steps_per_epoch - num_samples * thinning, thinning=thinning
    )


@dataclasses.dataclass
class EpochScheduledSGLD:
    """Stagereg's pSGLD with its burn-in and thinning given in epochs and samples.

    The step schedule needs the number of training batches per epoch, which only
    the posterior problem's train loader knows. It is therefore bound before the
    first update (or ``load_state``), from ``train_loader(split="train", epoch=0)``.
    A restored method never updates: loading retained samples needs no schedule.

    Attributes:
        step_size: The Langevin step size.
        epochs: The chain's total number of epochs (the stage's epochs).
        burn_in_epochs: Epochs discarded at least as burn-in.
        num_samples: The positions to retain (M).
    """

    step_size: float
    epochs: int
    burn_in_epochs: int
    num_samples: int
    _problem: PosteriorProblem | None = dataclasses.field(
        default=None, init=False, repr=False
    )
    _sampler: PreconditionedSGLD | None = dataclasses.field(
        default=None, init=False, repr=False
    )
    _schedule: ChainSchedule | None = dataclasses.field(
        default=None, init=False, repr=False
    )

    @property
    def supports_early_stopping(self) -> bool:
        """``False``: an early stopper would truncate the chain."""
        return False

    @property
    def diagnostics(self) -> tuple[str, ...]:
        """``()``: the stage records only the loss."""
        return ()

    @property
    def has_posterior(self) -> bool:
        """Whether at least one sample has been retained."""
        return self._sampler is not None and self._sampler.has_posterior

    @property
    def schedule(self) -> ChainSchedule | None:
        """The bound step schedule, ``None`` before training."""
        return self._schedule

    def init(self, problem: PosteriorProblem) -> None:
        """Start a chain at the problem's warm start; the schedule is unbound.

        Args:
            problem: The posterior problem.
        """
        self._problem = problem
        self._schedule = None
        self._sampler = self._new_sampler(ChainSchedule(burn_in=0, thinning=1))

    def update(self, batch: Batch, key: jax.Array) -> Mapping[str, Any]:
        """Take one Langevin step, binding the schedule before the first.

        Args:
            batch: The training batch.
            key: The step's PRNG key.

        Returns:
            ``{"loss": U}``, the negative log posterior before the step.
        """
        self._bind_schedule()
        return self._require_sampler().update(batch, key)

    def posterior(self) -> RetainedSamplesPosterior:
        """Return the posterior of the samples retained so far.

        Returns:
            The finite posterior.
        """
        return self._require_sampler().posterior()

    def state(self) -> PyTree:
        """Return the chain's full state.

        Returns:
            Position, preconditioner, step count and retained samples.
        """
        return self._require_sampler().state()

    def load_state(self, state: PyTree) -> None:
        """Resume the chain, binding the schedule first.

        Args:
            state: A state returned by ``state``.
        """
        self._bind_schedule()
        self._require_sampler().load_state(state)

    def posterior_state(self) -> PyTree:
        """Return the retained samples, stacked.

        Returns:
            The stacked samples.
        """
        return self._require_sampler().posterior_state()

    def load_posterior(self, state: PyTree) -> None:
        """Load retained samples after ``init``.

        Args:
            state: Stacked samples returned by ``posterior_state``.
        """
        self._require_sampler().load_posterior(state)

    def _bind_schedule(self) -> None:
        """Derive the step schedule from the train loader and restart the chain.

        Called before the first step, so no position is lost.

        Raises:
            ValueError: If the method was not initialized.
        """
        if self._schedule is not None:
            return
        if self._problem is None:
            raise ValueError("call init before update.")
        batches = self._problem.train_loader(split="train", epoch=0)
        self._schedule = chain_schedule(
            epochs=self.epochs,
            burn_in_epochs=self.burn_in_epochs,
            num_samples=self.num_samples,
            steps_per_epoch=len(list(batches)),
        )
        self._sampler = self._new_sampler(self._schedule)

    def _new_sampler(self, schedule: ChainSchedule) -> PreconditionedSGLD:
        """Build stagereg's pSGLD under a schedule, initialized on the problem.

        Args:
            schedule: The step schedule.

        Returns:
            The sampler.
        """
        sampler = PreconditionedSGLD(
            step_size=self.step_size,
            burn_in=schedule.burn_in,
            thinning=schedule.thinning,
            decay=DECAY,
            stability=STABILITY,
        )
        if self._problem is not None:
            sampler.init(self._problem)
        return sampler

    def _require_sampler(self) -> PreconditionedSGLD:
        """Return the sampler, refusing an uninitialized method.

        Returns:
            The sampler.

        Raises:
            ValueError: If the method was not initialized.
        """
        if self._sampler is None:
            raise ValueError("call init first.")
        return self._sampler


class PSGLDConfig(MethodConfigBase):
    """pSGLD: a Langevin chain over the conv layer, fixed length, no early stopping.

    Attributes:
        step_size: The Langevin step size (searched 1e-6 to 1e-2, log).
        burn_in_epochs: Epochs discarded as burn-in (protocol: 100 of 300).
        num_samples: Retained samples M, the posterior's draws (protocol: 100).
        stage: The shared posterior-stage schedule; ``max_epochs`` is the fixed
            chain length (protocol: 300), ``patience`` is unused.
    """

    method: ClassVar[Method] = Method.PSGLD

    step_size: float = Field(default=1e-4, gt=0.0)
    burn_in_epochs: int = Field(default=100, ge=0)
    num_samples: int = Field(default=100, ge=1)
    stage: PosteriorStageHyperparameters = PosteriorStageHyperparameters()

    @model_validator(mode="after")
    def _burn_in_shorter_than_the_chain(self) -> PSGLDConfig:
        """Refuse a burn-in that leaves no epoch to sample.

        Returns:
            The configuration.

        Raises:
            ValueError: If ``burn_in_epochs >= stage.max_epochs``.
        """
        if self.burn_in_epochs >= self.stage.max_epochs:
            raise ValueError(
                f"burn_in_epochs ({self.burn_in_epochs}) must be shorter than "
                f"stage.max_epochs ({self.stage.max_epochs})."
            )
        return self

    def posterior_hyperparameters(self) -> dict[str, Any]:
        """Return the pSGLD posterior hyperparameters, nested.

        Returns:
            ``step_size``, ``burn_in_epochs``, ``num_samples`` and ``stage``.
        """
        return self.model_dump()

    def posterior_spec(self) -> PosteriorSpec:
        """Return pSGLD's plug-in: a fixed-length chain, last epoch finalized.

        Returns:
            The posterior specification.
        """
        return PosteriorSpec(
            stage=self.stage,
            build_method=self._build_method,
            trainable_parameters=_chain_position,
            early_stopping=False,
        )

    def _build_method(self, key: jax.Array) -> InferenceMethod:
        """Build the chain (deterministic: it warm-starts from the mean's conv).

        Args:
            key: Unused; the chain's noise comes from the training state's key.

        Returns:
            A fresh ``EpochScheduledSGLD``.
        """
        del key
        return EpochScheduledSGLD(
            step_size=self.step_size,
            epochs=self.stage.max_epochs,
            burn_in_epochs=self.burn_in_epochs,
            num_samples=self.num_samples,
        )


def _chain_position(method: InferenceMethod) -> PyTree:
    """Return pSGLD's trained parameters: the chain's position (the conv layer).

    Args:
        method: An ``EpochScheduledSGLD``.

    Returns:
        The position over the inferred subnetwork.
    """
    return method.state()["parameters"]
