"""BBB: stagereg's ``BayesByBackprop`` on the conv layer."""

from __future__ import annotations

from typing import Any, ClassVar

import jax
import optax
from pydantic import Field
from stagereg.jax import BayesByBackprop, InferenceMethod

from ..posterior import PosteriorSpec, PosteriorStageHyperparameters, PyTree
from ..train import Method, OptimizerHyperparameters
from .base import MethodConfigBase


class BBBConfig(MethodConfigBase):
    """BBB: a mean-field Gaussian posterior fitted by the weight-space ELBO.

    Attributes:
        optimizer: Adam at a constant learning rate (searched 1e-5 to 1e-2).
        initial_std: Initial standard deviation of every variational factor
            (searched 1e-4 to 1e-1).
        stage: The shared posterior-stage schedule.
    """

    method: ClassVar[Method] = Method.BBB

    optimizer: OptimizerHyperparameters = OptimizerHyperparameters()
    initial_std: float = Field(default=1e-3, gt=0.0)
    stage: PosteriorStageHyperparameters = PosteriorStageHyperparameters()

    def posterior_hyperparameters(self) -> dict[str, Any]:
        """Return the BBB posterior hyperparameters, nested.

        Returns:
            ``optimizer``, ``initial_std`` and ``stage``.
        """
        return self.model_dump()

    def posterior_spec(self) -> PosteriorSpec:
        """Return BBB's plug-in: ``BayesByBackprop``, early-stopped, mean net.

        Returns:
            The posterior specification.
        """
        return PosteriorSpec(
            stage=self.stage,
            build_method=self._build_method,
            trainable_parameters=_variational_parameters,
        )

    def _build_method(self, key: jax.Array) -> InferenceMethod:
        """Build BBB (deterministic: it warm-starts from the mean's conv layer).

        Args:
            key: Unused.

        Returns:
            A fresh ``BayesByBackprop``.
        """
        del key
        return BayesByBackprop(
            optimizer=optax.adam(self.optimizer.learning_rate),
            initial_std=self.initial_std,
        )


def _variational_parameters(method: InferenceMethod) -> PyTree:
    """Return BBB's trained parameters: the variational means and rhos.

    Args:
        method: A ``BayesByBackprop``.

    Returns:
        ``{"means": ..., "rhos": ...}`` over the inferred subnetwork.
    """
    return method.state()["variational"]
