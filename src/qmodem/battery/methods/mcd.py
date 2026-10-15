"""MCD: stagereg's ``MCDropout`` on the conv layer of the input-dropout network."""

from __future__ import annotations

from typing import Any, ClassVar

import jax
import optax
from flax import nnx
from pydantic import Field
from stagereg.jax import InferenceMethod, MCDropout

from ..models import MCDropoutNet
from ..posterior import PosteriorSpec, PosteriorStageHyperparameters, PyTree
from ..train import Method, MeanNetworkHyperparameters, OptimizerHyperparameters
from .base import MethodConfigBase


def build_mcd_net(
    network: MeanNetworkHyperparameters, dropout_rate: float, key: jax.Array
) -> MCDropoutNet:
    """Build the MCD posterior network with the backbone's sizes.

    Args:
        network: Backbone sizes, taken from the mean network's configuration.
        dropout_rate: Probability of dropping an input element.
        key: Key of the parameter initializers and of the dropout stream.

    Returns:
        A freshly initialized MCD posterior network.
    """
    return MCDropoutNet(
        kernel_size=network.kernel_size,
        n_filters=network.n_filters,
        activation=getattr(nnx, network.activation),
        dropout_rate=dropout_rate,
        rngs=nnx.Rngs(key),
    )


class MCDConfig(MethodConfigBase):
    """MCD: the mean network plus input dropout, fine-tuned with dropout active.

    Attributes:
        optimizer: Adam at a constant learning rate (searched 1e-5 to 1e-2).
        dropout_rate: Input dropout rate (searched 0.01 to 0.5).
        stage: The shared posterior-stage schedule.
    """

    method: ClassVar[Method] = Method.MCD

    optimizer: OptimizerHyperparameters = OptimizerHyperparameters()
    dropout_rate: float = Field(default=0.1, gt=0.0, lt=1.0)
    stage: PosteriorStageHyperparameters = PosteriorStageHyperparameters()

    def posterior_hyperparameters(self) -> dict[str, Any]:
        """Return the MCD posterior hyperparameters, nested.

        Returns:
            ``optimizer``, ``dropout_rate`` and ``stage``.
        """
        return self.model_dump()

    def posterior_spec(self) -> PosteriorSpec:
        """Return MCD's plug-in: ``MCDropout``, early-stopped, input-dropout net.

        Returns:
            The posterior specification.
        """
        return PosteriorSpec(
            stage=self.stage,
            build_method=self._build_method,
            trainable_parameters=_fine_tuned_parameters,
            build_network=self._build_network,
        )

    def _build_network(
        self, network: MeanNetworkHyperparameters, key: jax.Array
    ) -> nnx.Module:
        """Build the posterior network (its Params are then warm-started).

        Args:
            network: The backbone sizes.
            key: Initialization and dropout-stream key.

        Returns:
            The MCD posterior network.
        """
        return build_mcd_net(network, self.dropout_rate, key)

    def _build_method(self, key: jax.Array) -> InferenceMethod:
        """Build MCD (deterministic: it warm-starts from the mean's conv layer).

        Args:
            key: Unused.

        Returns:
            A fresh ``MCDropout``.
        """
        del key
        return MCDropout(optimizer=optax.adam(self.optimizer.learning_rate))


def _fine_tuned_parameters(method: InferenceMethod) -> PyTree:
    """Return MCD's trained parameters: the fine-tuned conv layer.

    Args:
        method: An ``MCDropout``.

    Returns:
        The inferred subnetwork's parameters.
    """
    return method.state()["parameters"]
