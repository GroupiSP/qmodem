"""The AVI methods on the battery CNN: Q-AVI and, by its latent map only, C-AVI.

Both AVI methods share everything but the latent map: the hyperparameters
(``AVIHyperparameters``), the conv-layer generator's postprocessor, the
discriminator and the AVI inference method. A method's configuration only names its
latent-map builder and calls ``avi_posterior_spec``.

Every AVI method run logs both latent maps' parameter counts (``2Lq`` for the
circuit, ``h(q + 2) + q`` for the matched MLP), so the parameter match is explicit.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

import jax
import optax
from flax import nnx
from pydantic import BaseModel, ConfigDict, Field

from qmodem.avi import (
    AVI,
    ClassicalLatentMap,
    LatentMap,
    QuantumLatentMap,
    matched_hidden_width,
)

from ..models import ConvGenerator, WindowDiscriminator
from ..posterior import PosteriorSpec, PosteriorStageHyperparameters, PyTree
from ..train import Method, OptimizerHyperparameters
from .base import MethodConfigBase


class AVIHyperparameters(BaseModel):
    """The posterior hyperparameters shared by the AVI methods (protocol §5).

    Attributes:
        adversarial_weight: The adversarial weight λ (searched 1e-4 to 1, log).
        num_draws: Whole-function draws S per step (searched in {1, 4, 8, 16}).
        discriminator_steps: Discriminator steps k per generator step (searched
            in {1, 2, 3, 5}).
        n_qubits: The latent width q (searched 3 to 6).
        n_layers: The circuit layers L (searched 1 to 4).
        generator: Adam for the generator (LR searched 1e-5 to 1e-2, log).
        discriminator: Adam for the discriminator (LR searched 1e-5 to 1e-2, log).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    adversarial_weight: float = Field(default=0.1, gt=0.0)
    num_draws: int = Field(default=1, ge=1)
    discriminator_steps: int = Field(default=1, ge=1)
    n_qubits: int = Field(default=5, ge=2)
    n_layers: int = Field(default=1, ge=1)
    generator: OptimizerHyperparameters = OptimizerHyperparameters()
    discriminator: OptimizerHyperparameters = OptimizerHyperparameters()


LATENT_MAP_QUANTUM_PARAMETERS = "posterior/latent_map/quantum_parameters"
"""Metric: the parameter count ``2Lq`` of the circuit latent map (Q-AVI)."""

LATENT_MAP_CLASSICAL_PARAMETERS = "posterior/latent_map/classical_parameters"
"""Metric: the parameter count ``h(q + 2) + q`` of the matched MLP map (C-AVI)."""


type LatentMapBuilder = Callable[[AVIHyperparameters, nnx.Rngs], LatentMap]
"""Builds an AVI method's latent map from its hyperparameters and RNGs."""


def quantum_latent_map(hp: AVIHyperparameters, rngs: nnx.Rngs) -> LatentMap:
    """Build Q-AVI's latent map: a ``q``-qubit, ``L``-layer circuit.

    Args:
        hp: The AVI hyperparameters (``n_qubits``, ``n_layers``).
        rngs: RNGs of the circuit angles.

    Returns:
        The quantum latent map.
    """
    return QuantumLatentMap(hp.n_qubits, hp.n_layers, rngs=rngs)


def classical_latent_map(hp: AVIHyperparameters, rngs: nnx.Rngs) -> LatentMap:
    """Build C-AVI's latent map: a ``1 → h → q`` tanh MLP matched to the circuit.

    Args:
        hp: The AVI hyperparameters (``n_qubits`` is the width ``q``; ``n_layers``
            is the ``L`` of the circuit whose parameter count is matched).
        rngs: RNGs of the MLP's (Flax default) initializers.

    Returns:
        The classical latent map, with ``h = matched_hidden_width(q, L)``.
    """
    return ClassicalLatentMap(
        hp.n_qubits, matched_hidden_width(hp.n_qubits, hp.n_layers), rngs=rngs
    )


def latent_map_parameter_counts(hp: AVIHyperparameters) -> dict[str, float]:
    """Return both latent maps' parameter counts, as logged by an AVI method run.

    Args:
        hp: The AVI hyperparameters (``n_qubits``, ``n_layers``).

    Returns:
        ``LATENT_MAP_QUANTUM_PARAMETERS`` (``2Lq``) and
        ``LATENT_MAP_CLASSICAL_PARAMETERS`` (``h(q + 2) + q``).
    """
    q, n_layers = hp.n_qubits, hp.n_layers
    hidden = matched_hidden_width(q, n_layers)
    return {
        LATENT_MAP_QUANTUM_PARAMETERS: float(2 * n_layers * q),
        LATENT_MAP_CLASSICAL_PARAMETERS: float(hidden * (q + 2) + q),
    }


def avi_posterior_spec(
    *,
    hp: AVIHyperparameters,
    stage: PosteriorStageHyperparameters,
    build_latent_map: LatentMapBuilder,
) -> PosteriorSpec:
    """Return an AVI method's plug-in into the shared posterior stage.

    Args:
        hp: The AVI hyperparameters.
        stage: The shared stage schedule.
        build_latent_map: Builds the method's latent map.

    Returns:
        The posterior specification: early-stopped, on a clone of the mean net,
        counting the generator as the posterior and the discriminator as
        scaffolding, and logging both latent maps' parameter counts.
    """

    def build_generator(warm_start: PyTree, key: jax.Array) -> ConvGenerator:
        """Build the conv generator with the method's latent map.

        Args:
            warm_start: The inferred conv parameters (the generator's offset).
            key: Initializes the latent map and the postprocessor.

        Returns:
            The generator.
        """
        rngs = nnx.Rngs(key)
        return ConvGenerator(build_latent_map(hp, rngs), warm_start, rngs=rngs)

    def build_method(key: jax.Array) -> AVI:
        """Build a fresh AVI inference method.

        Args:
            key: The posterior-stage initialization key.

        Returns:
            The AVI method.
        """
        return AVI(
            build_generator=build_generator,
            build_discriminator=lambda k: WindowDiscriminator(rngs=nnx.Rngs(k)),
            key=key,
            generator_optimizer=optax.adam(hp.generator.learning_rate),
            discriminator_optimizer=optax.adam(hp.discriminator.learning_rate),
            adversarial_weight=hp.adversarial_weight,
            num_draws=hp.num_draws,
            discriminator_steps=hp.discriminator_steps,
        )

    return PosteriorSpec(
        stage=stage,
        build_method=build_method,
        trainable_parameters=_generator_parameters,
        scaffolding_parameters=_discriminator_parameters,
        method_metrics=latent_map_parameter_counts(hp),
    )


def _generator_parameters(method: AVI) -> PyTree:
    """Return the posterior an AVI method trains: the generator's parameters.

    Args:
        method: An initialized ``AVI``.

    Returns:
        The generator's ``nnx.Param`` state.
    """
    return method.state()["generator"]


def _discriminator_parameters(method: AVI) -> PyTree:
    """Return an AVI method's scaffolding: the discriminator's parameters.

    The discriminator is trained alongside the generator but is not part of the
    posterior (US 51), so T3 counts it separately.

    Args:
        method: An initialized ``AVI``.

    Returns:
        The discriminator's ``nnx.Param`` state.
    """
    return method.state()["discriminator"]


class QAVIConfig(MethodConfigBase):
    """Q-AVI: an AVI generator whose latent map is a parametrized quantum circuit.

    Attributes:
        avi: The AVI hyperparameters.
        stage: The shared posterior-stage schedule.
    """

    method: ClassVar[Method] = Method.Q_AVI

    avi: AVIHyperparameters = AVIHyperparameters()
    stage: PosteriorStageHyperparameters = PosteriorStageHyperparameters()

    def posterior_hyperparameters(self) -> dict[str, Any]:
        """Return the Q-AVI posterior hyperparameters, nested.

        Returns:
            ``avi`` and ``stage``.
        """
        return self.model_dump()

    def posterior_spec(self) -> PosteriorSpec:
        """Return Q-AVI's plug-in: AVI with the quantum latent map.

        Returns:
            The posterior specification.
        """
        return avi_posterior_spec(
            hp=self.avi, stage=self.stage, build_latent_map=quantum_latent_map
        )


class CAVIConfig(MethodConfigBase):
    """C-AVI: Q-AVI with a classical tanh MLP latent map of matched parameter count.

    Attributes:
        avi: The AVI hyperparameters (``n_layers`` sets the matched hidden width).
        stage: The shared posterior-stage schedule.
    """

    method: ClassVar[Method] = Method.C_AVI

    avi: AVIHyperparameters = AVIHyperparameters()
    stage: PosteriorStageHyperparameters = PosteriorStageHyperparameters()

    def posterior_hyperparameters(self) -> dict[str, Any]:
        """Return the C-AVI posterior hyperparameters, nested.

        Returns:
            ``avi`` and ``stage``.
        """
        return self.model_dump()

    def posterior_spec(self) -> PosteriorSpec:
        """Return C-AVI's plug-in: AVI with the classical latent map.

        Returns:
            The posterior specification.
        """
        return avi_posterior_spec(
            hp=self.avi, stage=self.stage, build_latent_map=classical_latent_map
        )
