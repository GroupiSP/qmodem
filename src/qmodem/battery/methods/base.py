"""The configuration protocol every method implements."""

from __future__ import annotations

from typing import Any, ClassVar, Protocol

from pydantic import BaseModel, ConfigDict

from ..posterior import PosteriorSpec
from ..train import Method


class MethodConfig(Protocol):
    """The per-method configuration of a method run.

    Each implementation is a frozen data container of the method's posterior
    hyperparameters, registered in ``METHOD_CONFIGS``.
    """

    method: ClassVar[Method]

    def posterior_hyperparameters(self) -> dict[str, Any]:
        """Return the posterior hyperparameters, nested, logged under ``posterior``.

        ``method_config(method, posterior_hyperparameters())`` must rebuild the
        configuration: it is how a later process restores the method run.

        Returns:
            The hyperparameters (empty for a method without posterior stage).
        """
        ...

    def posterior_spec(self) -> PosteriorSpec | None:
        """Return how the method plugs into the shared posterior stage.

        Returns:
            The posterior specification, ``None`` for a method without posterior
            stage (HNN).
        """
        ...


class MethodConfigBase(BaseModel):
    """Pydantic base of the method configurations: frozen, no extra fields."""

    model_config = ConfigDict(frozen=True, extra="forbid")
