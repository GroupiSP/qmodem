"""HNN: the reference method without posterior stage."""

from __future__ import annotations

from typing import Any, ClassVar

from ..posterior import PosteriorSpec
from ..train import Method
from .base import MethodConfigBase


class HNNConfig(MethodConfigBase):
    """HNN: no posterior stage; its predictive is the backbone's mean and variance."""

    method: ClassVar[Method] = Method.HNN

    def posterior_hyperparameters(self) -> dict[str, Any]:
        """Return the (empty) posterior hyperparameters of HNN.

        Returns:
            An empty mapping: HNN has no posterior stage.
        """
        return self.model_dump()

    def posterior_spec(self) -> PosteriorSpec | None:
        """Return no posterior specification: HNN trains nothing.

        Returns:
            ``None``.
        """
        return None
