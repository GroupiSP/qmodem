from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from .train import Method


class ModelBuildParameters(BaseModel):
    """Adapter class to hold the parameters that define the data-driven model."""

    model_config = ConfigDict(extra="ignore")

    method: Method
    conv_n_filters: int
    conv_kernel_size: int
    dropout_rate: float
    activation_function: str
    net_init_seed: int
