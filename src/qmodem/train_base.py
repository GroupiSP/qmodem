from __future__ import annotations

import dataclasses
from enum import Enum, auto
from typing import Callable

import flax.nnx as nnx


class TrainingPhase(Enum):
    INIT = auto()
    EPOCH_START = auto()
    EVAL_START = auto()
    EPOCH_END = auto()
    BEFORE_RETURN = auto()


@dataclasses.dataclass
class BaseTrainingContext:
    epoch: int
    val_loss: float
    best_val_loss: float
    model: nnx.Module  # Replace with the actual model type if known
    model_best_state: nnx.State  # Replace with the actual model state type if known


type Callback = Callable[[TrainingPhase, BaseTrainingContext], None]
