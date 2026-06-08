from __future__ import annotations

import dataclasses

DATA_GEN_RUN_ID = "db5ed872d78448dba887040cf49a74d1"


@dataclasses.dataclass
class TrainHyperparameters:
    batch_size: int = 32
    window_size: int = 20
    stride: int = 1
    normalize_rul: bool = True
    sampler_seeds: tuple[int, int] = (42, 0)
    net_init_seed: int = 0
    train_rng_seed: int = 1
    drop_remainder: bool = False
    learning_rate: float = 1e-2
    n_epochs: int = 500
    beta_nll: float = 0.0
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4
    scheduler_alpha: float = 0.1
