from __future__ import annotations

import mlflow

from qmodem.train_base import BaseTrainingContext, TrainingPhase


def track_conv_weights_variance(
    phase: TrainingPhase, context: BaseTrainingContext
) -> None:
    if phase == TrainingPhase.EPOCH_END:
        mlflow.log_metric(
            "conv_weights_variance",
            context.model.conv_mean_posterior_variance(),
            step=context.epoch,
        )
