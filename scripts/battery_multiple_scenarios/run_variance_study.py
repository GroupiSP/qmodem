"""Run the variance-stage HPO study on seed 0's frozen mean checkpoint (protocol §5).

The study tunes only the variance learning rate (1e-4–1e-2, log). Each trial
restores the selected backbone run's ``mean/best`` checkpoint, trains the variance
stage with the backbone's network sizes, batch size, cap and patience, and
minimizes its best-epoch validation Gamma NLL. TPE with a fixed sampler seed
proposes the trials, and a median pruner stops poor ones on the per-epoch
validation NLL.

MLflow layout: one top-level ``role=hpo_study`` run tagged ``stage=variance``,
``backbone_run_id`` and ``n_trials``, logging the selected params as
``best.<name>`` (``best.variance/optimizer/learning_rate``), with one
``role=hpo_trial`` child per trial (tag ``hpo.trial=<n>``) holding only its
``checkpoints/variance/best``.

The backbone run is selected by ``--backbone-run-id`` or by tags (``--seed``,
default 0). Configuration comes from the environment, falling back to ``.env``:
``RAW_DATA_DIR_MULTI``, ``DATA_GEN_RUN_ID_MULTI``, ``MLFLOW_BACKEND_STORE``,
``MLFLOW_ARTIFACT_STORE`` and ``MLFLOW_EXPERIMENT_NAME``. Variables already set in
the environment win over ``.env``.

Example:
    uv run scripts/battery_multiple_scenarios/run_variance_study.py --seed 0
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import pathlib
from collections.abc import Sequence

from qmodem.battery.backbone_data import load_backbone_context
from qmodem.battery.hpo import (
    STUDY_SEED,
    VARIANCE_STUDY,
    VarianceSearchSpace,
    variance_objective,
)
from qmodem.battery.train import (
    MEAN_CHECKPOINT_KEY,
    StageScheduleHyperparameters,
    VarianceHyperparameters,
    load_backbone_hyperparameters,
)
from qmodem.hpo import StudyLayout, run_study
from qmodem.script_session import script_session
from qmodem.tracking import (
    MLflowCheckpointStore,
    Role,
    RunSelector,
    Stage,
)

logger = logging.getLogger(__name__)

_BACKBONE = RunSelector(Role.BACKBONE, run_id_flag="--backbone-run-id")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line.

    Args:
        argv: Command-line arguments. Defaults to ``sys.argv[1:]``.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    _BACKBONE.add_arguments(parser)
    parser.set_defaults(seed=STUDY_SEED)
    parser.add_argument(
        "--n-trials", type=int, default=VARIANCE_STUDY.n_trials, help="Trial budget."
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Epoch cap of every trial; defaults to the backbone's variance cap "
        "(lower it only for smoke tests).",
    )
    parser.add_argument(
        "--optuna-storage",
        default=None,
        help="Optuna storage URL; the study stays in memory if omitted.",
    )
    parser.add_argument("--run-name", default="variance-study", help="Study run name.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the variance study and record it in MLflow.

    Args:
        argv: Command-line arguments. Defaults to ``sys.argv[1:]``.
    """
    args = parse_args(argv)
    with script_session() as session:
        backbone_run_id = _BACKBONE.resolve(
            args, experiment_ids=session.experiment_ids, client=session.client
        )
        mean_hp, variance_hp = load_backbone_hyperparameters(
            backbone_run_id, client=session.client
        )
        mean_checkpoint = MLflowCheckpointStore(
            backbone_run_id, client=session.client
        ).load(MEAN_CHECKPOINT_KEY)
        backbone = load_backbone_context(
            backbone_run_id,
            raw_data_dir=pathlib.Path(os.environ["RAW_DATA_DIR_MULTI"]),
            expected_data_generation_run_id=os.environ.get("DATA_GEN_RUN_ID_MULTI"),
            client=session.client,
        )
        logger.info("Backbone run %s (seed %d)", backbone_run_id, backbone.seed)

        space = VarianceSearchSpace()
        base = variance_hp
        if args.max_epochs is not None:
            base = VarianceHyperparameters(
                optimizer=variance_hp.optimizer,
                schedule=StageScheduleHyperparameters(
                    max_epochs=args.max_epochs, patience=variance_hp.schedule.patience
                ),
            )
        result = run_study(
            variance_objective(
                data=backbone.data,
                mean_hp=mean_hp,
                mean_checkpoint=mean_checkpoint,
                space=space,
                base=base,
                seed=backbone.seed,
            ),
            config=dataclasses.replace(VARIANCE_STUDY, n_trials=args.n_trials),
            layout=StudyLayout(stage=Stage.VARIANCE, backbone_run_id=backbone_run_id),
            setup=dataclasses.replace(session.setup, run_name=args.run_name),
            storage=args.optuna_storage,
        )
        logger.info(
            "Variance study %s: best value %.6g with %s",
            result.run_id,
            result.best_value,
            space.apply(result.best_params, base).model_dump(),
        )
        session.attach_log(result.run_id, "study_log.txt")


if __name__ == "__main__":
    main()
