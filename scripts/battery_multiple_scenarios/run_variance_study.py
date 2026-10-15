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

import mlflow
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

from qmodem.battery.data_processing import WINDOW_SIZE, make_data_pipeline, prepare_data
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
from qmodem.tracking import (
    MLFlowSetup,
    MLflowCheckpointStore,
    MLflowTracker,
    RunSelector,
)
from qmodem.utils import setup_script_logging

logger = logging.getLogger(__name__)

_BACKBONE = RunSelector("backbone", run_id_flag="--backbone-run-id")


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
    load_dotenv()
    log_stream = setup_script_logging()

    setup = MLFlowSetup(run_name=args.run_name)
    mlflow.set_tracking_uri(setup.backend_store)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(setup.experiment_name)
    experiment_ids = [experiment.experiment_id] if experiment is not None else []
    backbone_run_id = _BACKBONE.resolve(args, experiment_ids=experiment_ids)
    seed = int(client.get_run(backbone_run_id).data.tags["seed"])
    mean_hp, variance_hp = load_backbone_hyperparameters(backbone_run_id)
    mean_checkpoint = MLflowCheckpointStore(backbone_run_id).load(MEAN_CHECKPOINT_KEY)
    logger.info("Backbone run %s (seed %d)", backbone_run_id, seed)

    data = prepare_data(
        pathlib.Path(os.environ["RAW_DATA_DIR_MULTI"]),
        os.environ["DATA_GEN_RUN_ID_MULTI"],
        make_data_pipeline(WINDOW_SIZE),
    )
    logger.info("Windows: %d train, %d validation", len(data.train), len(data.val))

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
            data=data,
            mean_hp=mean_hp,
            mean_checkpoint=mean_checkpoint,
            space=space,
            base=base,
            seed=seed,
        ),
        config=dataclasses.replace(VARIANCE_STUDY, n_trials=args.n_trials),
        layout=StudyLayout(stage="variance", backbone_run_id=backbone_run_id),
        setup=setup,
        storage=args.optuna_storage,
    )
    logger.info(
        "Variance study %s: best value %.6g with %s",
        result.run_id,
        result.best_value,
        space.apply(result.best_params, base).model_dump(),
    )
    MLflowTracker(result.run_id).log_artifact("study_log.txt", log_stream.getvalue())


if __name__ == "__main__":
    main()
