"""Run the mean-stage HPO study on protocol seed 0 (protocol §5).

The study tunes the backbone sizes (kernel 2–8, filters 4–64, activation) and the
mean learning rate (1e-4–1e-2, log), with batch 32, patience 50 and a 500-epoch cap
fixed. Each trial trains the mean stage alone and minimizes its best-epoch
validation MSE. TPE with a fixed sampler seed proposes the trials, and a median
pruner (first 10 trials never pruned) stops poor ones on the per-epoch validation
MSE.

MLflow layout: one top-level ``role=hpo_study`` run tagged ``stage=mean`` and
``n_trials``, logging the selected params as ``best.<name>``
(``best.mean/network/kernel_size``, ...), with one ``role=hpo_trial`` child per
trial (tag ``hpo.trial=<n>``) holding only its ``checkpoints/mean/best``.

Configuration comes from the environment, falling back to ``.env``:
``RAW_DATA_DIR_MULTI``, ``DATA_GEN_RUN_ID_MULTI``, ``MLFLOW_BACKEND_STORE``,
``MLFLOW_ARTIFACT_STORE`` and ``MLFLOW_EXPERIMENT_NAME``. Variables already set in
the environment win over ``.env``.

Example:
    uv run scripts/battery_multiple_scenarios/run_mean_study.py
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import pathlib
from collections.abc import Sequence

from qmodem.battery.data_processing import WINDOW_SIZE, make_data_pipeline, prepare_data
from qmodem.battery.hpo import MEAN_STUDY, STUDY_SEED, MeanSearchSpace, mean_objective
from qmodem.battery.train import MeanHyperparameters, StageScheduleHyperparameters
from qmodem.hpo import StudyLayout, run_study
from qmodem.script_session import script_session
from qmodem.tracking import Stage

logger = logging.getLogger(__name__)

_DEFAULT_SCHEDULE = StageScheduleHyperparameters()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line.

    Args:
        argv: Command-line arguments. Defaults to ``sys.argv[1:]``.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--n-trials", type=int, default=MEAN_STUDY.n_trials, help="Trial budget."
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=_DEFAULT_SCHEDULE.max_epochs,
        help="Epoch cap of every trial (lower it only for smoke tests).",
    )
    parser.add_argument(
        "--optuna-storage",
        default=None,
        help="Optuna storage URL; the study stays in memory if omitted.",
    )
    parser.add_argument("--run-name", default="mean-study", help="Study run name.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the mean study and record it in MLflow.

    Args:
        argv: Command-line arguments. Defaults to ``sys.argv[1:]``.
    """
    args = parse_args(argv)
    with script_session() as session:
        data = prepare_data(
            pathlib.Path(os.environ["RAW_DATA_DIR_MULTI"]),
            os.environ["DATA_GEN_RUN_ID_MULTI"],
            make_data_pipeline(WINDOW_SIZE),
        )
        logger.info("Windows: %d train, %d validation", len(data.train), len(data.val))
        space = MeanSearchSpace()
        base = MeanHyperparameters(
            schedule=StageScheduleHyperparameters(max_epochs=args.max_epochs)
        )
        result = run_study(
            mean_objective(data=data, space=space, base=base, seed=STUDY_SEED),
            config=dataclasses.replace(MEAN_STUDY, n_trials=args.n_trials),
            layout=StudyLayout(stage=Stage.MEAN),
            setup=dataclasses.replace(session.setup, run_name=args.run_name),
            storage=args.optuna_storage,
        )
        logger.info(
            "Mean study %s: best value %.6g with %s",
            result.run_id,
            result.best_value,
            space.apply(result.best_params, base).model_dump(),
        )
        session.attach_log(result.run_id, "study_log.txt")


if __name__ == "__main__":
    main()
