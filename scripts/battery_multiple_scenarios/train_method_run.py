"""Train one method run on a backbone run of the multiple-scenarios data.

The method run is an MLflow child of its backbone run (``mlflow.parentRunId``),
tagged ``role=method``, ``method``, ``seed`` and ``stagereg_version``. It restores the
backbone's finalized ``mean`` and ``variance`` checkpoints, never re-trains them, and
logs its posterior hyperparameters, posterior-stage metrics and finalized
``checkpoints/posterior/best``. HNN trains nothing: its run only holds the
evaluation logged later by ``evaluate.py``.

The backbone is selected by ``--backbone-run-id`` or by ``--seed`` (newest finished
``role=backbone`` run of that seed). Configuration comes from the environment,
falling back to ``.env`` (see ``train_backbone.py``).

Example:
    uv run scripts/battery_multiple_scenarios/train_method_run.py --method hnn --seed 0
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
from qmodem.battery.method_run import METHOD_CONFIGS, method_config, track_method_run
from qmodem.battery.train import Method
from qmodem.tracking import MLFlowSetup, MLflowTracker, RunSelector
from qmodem.utils import setup_script_logging

logger = logging.getLogger(__name__)

BACKBONE_SELECTOR = RunSelector("backbone", run_id_flag="--backbone-run-id")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line.

    Args:
        argv: Command-line arguments. Defaults to ``sys.argv[1:]``.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--method",
        required=True,
        choices=[method.value for method in METHOD_CONFIGS],
        help="Inference method of the run.",
    )
    BACKBONE_SELECTOR.add_arguments(parser)
    parser.add_argument("--run-name", default=None, help="MLflow run name.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Train and record one method run.

    Args:
        argv: Command-line arguments. Defaults to ``sys.argv[1:]``.
    """
    args = parse_args(argv)
    load_dotenv()
    log_stream = setup_script_logging()

    method = Method(args.method)
    setup = MLFlowSetup(run_name=args.run_name)
    mlflow.set_tracking_uri(setup.backend_store)  # prepare_data reads the data run
    client = MlflowClient()
    experiment = client.get_experiment_by_name(setup.experiment_name)
    if experiment is None:
        raise SystemExit(f"No experiment {setup.experiment_name!r}; train a backbone.")
    backbone_run_id = BACKBONE_SELECTOR.resolve(
        args, experiment_ids=[experiment.experiment_id], client=client
    )
    seed = int(client.get_run(backbone_run_id).data.tags["seed"])
    logger.info("Backbone run %s (seed %d)", backbone_run_id, seed)

    data = prepare_data(
        pathlib.Path(os.environ["RAW_DATA_DIR_MULTI"]),
        os.environ["DATA_GEN_RUN_ID_MULTI"],
        make_data_pipeline(WINDOW_SIZE),
    )
    run = track_method_run(
        backbone_run_id=backbone_run_id,
        data=data,
        config=method_config(method, {}),
        seed=seed,
        setup=dataclasses.replace(
            setup, run_name=args.run_name or f"{method.value}-seed-{seed}"
        ),
        client=client,
    )
    logger.info("Method run %s (%s, seed %d)", run.run_id, method.value, seed)
    MLflowTracker(run.run_id).log_artifact("training_log.txt", log_stream.getvalue())


if __name__ == "__main__":
    main()
