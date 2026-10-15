"""Evaluate one method run on the multiple-scenarios test cases (protocol §6).

The method run is restored in this process (backbone checkpoints and scalers from its
parent backbone run, its posterior from its own ``checkpoints/posterior/best``). Every
test case is evaluated at ``--n-timestamps`` timestamps with M predictive samples per
timestamp against the simulator's reference samples, and the evaluation is logged
into the method run as ``evaluation/test/<case>/<metric>``,
``evaluation/test/<metric>`` and ``evaluation/test/policy_<p>/<metric>``, with the
per-case samples as artifacts.

The method run is selected by ``--run-id`` or by ``--method`` and ``--seed`` (newest
finished ``role=method`` run). Configuration comes from the environment, falling
back to ``.env`` (see ``train_backbone.py``). The simulator config comes from the
data-generation run tagged on the parent backbone run; a configured
``DATA_GEN_RUN_ID_MULTI`` that disagrees with it is refused.

Example:
    uv run scripts/battery_multiple_scenarios/evaluate.py --method hnn --seed 0
"""

from __future__ import annotations

import argparse
import logging
import os
import pathlib
from collections.abc import Sequence

import mlflow
import pandas as pd
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

from qmodem.battery.backbone_data import backbone_data_generation_run_id
from qmodem.battery.data_generation import (
    load_simulation_config,
    reconstruct_true_rul_distribution,
    sim_updater_two_scenarios,
)
from qmodem.battery.evaluate import EvaluationParameters
from qmodem.battery.method_run import PARENT_RUN_TAG, evaluate_tracked_method_run
from qmodem.tracking import MLFlowSetup, MLflowTracker, RunSelector
from qmodem.utils import setup_script_logging

logger = logging.getLogger(__name__)

METHOD_RUN_SELECTOR = RunSelector("method")
_DEFAULTS = EvaluationParameters()
_EVENT_TIME = 900.0
"""Time after which the simulator's policy is known (two-scenarios event)."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line.

    Args:
        argv: Command-line arguments. Defaults to ``sys.argv[1:]``.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    METHOD_RUN_SELECTOR.add_arguments(parser)
    parser.add_argument(
        "--n-timestamps",
        type=int,
        default=_DEFAULTS.n_timestamps,
        help="Evaluation timestamps per test case, t0 included.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=_DEFAULTS.num_samples,
        help="Predictive samples per timestamp (M).",
    )
    parser.add_argument(
        "--n-reference-samples",
        type=int,
        default=100,
        help="Simulator reference samples per timestamp.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Evaluate one method run and log the evaluation into it.

    Args:
        argv: Command-line arguments. Defaults to ``sys.argv[1:]``.
    """
    args = parse_args(argv)
    load_dotenv()
    log_stream = setup_script_logging()

    setup = MLFlowSetup()
    mlflow.set_tracking_uri(setup.backend_store)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(setup.experiment_name)
    if experiment is None:
        raise SystemExit(f"No experiment {setup.experiment_name!r}.")
    run_id = METHOD_RUN_SELECTOR.resolve(
        args, experiment_ids=[experiment.experiment_id], client=client
    )
    logger.info("Evaluating method run %s", run_id)
    data_run_id = backbone_data_generation_run_id(
        client.get_run(run_id).data.tags[PARENT_RUN_TAG],
        expected=os.environ.get("DATA_GEN_RUN_ID_MULTI"),
        client=client,
    )

    parameters = EvaluationParameters(
        n_timestamps=args.n_timestamps, num_samples=args.num_samples
    )
    test_histories = pd.read_csv(
        pathlib.Path(os.environ["RAW_DATA_DIR_MULTI"]) / "test.csv"
    )
    reference_samples = reconstruct_true_rul_distribution(
        test_data=test_histories,
        simulator_base_config=load_simulation_config(run_id=data_run_id),
        n_soc0s=parameters.n_timestamps,
        n_mc_samples=args.n_reference_samples,
        simulator_updater=sim_updater_two_scenarios,
        event_fn=lambda t0, soc0: t0 > _EVENT_TIME,
    )
    evaluation = evaluate_tracked_method_run(
        run_id,
        test_histories=test_histories,
        reference_samples=reference_samples,
        parameters=parameters,
        client=client,
    )
    for metric, value in evaluation.aggregate.overall.items():
        logger.info("evaluation/test/%s = %.4f", metric, value)
    MLflowTracker(run_id).log_artifact("evaluation_log.txt", log_stream.getvalue())


if __name__ == "__main__":
    main()
