"""Evaluate one method run on the multiple-scenarios test cases (protocol §6).

The method run is restored in this process (backbone checkpoints and scalers from its
parent backbone run, its posterior from its own ``checkpoints/posterior/best``). Every
test case is evaluated at ``--n-timestamps`` timestamps from the exact posterior
predictive over M posterior draws (``--num-draws``) against the simulator's reference
samples, and the evaluation is logged into the method run as
``evaluation/test/<case>/<metric>``, ``evaluation/test/<metric>`` and
``evaluation/test/policy_<p>/<metric>``, with each case's posterior predictive as an
artifact.

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

import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

from qmodem.battery.backbone_data import backbone_data_generation_run_id
from qmodem.battery.data_generation import (
    load_simulation_config,
    reconstruct_true_rul_distribution,
    sim_updater_two_scenarios,
)
from qmodem.battery.evaluate import EvaluationParameters
from qmodem.battery.method_run import PARENT_RUN_TAG, evaluate_tracked_method_run
from qmodem.script_session import script_session
from qmodem.tracking import Role, RunSelector

logger = logging.getLogger(__name__)

METHOD_RUN_SELECTOR = RunSelector(Role.METHOD)
_DEFAULTS = EvaluationParameters()
_EVENT_TIME = 900.0
"""Time after which the simulator's policy is known (two-scenarios event)."""


def reference_samples(
    run_id: str,
    test_histories: pd.DataFrame,
    *,
    parameters: EvaluationParameters,
    n_reference_samples: int,
    client: MlflowClient,
) -> dict[str, np.ndarray]:
    """Simulate the reference samples of the method run's data-generation run.

    Args:
        run_id: The method run (its parent backbone names the data run).
        test_histories: The test histories.
        parameters: Evaluation parameters (timestamps per case).
        n_reference_samples: Simulator samples per timestamp.
        client: The MLflow client.

    Returns:
        Reference samples keyed ``test_case_<i>``.

    Raises:
        ValueError: If ``DATA_GEN_RUN_ID_MULTI`` disagrees with the backbone.
    """
    data_run_id = backbone_data_generation_run_id(
        client.get_run(run_id).data.tags[PARENT_RUN_TAG],
        expected=os.environ.get("DATA_GEN_RUN_ID_MULTI"),
        client=client,
    )
    return reconstruct_true_rul_distribution(
        test_data=test_histories,
        simulator_base_config=load_simulation_config(run_id=data_run_id),
        n_soc0s=parameters.n_timestamps,
        n_mc_samples=n_reference_samples,
        simulator_updater=sim_updater_two_scenarios,
        event_fn=lambda t0, soc0: t0 > _EVENT_TIME,
    )


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
        "--num-draws",
        type=int,
        default=_DEFAULTS.num_draws,
        help="Posterior draws per timestamp (M).",
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
    parameters = EvaluationParameters(
        n_timestamps=args.n_timestamps, num_draws=args.num_draws
    )
    with script_session() as session:
        run_id = METHOD_RUN_SELECTOR.resolve(
            args,
            experiment_ids=session.require_experiment_ids(),
            client=session.client,
        )
        logger.info("Evaluating method run %s", run_id)
        test_histories = pd.read_csv(
            pathlib.Path(os.environ["RAW_DATA_DIR_MULTI"]) / "test.csv"
        )
        evaluation = evaluate_tracked_method_run(
            run_id,
            test_histories=test_histories,
            reference_samples=reference_samples(
                run_id,
                test_histories,
                parameters=parameters,
                n_reference_samples=args.n_reference_samples,
                client=session.client,
            ),
            parameters=parameters,
            client=session.client,
        )
        for metric, value in evaluation.aggregate.overall.items():
            logger.info("evaluation/test/%s = %.4f", metric, value)
        session.attach_log(run_id, "evaluation_log.txt")


if __name__ == "__main__":
    main()
