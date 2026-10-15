"""Run one method's posterior HPO study on seed 0's backbone run (protocol §5).

Every method with a posterior stage (MCD, BBB, pSGLD, Q-AVI, C-AVI) gets the same
budget: 50 trials, TPE with a fixed sampler seed, and a median pruner (5 startup
trials, 60 warm-up epochs) on the per-epoch validation CRPS. Each trial restores the
selected backbone run and trains one method run with the same key
``fold_in(0, method)``, so trials differ only in their hyperparameters. The
objective is the validation CRPS of the finalized epoch: the best one for the
early-stopped methods, the last one (the final retained chain) for pSGLD.

MLflow layout: one top-level ``role=hpo_study`` run tagged ``stage=posterior``,
``method``, ``backbone_run_id``, ``seed`` and ``n_trials``, logging the selected
params as ``best.<name>`` (e.g. ``best.posterior/dropout_rate``), with one
``role=hpo_trial`` child per trial holding its ``checkpoints/posterior/best``.
Protocol method runs are then re-trained from these with
``train_method_run.py --study-run-id``.

The backbone run is selected by ``--backbone-run-id`` or by tags (``--seed``,
default 0). ``--max-epochs``, ``--patience`` and ``--posterior-param NAME=VALUE``
change the fixed settings, e.g. for a smoke run. Configuration comes from the
environment, falling back to ``.env`` (see ``train_backbone.py``).

Example:
    uv run scripts/battery_multiple_scenarios/run_posterior_study.py --method mcd
    uv run scripts/battery_multiple_scenarios/run_posterior_study.py --method psgld \
        --n-trials 2 --max-epochs 4 --posterior-param burn_in_epochs=2
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

from qmodem.battery.backbone_data import prepare_backbone_data
from qmodem.battery.hpo import STUDY_SEED
from qmodem.battery.method_run import method_config, posterior_overrides
from qmodem.battery.posterior_hpo import (
    POSTERIOR_SEARCH_SPACES,
    POSTERIOR_STUDY,
    run_posterior_study,
    search_space,
)
from qmodem.battery.train import Method
from qmodem.tracking import MLFlowSetup, MLflowTracker, RunSelector
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
    parser.add_argument(
        "--method",
        required=True,
        choices=[method.value for method in POSTERIOR_SEARCH_SPACES],
        help="Method whose posterior stage is tuned.",
    )
    _BACKBONE.add_arguments(parser)
    parser.set_defaults(seed=STUDY_SEED)
    parser.add_argument(
        "--n-trials", type=int, default=POSTERIOR_STUDY.n_trials, help="Trial budget."
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Posterior-stage epoch cap of every trial (default: the protocol's 300).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Posterior-stage patience of every trial (default: the protocol's 60).",
    )
    parser.add_argument(
        "--posterior-param",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Override one fixed posterior hyperparameter (repeatable).",
    )
    parser.add_argument(
        "--optuna-storage",
        default=None,
        help="Optuna storage URL; the study stays in memory if omitted.",
    )
    parser.add_argument("--run-name", default=None, help="Study run name.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run one posterior study and record it in MLflow.

    Args:
        argv: Command-line arguments. Defaults to ``sys.argv[1:]``.

    Raises:
        SystemExit: If an override is malformed or no experiment exists.
    """
    args = parse_args(argv)
    load_dotenv()
    log_stream = setup_script_logging()

    method = Method(args.method)
    try:
        overrides = posterior_overrides(
            args.posterior_param, max_epochs=args.max_epochs, patience=args.patience
        )
    except ValueError as error:
        raise SystemExit(f"--posterior-param: {error}") from error
    base = method_config(method, overrides)

    setup = MLFlowSetup(run_name=args.run_name or f"posterior-study-{method.value}")
    mlflow.set_tracking_uri(setup.backend_store)  # prepare_data reads the data run
    client = MlflowClient()
    experiment = client.get_experiment_by_name(setup.experiment_name)
    if experiment is None:
        raise SystemExit(f"No experiment {setup.experiment_name!r}; train a backbone.")
    backbone_run_id = _BACKBONE.resolve(
        args, experiment_ids=[experiment.experiment_id], client=client
    )
    seed = int(client.get_run(backbone_run_id).data.tags["seed"])
    logger.info("Backbone run %s (seed %d)", backbone_run_id, seed)

    data = prepare_backbone_data(
        backbone_run_id,
        raw_data_dir=pathlib.Path(os.environ["RAW_DATA_DIR_MULTI"]),
        expected_data_generation_run_id=os.environ.get("DATA_GEN_RUN_ID_MULTI"),
        client=client,
    )
    logger.info("Windows: %d train, %d validation", len(data.train), len(data.val))

    result = run_posterior_study(
        backbone_run_id=backbone_run_id,
        data=data,
        base=base,
        setup=setup,
        config=dataclasses.replace(POSTERIOR_STUDY, n_trials=args.n_trials),
        seed=seed,
        storage=args.optuna_storage,
        client=client,
    )
    logger.info(
        "Posterior study %s (%s): best value %.6g with %s",
        result.run_id,
        method.value,
        result.best_value,
        search_space(method)
        .apply(result.best_params, base)
        .posterior_hyperparameters(),
    )
    MLflowTracker(result.run_id).log_artifact("study_log.txt", log_stream.getvalue())


if __name__ == "__main__":
    main()
