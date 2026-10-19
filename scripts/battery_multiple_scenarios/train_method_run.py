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

Methods with a posterior stage take the protocol's schedule (300 epochs, patience
60) unless ``--max-epochs`` / ``--patience`` override it, e.g. for a smoke run. Any
other posterior hyperparameter is overridden by ``--posterior-param NAME=VALUE``
(e.g. pSGLD's ``burn_in_epochs``, which must stay below ``--max-epochs``). Any
override tags the run ``protocol_overrides=true`` and lists the overridden settings
in ``protocol_overrides.params``; so does training a method with a posterior stage
without ``--study-run-id`` (untuned defaults, listed as ``posterior_study``). Tag
selection (``evaluate.py --method --seed``) skips such runs unless
``--include-overridden`` is passed.

Protocol runs (seed 0 included) are re-trained from the method's posterior study:
``--study-run-id`` applies the study run's frozen ``best.*`` params (the study's
fixed settings, e.g. its epoch cap, must be passed again as overrides if they were
changed) and is recorded as the tag ``posterior_study_run_id``. A trial run is
never reused.

Example:
    uv run scripts/battery_multiple_scenarios/train_method_run.py --method hnn --seed 0
    uv run scripts/battery_multiple_scenarios/train_method_run.py --method bbb --seed 0 \
        --max-epochs 5
    uv run scripts/battery_multiple_scenarios/train_method_run.py --method psgld \
        --seed 0 --max-epochs 4 --posterior-param burn_in_epochs=2
    uv run scripts/battery_multiple_scenarios/train_method_run.py --method mcd --seed 1 \
        --study-run-id <posterior study run ID>
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import pathlib
from collections.abc import Sequence
from typing import Any

from mlflow.tracking import MlflowClient
from stagereg.core import flatten_parameters

from qmodem.battery.backbone_data import load_backbone_context
from qmodem.battery.data_processing import log_window_counts
from qmodem.battery.method_run import (
    METHOD_CONFIGS,
    MethodConfig,
    method_config,
    method_override_tags,
    posterior_overrides,
    track_method_run,
)
from qmodem.battery.posterior_hpo import frozen_method_config
from qmodem.battery.train import Method
from qmodem.script_session import script_session
from qmodem.tracking import Role, RunSelector

logger = logging.getLogger(__name__)

BACKBONE_SELECTOR = RunSelector(Role.BACKBONE, run_id_flag="--backbone-run-id")


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
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Posterior-stage epoch cap (default: the protocol's 300).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Posterior-stage early-stopping patience (default: the protocol's 60).",
    )
    parser.add_argument(
        "--posterior-param",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help=(
            "Override one posterior hyperparameter, nested with '/', e.g. "
            "burn_in_epochs=2 (repeatable)."
        ),
    )
    parser.add_argument(
        "--study-run-id",
        default=None,
        help=(
            "Posterior study run (role=hpo_study) of this method whose frozen best.* "
            "params set the tuned HPs (protocol runs); the others come from the "
            "defaults and overrides."
        ),
    )
    parser.add_argument("--run-name", default=None, help="MLflow run name.")
    return parser.parse_args(argv)


def overrides_of(args: argparse.Namespace) -> dict[str, Any]:
    """Return the posterior overrides given on the command line, nested.

    Args:
        args: The parsed arguments.

    Returns:
        The overrides (empty if none were given).

    Raises:
        SystemExit: If an override is malformed.
    """
    try:
        return posterior_overrides(
            args.posterior_param, max_epochs=args.max_epochs, patience=args.patience
        )
    except ValueError as error:
        raise SystemExit(f"--posterior-param: {error}") from error


def method_configuration(
    args: argparse.Namespace, client: MlflowClient
) -> MethodConfig:
    """Return the run's configuration: defaults, frozen study HPs, CLI overrides.

    Args:
        args: The parsed arguments.
        client: The MLflow client (reads the study run, if one is given).

    Returns:
        The method configuration.

    Raises:
        SystemExit: If overrides are given for a method without posterior stage,
            or an override is malformed.
    """
    method = Method(args.method)
    overrides = overrides_of(args)
    if overrides and METHOD_CONFIGS[method]().posterior_spec() is None:
        raise SystemExit(f"Method {args.method!r} has no posterior stage.")
    base = method_config(method, overrides)
    if args.study_run_id is None:
        return base
    return frozen_method_config(args.study_run_id, base, client=client)


def main(argv: Sequence[str] | None = None) -> None:
    """Train and record one method run.

    Args:
        argv: Command-line arguments. Defaults to ``sys.argv[1:]``.
    """
    args = parse_args(argv)
    method = Method(args.method)
    with script_session() as session:
        backbone_run_id = BACKBONE_SELECTOR.resolve(
            args,
            experiment_ids=session.require_experiment_ids(),
            client=session.client,
        )
        backbone = load_backbone_context(
            backbone_run_id,
            raw_data_dir=pathlib.Path(os.environ["RAW_DATA_DIR_MULTI"]),
            expected_data_generation_run_id=os.environ.get("DATA_GEN_RUN_ID_MULTI"),
            client=session.client,
        )
        config = method_configuration(args, session.client)
        if config.posterior_spec() is None:
            logger.info(
                "%s has no posterior stage: nothing to train, the run only holds "
                "the evaluation logged later by evaluate.py",
                method.value.upper(),
            )
        run = track_method_run(
            backbone_run_id=backbone_run_id,
            data=backbone.data,
            config=config,
            seed=backbone.seed,
            setup=dataclasses.replace(
                session.setup,
                run_name=args.run_name or f"{method.value}-seed-{backbone.seed}",
                tags=method_override_tags(
                    method,
                    list(flatten_parameters(overrides_of(args))),
                    study_run_id=args.study_run_id,
                ),
            ),
            client=session.client,
            study_run_id=args.study_run_id,
        )
        logger.info(
            "Method run %s (%s, seed %d), windows per split: %s",
            run.run_id,
            method.value,
            backbone.seed,
            backbone.data.window_counts(),
        )
        log_window_counts(run.run_id, backbone.data, client=session.client)
        session.attach_log(run.run_id, "training_log.txt")


if __name__ == "__main__":
    main()
