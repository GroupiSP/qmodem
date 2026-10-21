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
default 0); it must have been re-trained from both the mean and the variance study
(tag ``variance_study_run_id``), so the trials see the tuned aleatoric variance.
``--max-epochs``, ``--patience`` and ``--posterior-param NAME=VALUE`` change the
fixed settings, e.g. for a smoke run, and tag the study run
``protocol_overrides=true``. Configuration comes from the
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

from stagereg.core import flatten_parameters

from qmodem.battery.backbone_data import BackboneContext, load_backbone_context
from qmodem.battery.hpo import STUDY_SEED
from qmodem.battery.method_run import (
    MethodConfig,
    method_config,
    posterior_overrides,
)
from qmodem.battery.posterior_hpo import (
    POSTERIOR_SEARCH_SPACES,
    POSTERIOR_STUDY,
    require_tuned_backbone,
    run_posterior_study,
    search_space,
)
from qmodem.battery.train import Method
from qmodem.script_session import ScriptSession, script_session
from qmodem.tracking import (
    Role,
    RunSelector,
    log_window_counts,
    protocol_override_tags,
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


def study_base(args: argparse.Namespace) -> tuple[MethodConfig, dict[str, str]]:
    """Return the study's fixed settings and the tags of their protocol overrides.

    Args:
        args: The parsed arguments.

    Returns:
        The base configuration and its ``protocol_overrides`` tags.

    Raises:
        SystemExit: If an override is malformed.
    """
    try:
        overrides = posterior_overrides(
            args.posterior_param, max_epochs=args.max_epochs, patience=args.patience
        )
    except ValueError as error:
        raise SystemExit(f"--posterior-param: {error}") from error
    base = method_config(Method(args.method), overrides)
    return base, protocol_override_tags(list(flatten_parameters(overrides)))


def tuned_backbone(
    args: argparse.Namespace, session: ScriptSession
) -> tuple[str, BackboneContext]:
    """Select the study's backbone run and load its data.

    Args:
        args: The parsed arguments.
        session: The script session.

    Returns:
        The backbone run ID and its seed and windows.

    Raises:
        SystemExit: If no experiment exists or the backbone was not trained from
            the variance study.
    """
    backbone_run_id = _BACKBONE.resolve(
        args, experiment_ids=session.require_experiment_ids(), client=session.client
    )
    try:
        require_tuned_backbone(backbone_run_id, client=session.client)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    backbone = load_backbone_context(
        backbone_run_id,
        raw_data_dir=pathlib.Path(os.environ["RAW_DATA_DIR_MULTI"]),
        expected_data_generation_run_id=os.environ.get("DATA_GEN_RUN_ID_MULTI"),
        client=session.client,
    )
    logger.info("Backbone run %s (seed %d)", backbone_run_id, backbone.seed)
    return backbone_run_id, backbone


def main(argv: Sequence[str] | None = None) -> None:
    """Run one posterior study and record it in MLflow.

    Args:
        argv: Command-line arguments. Defaults to ``sys.argv[1:]``.

    Raises:
        SystemExit: If an override is malformed, no experiment exists or the
            backbone is not tuned.
    """
    args = parse_args(argv)
    method = Method(args.method)
    base, override_tags = study_base(args)
    with script_session() as session:
        backbone_run_id, backbone = tuned_backbone(args, session)
        result = run_posterior_study(
            backbone_run_id=backbone_run_id,
            data=backbone.data,
            base=base,
            setup=dataclasses.replace(
                session.setup,
                run_name=args.run_name or f"posterior-study-{method.value}",
                tags=override_tags,
            ),
            config=dataclasses.replace(POSTERIOR_STUDY, n_trials=args.n_trials),
            seed=backbone.seed,
            storage=args.optuna_storage,
            client=session.client,
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
        log_window_counts(
            result.run_id,
            backbone.data.window_counts(),
            logger=logger,
            client=session.client,
        )
        session.attach_log(result.run_id, "study_log.txt")


if __name__ == "__main__":
    main()
