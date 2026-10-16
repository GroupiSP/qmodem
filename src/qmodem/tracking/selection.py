"""Selecting the run a script consumes: by exact run ID or by identity tags."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from .tags import PROTOCOL_OVERRIDES_TAG, Role

IDENTITY_TAGS: Mapping[str, tuple[str, ...]] = {
    Role.BACKBONE: ("seed",),
    Role.METHOD: ("method", "seed"),
}
"""Roles selectable by tags, and the tags that identify one run of each role.

HPO roles (``hpo_study``, ``hpo_trial``) are absent on purpose, so tag selection
never returns a tuning run.
"""


class RunSelectionError(LookupError):
    """Raised when run selection finds no run, or cannot tell which run is meant."""


@dataclass(frozen=True)
class RunCriteria:
    """Tag criteria selecting one protocol run.

    Attributes:
        role: The ``role`` tag, e.g. ``"backbone"`` or ``"method"``.
        seed: The ``seed`` tag (protocol seed r), or ``None`` to leave it open.
        method: The ``method`` tag, e.g. ``"q-avi"``, or ``None`` to leave it open.
        include_overridden: Also select runs tagged ``PROTOCOL_OVERRIDES_TAG``
            (e.g. smoke runs); by default they are skipped.
    """

    role: str
    seed: int | None = None
    method: str | None = None
    include_overridden: bool = False


def resolve_run_id(
    *,
    run_id: str | None,
    criteria: RunCriteria | None,
    experiment_ids: Sequence[str],
    client: MlflowClient | None = None,
) -> str:
    """Return the run a script consumes: the explicit ``run_id`` or a tag selection.

    Args:
        run_id: An exact run ID (``--run-id``). It wins over ``criteria``, but its
            ``role`` tag must still equal ``criteria.role`` when criteria are given.
        criteria: Tag criteria, used when ``run_id`` is ``None``.
        experiment_ids: Experiments searched by tag selection.
        client: The MLflow client. Defaults to one on the current tracking URI.

    Returns:
        The selected run's ID.

    Raises:
        ValueError: If neither ``run_id`` nor ``criteria`` is given, or the role is
            not selectable (see ``select_run``).
        RunSelectionError: If ``run_id`` does not exist or has a role other than
            ``criteria.role``, or tag selection finds no run or an ambiguous one
            (see ``select_run``).
    """
    client = client if client is not None else MlflowClient()
    if run_id is not None:
        try:
            run = client.get_run(run_id)
        except MlflowException as error:
            raise RunSelectionError(f"No run with ID {run_id!r}.") from error
        role = run.data.tags.get("role")
        if criteria is not None and role != criteria.role:
            raise RunSelectionError(
                f"Run {run_id} has role {role!r}, expected {str(criteria.role)!r}."
            )
        return run.info.run_id
    if criteria is None:
        raise ValueError("Pass a run ID or tag criteria to select a run.")
    return select_run(criteria, experiment_ids=experiment_ids, client=client)


@dataclass(frozen=True)
class RunSelector:
    """Command-line run selection for a script that consumes a run of one role.

    It adds an exact run-ID flag plus one flag per identity tag of the role
    (``--seed`` for a backbone, ``--method`` and ``--seed`` for a method run), and
    resolves the parsed arguments with ``resolve_run_id``.

    Attributes:
        role: The role of the consumed run, a key of ``IDENTITY_TAGS``.
        run_id_flag: The exact run-ID flag, e.g. ``"--backbone-run-id"``.
    """

    role: str
    run_id_flag: str = "--run-id"

    @property
    def _run_id_dest(self) -> str:
        return self.run_id_flag.lstrip("-").replace("-", "_")

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add the run-ID flag and the role's identity-tag flags.

        Args:
            parser: The script's parser.
        """
        parser.add_argument(
            self.run_id_flag,
            default=None,
            help=f"Exact {self.role} run ID; wins over tag selection.",
        )
        types = {"seed": int, "method": str}
        for tag in IDENTITY_TAGS[self.role]:
            parser.add_argument(
                f"--{tag}",
                type=types[tag],
                default=None,
                help=f"Select the newest finished {self.role} run by {tag}.",
            )
        parser.add_argument(
            "--include-overridden",
            action="store_true",
            help=(
                f"Let tag selection pick {self.role} runs with overridden protocol "
                "settings (smoke runs)."
            ),
        )

    def resolve(
        self,
        args: argparse.Namespace,
        *,
        experiment_ids: Sequence[str],
        client: MlflowClient | None = None,
    ) -> str:
        """Return the run selected by the parsed arguments.

        Args:
            args: Arguments parsed by a parser set up with ``add_arguments``.
            experiment_ids: Experiments searched by tag selection.
            client: The MLflow client. Defaults to one on the current tracking URI.

        Returns:
            The selected run's ID.

        Raises:
            RunSelectionError: See ``resolve_run_id``.
        """
        tags = {tag: getattr(args, tag) for tag in IDENTITY_TAGS[self.role]}
        return resolve_run_id(
            run_id=getattr(args, self._run_id_dest),
            criteria=RunCriteria(
                role=self.role,
                include_overridden=args.include_overridden,
                **tags,
            ),
            experiment_ids=experiment_ids,
            client=client,
        )


def _filter_string(criteria: RunCriteria) -> str:
    """Build the MLflow search filter of finished runs matching ``criteria``.

    Args:
        criteria: The tag criteria.

    Returns:
        The filter string.
    """
    tags = {"role": criteria.role, "seed": criteria.seed, "method": criteria.method}
    clauses = [
        f"tags.{key} = '{value}'" for key, value in tags.items() if value is not None
    ]
    return " AND ".join([*clauses, "attributes.status = 'FINISHED'"])


def select_run(
    criteria: RunCriteria,
    *,
    experiment_ids: Sequence[str],
    client: MlflowClient | None = None,
) -> str:
    """Select the newest finished run whose tags match ``criteria``.

    Args:
        criteria: The tag criteria.
        experiment_ids: Experiments to search.
        client: The MLflow client. Defaults to one on the current tracking URI.

    Selection is ambiguous, and raises, when the criteria leave one of the role's
    identity tags (``IDENTITY_TAGS``) open, or when two or more finished matches
    share the newest start time. Older ties do not matter. Runs tagged
    ``PROTOCOL_OVERRIDES_TAG`` are skipped unless ``criteria.include_overridden``.

    Returns:
        The ID of the newest (latest start time) finished match.

    Raises:
        ValueError: If ``criteria.role`` is not selectable by tags (e.g.
            ``hpo_trial``); HPO runs are never selected as protocol runs.
        RunSelectionError: If no finished run matches, or the selection is
            ambiguous.
    """
    if criteria.role not in IDENTITY_TAGS:
        raise ValueError(
            f"Role {criteria.role!r} is not selectable by tags; "
            f"selectable roles are {sorted(IDENTITY_TAGS)}."
        )
    open_tags = [
        tag for tag in IDENTITY_TAGS[criteria.role] if getattr(criteria, tag) is None
    ]
    if open_tags:
        raise RunSelectionError(
            f"Ambiguous selection {criteria}: role {criteria.role!r} needs "
            f"{', '.join(open_tags)}."
        )
    client = client if client is not None else MlflowClient()
    runs = [
        run
        for run in client.search_runs(
            list(experiment_ids),
            filter_string=_filter_string(criteria),
            order_by=["attributes.start_time DESC"],
        )
        if criteria.include_overridden
        or run.data.tags.get(PROTOCOL_OVERRIDES_TAG) != "true"
    ][:2]
    if not runs:
        raise RunSelectionError(f"No finished run matches {criteria}.")
    if len(runs) == 2 and runs[0].info.start_time == runs[1].info.start_time:
        raise RunSelectionError(
            f"Ambiguous selection {criteria}: runs {runs[0].info.run_id} and "
            f"{runs[1].info.run_id} share the newest start time; pass --run-id."
        )
    return runs[0].info.run_id
