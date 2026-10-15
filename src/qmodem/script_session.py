"""The shared setup of every script: environment, logging and MLflow backend.

Every script under ``scripts/`` runs inside ``script_session``::

    with script_session() as session:
        ids = session.require_experiment_ids()
        ...
        session.attach_log(run_id, "training_log.txt")

The session loads ``.env`` (variables already set in the environment win),
captures the script's log, points MLflow at the configured backend and, if the
script attached its log to a run, logs it there when the script succeeds.
"""

from __future__ import annotations

import dataclasses
import io
from collections.abc import Iterator
from contextlib import contextmanager

import mlflow
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

from .tracking import MLFlowSetup, MLflowTracker
from .utils import setup_script_logging


@dataclasses.dataclass
class ScriptSession:
    """What a running script shares with the tracking backend.

    Attributes:
        setup: The configured backend, artifact store and experiment (from the
            environment); scripts derive their run's setup from it.
        client: A client on the configured backend.
        log_stream: The script's captured log.
    """

    setup: MLFlowSetup
    client: MlflowClient
    log_stream: io.StringIO
    _log_target: tuple[str, str] | None = dataclasses.field(default=None, repr=False)

    @property
    def experiment_ids(self) -> list[str]:
        """The configured experiment's ID, or ``[]`` if it does not exist yet."""
        experiment = self.client.get_experiment_by_name(self.setup.experiment_name)
        return [] if experiment is None else [experiment.experiment_id]

    def require_experiment_ids(self) -> list[str]:
        """Return the configured experiment's ID, refusing a missing experiment.

        Returns:
            ``[experiment_id]``.

        Raises:
            SystemExit: If the experiment does not exist.
        """
        ids = self.experiment_ids
        if not ids:
            raise SystemExit(
                f"No experiment {self.setup.experiment_name!r}; train a backbone."
            )
        return ids

    def attach_log(self, run_id: str, name: str) -> None:
        """Log the script's captured log into ``run_id`` when the session ends.

        Args:
            run_id: The run the script created or evaluated.
            name: The artifact name, e.g. ``"training_log.txt"``.
        """
        self._log_target = (run_id, name)


@contextmanager
def script_session(*, dotenv: bool = True) -> Iterator[ScriptSession]:
    """Run a script body with the shared environment, logging and backend.

    Args:
        dotenv: Load ``.env`` first (without overriding the environment).

    Yields:
        The session.
    """
    if dotenv:
        load_dotenv()
    log_stream = setup_script_logging()
    setup = MLFlowSetup()
    mlflow.set_tracking_uri(setup.backend_store)
    session = ScriptSession(setup=setup, client=MlflowClient(), log_stream=log_stream)
    yield session
    if session._log_target is not None:
        run_id, name = session._log_target
        MLflowTracker(run_id, client=session.client).log_artifact(
            name, log_stream.getvalue()
        )
