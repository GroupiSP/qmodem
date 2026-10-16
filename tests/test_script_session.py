"""The shared setup of every script: environment, logging, MLflow backend."""

from __future__ import annotations

import logging

import pytest

from qmodem.script_session import script_session

ENV = ("MLFLOW_BACKEND_STORE", "MLFLOW_ARTIFACT_STORE", "MLFLOW_EXPERIMENT_NAME")


@pytest.fixture
def backend_env(mlflow_backend, monkeypatch: pytest.MonkeyPatch):
    """Point the environment at the test backend, as a scratch smoke run would."""
    setup = mlflow_backend.setup
    monkeypatch.setenv("MLFLOW_BACKEND_STORE", setup.backend_store)
    monkeypatch.setenv("MLFLOW_ARTIFACT_STORE", str(setup.artifact_store))
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", setup.experiment_name)
    return mlflow_backend


def test_a_session_uses_the_configured_backend_and_experiment(backend_env) -> None:
    with script_session(dotenv=False) as session:
        assert session.setup.backend_store == backend_env.setup.backend_store
        assert session.experiment_ids == [backend_env.experiment_id]
        assert session.require_experiment_ids() == [backend_env.experiment_id]


def test_a_missing_experiment_is_refused_when_required(
    backend_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "no-such-experiment")

    with script_session(dotenv=False) as session:
        assert session.experiment_ids == []
        with pytest.raises(SystemExit, match="no-such-experiment"):
            session.require_experiment_ids()


def test_the_script_log_is_attached_to_the_chosen_run_at_exit(backend_env) -> None:
    run_id = backend_env.new_run()

    with script_session(dotenv=False) as session:
        logging.getLogger("a-script").info("hello from the script")
        session.attach_log(run_id, "script_log.txt")

    text = backend_env.client.download_artifacts(run_id, "script_log.txt")
    with open(text) as file:
        assert "hello from the script" in file.read()


def test_no_log_is_attached_if_the_script_fails(backend_env) -> None:
    run_id = backend_env.new_run()

    with pytest.raises(RuntimeError), script_session(dotenv=False) as session:
        session.attach_log(run_id, "script_log.txt")
        raise RuntimeError("boom")

    assert backend_env.client.list_artifacts(run_id) == []
