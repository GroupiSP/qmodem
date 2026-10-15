"""Seam 2: a method run tracked in MLflow as a child of its backbone run."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest
import sklearn.preprocessing as skpp

from qmodem.battery.data_processing import FEATURES, WINDOW_SIZE
from qmodem.battery.evaluate import EvaluationParameters, evaluate_predictive_model
from qmodem.battery.method_run import (
    BackboneHandle,
    HNNConfig,
    evaluate_tracked_method_run,
    restore_method_run,
    track_method_run,
)
from qmodem.battery.train import TrackedBackbone, track_backbone_run
from qmodem.tracking import ScalerArtifact

N_TIMESTAMPS = 5


@dataclasses.dataclass(frozen=True)
class _Tracked:
    backbone: TrackedBackbone
    method_run_id: str
    x_scaler: skpp.StandardScaler
    y_scaler: skpp.MinMaxScaler


@pytest.fixture(scope="module")
def tracked(mlflow_backend, trained_backbone) -> _Tracked:
    rng = np.random.default_rng(0)
    x_scaler = skpp.StandardScaler().fit(rng.normal(3.0, 2.0, (50, len(FEATURES))))
    y_scaler = skpp.MinMaxScaler().fit(rng.uniform(0.0, 900.0, (50, 1)))
    backbone = track_backbone_run(
        data=trained_backbone.battery.data,
        mean_hp=trained_backbone.mean_hp,
        variance_hp=trained_backbone.variance_hp,
        seed=trained_backbone.seed,
        data_generation_run_id="data-generation-run",
        scalers=(
            ScalerArtifact("x_scaler", x_scaler, np.ones((WINDOW_SIZE, 2))),
            ScalerArtifact("y_scaler", y_scaler, np.ones((1, 1)), "inverse_transform"),
        ),
        setup=mlflow_backend.setup,
    )
    method_run = track_method_run(
        backbone_run_id=backbone.run_id,
        data=trained_backbone.battery.data,
        config=HNNConfig(),
        seed=trained_backbone.seed,
        setup=mlflow_backend.setup,
    )
    return _Tracked(backbone, method_run.run_id, x_scaler, y_scaler)


def test_the_method_run_is_a_tagged_child_of_its_backbone(
    tracked, mlflow_backend, trained_backbone
) -> None:
    run = mlflow_backend.client.get_run(tracked.method_run_id)

    assert run.data.tags["mlflow.parentRunId"] == tracked.backbone.run_id
    assert {key: run.data.tags.get(key) for key in ("role", "method", "seed")} == {
        "role": "method",
        "method": "hnn",
        "seed": str(trained_backbone.seed),
    }
    assert run.data.tags["stagereg_version"] == "0.2.0"
    assert run.info.status == "FINISHED"


def test_an_hnn_method_run_logs_no_checkpoint_and_no_backbone_params(
    tracked, mlflow_backend
) -> None:
    client = mlflow_backend.client
    run = client.get_run(tracked.method_run_id)

    assert client.list_artifacts(tracked.method_run_id, "checkpoints") == []
    assert not any(key.startswith(("mean/", "variance/")) for key in run.data.params)


def _test_histories() -> pd.DataFrame:
    frames = []
    for case_id, (policy, length) in enumerate([(0, 48), (1, 40)]):
        time = 5.0 * np.arange(length)
        frames.append(
            pd.DataFrame(
                {
                    "run_id": case_id,
                    "policy_id": policy,
                    "time": time,
                    "load": 1.0 + 0.1 * case_id + np.zeros(length),
                    "voltage": np.linspace(4.2, 3.4, length),
                    "rul": time[-1] - time,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_the_evaluation_is_logged_into_the_method_run_with_the_parent_scalers(
    tracked, mlflow_backend, trained_backbone, store_factory
) -> None:
    histories = _test_histories()
    rng = np.random.default_rng(1)
    reference = {
        f"test_case_{i}": rng.uniform(0, 250, (N_TIMESTAMPS, 7)) for i in range(2)
    }
    parameters = EvaluationParameters(n_timestamps=N_TIMESTAMPS, num_samples=20)

    evaluation = evaluate_tracked_method_run(
        tracked.method_run_id,
        test_histories=histories,
        reference_samples=reference,
        parameters=parameters,
        client=mlflow_backend.client,
    )

    in_process = restore_method_run(
        backbone=BackboneHandle(
            mean_hp=trained_backbone.mean_hp,
            variance_hp=trained_backbone.variance_hp,
            store=store_factory(trained_backbone.persistent),
        ),
        config=HNNConfig(),
        seed=trained_backbone.seed,
        store=store_factory(),
    )
    expected = evaluate_predictive_model(
        in_process,
        test_histories=histories,
        reference_samples=reference,
        x_scaler=tracked.x_scaler,
        y_scaler=tracked.y_scaler,
        parameters=parameters,
        key=in_process.evaluation_key,
    )
    for case, row in expected.table.items():
        assert evaluation.table[case] == pytest.approx(row)

    client = mlflow_backend.client
    metrics = client.get_run(tracked.method_run_id).data.metrics
    assert metrics["evaluation/test/crps"] == pytest.approx(
        evaluation.aggregate.overall["crps"]
    )
    assert "evaluation/test/case_1/wsu" in metrics
    assert "evaluation/test/policy_1/rmse" in metrics
    artifacts = {
        a.path for a in client.list_artifacts(tracked.method_run_id, "evaluation/test")
    }
    assert artifacts == {"evaluation/test/case_0", "evaluation/test/case_1"}
