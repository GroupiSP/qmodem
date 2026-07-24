from __future__ import annotations

import contextlib
import io
import runpy
from types import SimpleNamespace
from unittest.mock import Mock

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
import pytest

import qmodem.battery.train as battery_train
from qmodem.battery.models import HeteroscedasticCNN
from qmodem.battery.train import QAVITrainHyperparameters, TrainHyperparameters
from qmodem.battery.train_steps import make_nll_steps
from qmodem.callbacks import track_conv_weights_variance
from qmodem.tracking import MLFlowSetup
from qmodem.train_base import TrainingPhase


def _states_differ(before: nnx.State, after: nnx.State) -> bool:
    return any(
        not np.array_equal(before_leaf, after_leaf)
        for before_leaf, after_leaf in zip(
            jax.tree.leaves(before), jax.tree.leaves(after), strict=True
        )
    )


def _copy_params(model: nnx.Module) -> nnx.State:
    return jax.tree.map(lambda value: value.copy(), nnx.state(model, nnx.Param))


def test_default_nll_steps_update_only_during_training() -> None:
    model = HeteroscedasticCNN(kernel_size=3, rngs=nnx.Rngs(0))
    optimizer = nnx.Optimizer(model, optax.sgd(1e-2), wrt=nnx.Param)
    batch = (jnp.ones((4, 5, 1)), jnp.ones((4, 1)))
    keys = jax.random.split(jax.random.key(0), num=4)
    train_step, eval_step = make_nll_steps()

    before_train = _copy_params(model)
    train_loss = train_step(model, batch, keys, optimizer)
    after_train = _copy_params(model)
    eval_loss = eval_step(model, batch, keys, optimizer)
    after_eval = _copy_params(model)

    assert jnp.isfinite(train_loss)
    assert jnp.isfinite(eval_loss)
    assert _states_differ(before_train, after_train)
    assert not _states_differ(after_train, after_eval)


def test_prepare_data_fits_scaler_on_training_targets_only(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = pd.DataFrame(
        {
            "run_id": [0] * 4 + [1] * 4,
            "time": [0, 1, 2, 3, 0, 10, 20, 30],
            "voltage": [4.2, 4.0, 3.8, 3.6, 4.2, 4.0, 3.8, 3.6],
        }
    )
    data.to_csv(tmp_path / "train.csv", index=False)
    monkeypatch.setattr(
        battery_train.mlflow,
        "get_run",
        lambda run_id: SimpleNamespace(
            data=SimpleNamespace(params={"n_histories_train": "1"})
        ),
    )

    prepared = battery_train._prepare_data(
        tmp_path,
        "data-run",
        TrainHyperparameters(window_size=2),
    )

    assert len(prepared.train) == 3
    assert len(prepared.val) == 3
    assert prepared.scaler.data_max_.item() == pytest.approx(1.0)
    assert prepared.train.y.max().item() == pytest.approx(1.0)
    assert prepared.val.y.max().item() == pytest.approx(10.0)


def test_run_training_injects_dataset_size_and_extra_callbacks(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = pd.DataFrame(
        {
            "run_id": [0] * 4 + [1] * 4,
            "time": [0, 1, 2, 3] * 2,
            "voltage": [4.2, 4.0, 3.8, 3.6] * 2,
        }
    )
    data.to_csv(tmp_path / "train.csv", index=False)
    monkeypatch.setattr(
        battery_train.mlflow,
        "get_run",
        lambda run_id: SimpleNamespace(
            data=SimpleNamespace(params={"n_histories_train": "1"})
        ),
    )

    @contextlib.contextmanager
    def fake_tracking(*, setup):
        yield SimpleNamespace()

    monkeypatch.setattr(battery_train, "track_mlflow", fake_tracking)
    monkeypatch.setattr(battery_train.mlflow.sklearn, "log_model", Mock())
    monkeypatch.setattr(battery_train.mlflow, "log_params", Mock())
    monkeypatch.setattr(battery_train.mlflow, "log_param", Mock())
    monkeypatch.setattr(battery_train.mlflow, "log_text", Mock())
    monkeypatch.setattr(battery_train, "count_parameters", lambda model: 1)

    captured: dict = {}

    def fake_train_loop(**kwargs) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(battery_train, "train_loop", fake_train_loop)
    factory_sizes: list[int] = []

    def step_factory(context):
        factory_sizes.append(context.n_train_samples)
        return Mock(), Mock()

    callback = Mock()
    model = HeteroscedasticCNN(kernel_size=3, rngs=nnx.Rngs(0))
    battery_train.run_training(
        model=model,
        hp=TrainHyperparameters(window_size=2, batch_size=2, n_epochs=1),
        mlflow_setup=MLFlowSetup(run_name="test", experiment_name="test"),
        raw_data_dir=tmp_path,
        data_gen_run_id="data-run",
        log_stream=io.StringIO(),
        step_factory=step_factory,
        callbacks=(callback,),
    )

    assert factory_sizes == [3]
    assert captured["callbacks"][-1] is callback
    assert captured["train_batch_fn"] is not captured["eval_batch_fn"]


def test_run_adversarial_training_wires_steps_and_logs_loss_weight(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = pd.DataFrame(
        {
            "run_id": [0] * 4 + [1] * 4,
            "time": [0, 1, 2, 3] * 2,
            "voltage": [4.2, 4.0, 3.8, 3.6] * 2,
        }
    )
    data.to_csv(tmp_path / "train.csv", index=False)
    monkeypatch.setattr(
        battery_train.mlflow,
        "get_run",
        lambda run_id: SimpleNamespace(
            data=SimpleNamespace(params={"n_histories_train": "1"})
        ),
    )

    @contextlib.contextmanager
    def fake_tracking(*, setup):
        yield SimpleNamespace()

    log_params = Mock()
    monkeypatch.setattr(battery_train, "track_mlflow", fake_tracking)
    monkeypatch.setattr(battery_train.mlflow.sklearn, "log_model", Mock())
    monkeypatch.setattr(battery_train.mlflow, "log_params", log_params)
    monkeypatch.setattr(battery_train.mlflow, "log_param", Mock())
    monkeypatch.setattr(battery_train.mlflow, "log_text", Mock())
    monkeypatch.setattr(battery_train, "count_parameters", lambda model: 1)

    captured: dict = {}
    monkeypatch.setattr(
        battery_train,
        "adversarial_train_loop",
        lambda **kwargs: captured.update(kwargs),
    )
    generator_step, discriminator_step, eval_step = Mock(), Mock(), Mock()
    callback = Mock()
    hp = QAVITrainHyperparameters(
        window_size=2,
        batch_size=2,
        n_epochs=1,
        adversarial_loss_weight=0.25,
    )

    battery_train.run_adversarial_training(
        model=HeteroscedasticCNN(kernel_size=3, rngs=nnx.Rngs(0)),
        discriminator=nnx.Linear(3, 1, rngs=nnx.Rngs(1)),
        hp=hp,
        mlflow_setup=MLFlowSetup(run_name="test", experiment_name="test"),
        raw_data_dir=tmp_path,
        data_gen_run_id="data-run",
        log_stream=io.StringIO(),
        generator_batch_fn=generator_step,
        discriminator_batch_fn=discriminator_step,
        eval_batch_fn=eval_step,
        callbacks=(callback,),
    )

    assert captured["generator_batch_fn"] is generator_step
    assert captured["discriminator_batch_fn"] is discriminator_step
    assert captured["eval_batch_fn"] is eval_step
    assert captured["callbacks"][-1] is callback
    assert log_params.call_args.args[0]["adversarial_loss_weight"] == pytest.approx(
        0.25
    )


def test_track_conv_weights_variance(monkeypatch: pytest.MonkeyPatch) -> None:
    model = Mock()
    model.conv_mean_posterior_variance.return_value = 0.25
    log_metric = Mock()
    monkeypatch.setattr("qmodem.callbacks.mlflow.log_metric", log_metric)
    context = SimpleNamespace(model=model, epoch=7)

    track_conv_weights_variance(TrainingPhase.EPOCH_END, context)

    log_metric.assert_called_once_with("conv_weights_variance", 0.25, step=7)


@pytest.mark.parametrize(
    "script_path",
    [
        "scripts/battery/bnn_train.py",
        "scripts/battery/hnn_train.py",
        "scripts/battery/mcd_train.py",
        "scripts/battery/qavi_train.py",
        "scripts/battery_multiple_scenarios/bnn_train.py",
        "scripts/battery_multiple_scenarios/hnn_train.py",
        "scripts/battery_multiple_scenarios/mcd_train.py",
        "scripts/battery_multiple_scenarios/qavi_train.py",
    ],
)
def test_training_script_imports(script_path: str) -> None:
    runpy.run_path(script_path, run_name="__test__")
