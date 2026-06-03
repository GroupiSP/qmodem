from enum import StrEnum
from functools import partial
from multiprocessing import Pool

import jax
import mlflow
import optax
import optuna
import pandas as pd
from flax import nnx
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from sklearn.preprocessing import StandardScaler

from qmodem import LSTM, create_dataloaders
from qmodem.data import CMAPSSDataSource
from qmodem.metrics import compute_point_crps
from qmodem.tracking import MLFlowSetup, Tags, track_mlflow
from qmodem.train import (
    EarlyStopper,
    TrainingContext,
    train_loop,
)
from qmodem.utils import PROCESSED_DATA_DIR_PATH


class Hyperparameter(StrEnum):
    DROPOUT_RATE = "dropout_rate"
    HIDDEN_SIZE = "hidden_size"
    LR = "lr"
    WINDOW_SIZE = "window_size"


BATCH_SIZE = 10
N_EPOCHS = 1000
NUM_SENSORS = 9
PATIENCE = 10
PRINT_EVERY = 10
N_TRIALS = 20
N_PROCESSES = 5
SEED = 42


def load_datasets() -> tuple[CMAPSSDataSource, CMAPSSDataSource]:
    train_df = pd.read_csv(PROCESSED_DATA_DIR_PATH / "cmapss_fd001_train_train.csv")
    val_df = pd.read_csv(PROCESSED_DATA_DIR_PATH / "cmapss_fd001_train_val.csv")

    return train_df, val_df


def report_condition_every(context: TrainingContext) -> bool:
    return (context.epoch + 1) % PRINT_EVERY == 0 or context.epoch == 0


def reporter(context: TrainingContext) -> None:
    print(
        f"Epoch {context.epoch + 1:3d}/{N_EPOCHS} | "
        f"Train Loss: {context.train_loss:.6f} | "
        f"Val Loss: {context.val_loss:.6f} | "
        f"Best Val Loss: {context.best_val_loss:.6f}"
    )


def mse_loss(model: nnx.Module, batch: jax.Array):
    x, y = batch
    y_pred = model(x, rngs=nnx.Rngs(0))
    return ((y - y_pred) ** 2).mean()


@nnx.jit
def train_step(model: nnx.Module, optimizer: nnx.Optimizer, batch: jax.Array):
    def loss_fn(model):
        return mse_loss(model, batch)

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss


@nnx.jit
def eval_step(model: nnx.Module, batch: jax.Array):
    return mse_loss(model, batch)


def objective(trial: optuna.Trial, parent_run_id: str) -> float:
    with mlflow.start_run(nested=True, parent_run_id=parent_run_id):
        scaler = StandardScaler()

        train_df, val_df = load_datasets()

        window_size = trial.suggest_int(Hyperparameter.WINDOW_SIZE, 10, 50)

        ds_train = CMAPSSDataSource(
            train_df, train_or_test="train", scaler=scaler, window_size=window_size
        )
        ds_val = CMAPSSDataSource(
            val_df, train_or_test="test", scaler=scaler, window_size=window_size
        )

        dl_train, dl_val = create_dataloaders(
            ds_train=ds_train,
            ds_val=ds_val,
            batch_size=BATCH_SIZE,
            seed_train=SEED,
            seed_val=SEED + 1,
            shuffle_train=False,
            shuffle_val=False,
        )

        model = LSTM(
            input_size=NUM_SENSORS,
            hidden_size=trial.suggest_int(Hyperparameter.HIDDEN_SIZE, 20, 100),
            dropout_rate=trial.suggest_float(Hyperparameter.DROPOUT_RATE, 0.1, 0.5),
            rngs=nnx.Rngs(0),
        )

        schedule = optax.cosine_decay_schedule(
            init_value=trial.suggest_float(Hyperparameter.LR, 1e-4, 1e-2, log=True),
            decay_steps=N_EPOCHS * (len(ds_train) // BATCH_SIZE),
            alpha=0.1,
        )

        optimizer = nnx.Optimizer(model, optax.adam(schedule), wrt=nnx.Param)
        early_stopper = EarlyStopper(patience=PATIENCE, min_delta=1e-4)

        model_state_best = jax.tree.map(
            lambda x: x, nnx.state(model, nnx.Param)
        )  # initial model state copy

        def on_validation_improvement():
            global model_state_best
            model_state_best = jax.tree.map(lambda x: x, nnx.state(model, nnx.Param))

        train_loop(
            n_epochs=N_EPOCHS,
            dataloader_train=dl_train,
            dataloader_val=dl_val,
            train_batch_fn=lambda batch: train_step(model, optimizer, batch),
            eval_batch_fn=lambda batch: eval_step(model, batch),
            early_stopper=early_stopper,
            report_condition=report_condition_every,
            reporter=reporter,
            on_train_epoch_start=model.train,
            on_val_epoch_start=model.eval,
            on_validation_improvement=on_validation_improvement,
        )

        nnx.update(model, model_state_best)

        model.train()  # enable dropout for stochastic predictions
        mean_crps = compute_point_crps(ds_val, model)

        # Log to MLFlow
        mlflow.log_params(trial.params)
        mlflow.log_metric("mean_crps", mean_crps)

    return mean_crps


def run_optimization(sampler_seed: int, parent_run_id: str) -> optuna.Study:
    # The sampler seed needs to be different for every process to avoid identical trials.
    hp_sampler = optuna.samplers.RandomSampler(seed=sampler_seed)

    study = optuna.create_study(
        sampler=hp_sampler,
        direction="minimize",
        storage=JournalStorage(JournalFileBackend("hpo.log")),
        load_if_exists=True,  # important for multiprocessing to avoid race conditions
    )
    study.optimize(
        partial(objective, parent_run_id=parent_run_id),
        # TODO: make n_trials an argument to handle non divisible cases.
        n_trials=N_TRIALS // N_PROCESSES,
    )
    return study


def main():
    tracking_setup = MLFlowSetup(
        run_name="hyperparameter_optimization",
        experiment_name="cmapss_lstm_optuna_random_search_mp",
        tags=Tags(),  # choices here correspond to the defaults
    )

    with track_mlflow(tracking_setup) as parent_run:
        args = [(SEED + i, parent_run.info.run_id) for i in range(N_PROCESSES)]
        with Pool(processes=N_PROCESSES) as pool:
            studies = pool.starmap(run_optimization, args)

        best_study = min(studies, key=lambda s: s.best_value)

        mlflow.log_params(
            best_study.best_trial.params
        )  # log best trial params to the parent run

        # log general parameters
        mlflow.log_param("n_trials", N_TRIALS)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("n_epochs", N_EPOCHS)
        mlflow.log_param("seed", SEED)
        mlflow.log_param("n_processes", N_PROCESSES)


if __name__ == "__main__":
    main()
