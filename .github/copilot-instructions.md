# Copilot instructions for QMoDeM

## Project overview

QMoDeM is a Python 3.12 project for uncertainty-aware and quantum-aided time-series
regression, currently focused on battery remaining-useful-life (RUL) prediction. The
installable library lives under `src/qmodem`; `scripts/` contains executable
experiment entry points, and `tests/` contains pytest coverage for both the reusable
library and the battery workflows.

### Use cases
There are three use cases considered so far in QMoDeM, which are available from the `scripts/` directory:
1. Battery discharge with a single constant amplitude load -> `scripts/battery`
2. Battery discharge with a probabilistic mixture of two possible variable loads -> `scripts/battery_multiple_scenarios`
3. CMAPSS turbofan engine degradation -> `scripts/cmapss`

> **NOTE: unmaintained code**: `scripts/cmapss` and `scripts/battery` are not maintained currently. Also unmaintained is any other code related to CMAPSS. All of this logic and implementation can be for now safely ignored when developing QMoDeM.

### Uncertainty modeling techniques
- HNN: Heteroscedastic Neural Network, a deterministic neural network with a Gaussian head that predicts both the mean and variance of the target distribution.
- MCD: Monte Carlo Dropout, a Bayesian neural network that uses dropout at inference time to sample from the posterior distribution of the model parameters.
- BNN: Bayesian Neural Network, a Bayesian neural network that uses mean-field variational inference to approximate the posterior distribution of the model parameters.
- QAVI: Quantum Adversarial Variational Inference, a quantum-classical hybrid model that uses a quantum circuit and possibly classical postprocessing to generate samples from the posterior distribution of the model parameters. An adversarial training loop trains the quantum circuit to maximize the lieklihood of the data, while at the same time fooling a classical discriminator which tries to distinguish between the true and predicted distributions.

### Training flow -> `scripts/battery_multiple_scenarios/run_train.py`

1. Battery scripts load `.env`, construct model and hyperparameter objects, and set up
   an MLflow run.
2. `qmodem.battery.data_processing.prepare_data` reads `train.csv`, uses the
   data-generation MLflow run's `n_histories_train` parameter to split histories into
   train/validation sets, and applies a `DataPipeline`.
3. The pipeline scales training data in `FIT_TRANSFORM` mode, then switches to
   `TRANSFORM` for validation so validation never fits its own scaler. Data is exposed
   through Grain `DataLoader` builders.
4. `qmodem.train.train_loop` runs standard supervised training; Bayesian models use
   ELBO/NLL step factories, while QAVI uses
   `qmodem.train_adversarial.train_loop` with generator and discriminator steps.
5. Training callbacks report/log losses and persist the best parameter state to MLflow.

The shared model layer in `qmodem.battery.models` is a 1D CNN with convolution,
activation, dropout, global-average pooling, and a Gaussian mean/positive-variance
head. Convolutions may be deterministic, Bayesian, or quantum-generated. The
quantum-generated path is built from PennyLane circuit factories and weight
generators in `qmodem.quantum_circuits` and `qmodem.battery.models`.

### Hyperparameter tuning -> `scripts/battery_multiple_scenarios/run_hpo.py`
Hyperparameter optimization (HPO) is implemented as an Optuna study, where the objective function is essentially a wrapper around the training main function. The main difference is that the HPO's objective also takes care of computing the score of the HPO iteration, which is the average CRPS across validation histories.

### Testing flow -> `scripts/battery_multiple_scenarios/run_test.py`

Testing is always meant to happen after a training run has happened.

1. Environment loading and logging setup happen as for training.
2. Since the training run is already complete, the test script retrieves the training run ID from `.last_trained.json` and reloads that run. This allows to load the training run parameters. the data scalers and the trained model state, as well as to log test metrics to the same MLflow run.
3. Because the data is generated, the test phase also takes care of reconstreconstrucing the true RUL distribution (`qmodem.battery.data_processing:reconstruct_rul_distribution`) for each test history. It does that by selecing a grid of intermediate state of charges (SoCs) and running Monte Carlo simulations from each intermediate SoC to the end of discharge.
4. The bulk of the test phase is to run `qmodem.battery.evaluate:run_evaluation`, which uploads the data of each test history (`test_case`s), selects the evaluation time windows and RUL targets, predicts with the trained model and computes all the metrics of interest. The metrics computed are: RMSE, coverage, weighted spread of uncertainty (WSU) and continuous-ranked probability score (CRPS). The metrics are computed and logged both per test case and averaged across test cases to provide a global view of the model performance.

### Standard vs adversarial training

Training can happen in a standard fully supervised way, as in `scripts/battery_multiple_scenarios/run_train.py`, or in an adversarial way, as in `scripts/battery_multiple_scenarios/run_train_adversarial.py`.

The main difference is that the adversarial training uses a generator and a discriminator, which are trained in an adversarial way. The generator is the model that predicts the RUL, while the discriminator is a model that tries to distinguish between the true RUL distribution and the predicted RUL distribution. The generator is trained to fool the discriminator, while the discriminator is trained to correctly classify the true and predicted RUL samples.

Adversarial training currently serves only the QAVI method.

## Environment and configuration

- Use `uv`; the project requires Python `>=3.12` and `.python-version` pins `3.12`.
- Install/sync dependencies with `uv sync` (CI uses
  `uv sync --locked --all-extras --dev`).
- Copy `.env.example` to `.env` before running battery experiments. Battery scripts
  require absolute raw-data paths, data-generation MLflow run IDs, and MLflow backend
  and artifact-store settings. Do not commit `.env`, generated data, checkpoints,
  model outputs, or MLflow state.
- Scripts expect `train.csv` and `test.csv` in the configured raw-data directory.
  `generate-all` creates both single- and multi-scenario histories before training.

## Code Testing

Code testing does not have a consistent convention at the moment, since some tests are organized by classes and some are standalone functions.

When adding new tests or updating old ones, pytest and the *functional* convention should be used.

## Commands

Run from the repository root:

```bash
uv sync --locked --all-extras --dev
uv run pytest
uv run pre-commit install
uv run pre-commit run --all-files
```

The CI workflows run `uv run pytest` and `uv run pre-commit run --all-files`.
Pre-commit includes trailing-whitespace, EOF, YAML, large-file checks, Ruff
check-with-fixes, Ruff format, and docformatter configured from `pyproject.toml`.

## Repository-specific conventions

- Keep reusable code in `src/qmodem` and use `scripts/` only for experiment,
  comparison, or dataset-specific orchestration. Preserve the `src` layout and
  import library code as `qmodem...`.
- JAX PRNG keys are explicit inputs. Split keys per batch/sample and pass them into
  Flax NNX modules through `nnx.Rngs`; do not introduce implicit global randomness.
- NNX modules are mutable stateful objects. Training/evaluation mode matters:
  `CNN.eval()` deliberately keeps MCD models in training mode so dropout remains
  stochastic for Monte Carlo predictions.
- Model outputs for the heteroscedastic head are `[mean, positive_variance]`.
  NLL-based steps clip variance to a small positive value; preserve this convention
  when adding losses or sampling code.
- Training loops communicate through `TrainingPhase` callbacks
  (`INIT`, `EPOCH_START`, `EVAL_START`, `EPOCH_END`, `BEFORE_RETURN`). Add reporting,
  checkpointing, or metrics as callbacks rather than embedding unrelated side effects
  in the loop.
- During the test phase, the input time window is the one one time-step before the SoC of interest, while the target RUL is at the time of the SoC of interest.
- Bayesian training adds KL divergence normalized by the number of training samples;
  step factories receive `StandardStepFactoryContext` for this dataset-size
  dependency. QAVI has separate generator/discriminator optimizers and steps.
- Battery data is grouped by `run_id`/history. Windows must not cross history
  boundaries, and RUL targets are aligned to the end of each window. Preserve
  train-only scaler fitting when changing preprocessing.
- MLflow is the experiment-tracking boundary: training logs metrics and best model
  state there, while test scripts retrieve training setup/parameters using the
  configured run ID or `.last_trained.json`.
- Prefer typed protocols/dataclasses and existing factories (`ConvType`,
  `Method`, step factories, data pipelines) over ad-hoc branching. Follow the
  existing Ruff, docformatter, and Google-style docstring configuration.
