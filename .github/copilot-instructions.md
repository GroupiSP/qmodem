# Copilot instructions for QMoDeM

## Project overview

QMoDeM is a Python 3.12 project for uncertainty-aware and quantum-aided time-series
regression, currently focused on battery remaining-useful-life (RUL) prediction. The
installable library lives under `src/qmodem`; `scripts/` contains executable
experiment entry points, and `tests/` contains pytest coverage for both the reusable
library and the battery workflows.

### Use case
QMoDeM currently considers a single use case, available from the `scripts/` directory:
battery discharge with a probabilistic mixture of two possible variable loads -> `scripts/battery_multiple_scenarios`.

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
4. Backbone runs (`qmodem.battery.train.train_backbone`) train stagereg's
   `MeanStage` and `GammaVarianceStage`; method runs
   (`qmodem.battery.method_run.train_method_run`) restore them and train the shared
   posterior stage (`qmodem.battery.posterior`) with the method's inference method.
5. Training callbacks report/log losses and persist the best parameter state to MLflow.

The networks in `qmodem.battery.models` are the mean network
(`Conv1D -> activation -> GAP -> Linear`) and the variance network (same backbone,
stagereg `GammaHead`), plus the AVI conv-layer generator (`ConvGenerator`) and
discriminator (`WindowDiscriminator`). Posteriors are inferred over the conv layer
only (`qmodem.battery.posterior.INFERRED_SUBNETWORK`). PennyLane circuit factories
live in `qmodem.quantum_circuits`.

### Hyperparameter tuning -> `scripts/battery_multiple_scenarios/run_{mean,variance}_study.py`
Hyperparameter optimization (HPO) follows protocol §5: one Optuna study per stage (and per method for posterior studies). `qmodem.hpo.run_study` records a study as one `role=hpo_study` MLflow run (tags `stage`, `method`/`backbone_run_id` where applicable, `n_trials`; params `best.<name>`) with one `role=hpo_trial` child per trial (tag `hpo.trial=<n>`). The objectives in `qmodem.battery.hpo` train their stage through the same operations as backbone runs (`train_mean_stage`, `train_variance_stage`) and report the per-epoch validation loss to a median pruner. The legacy `run_hpo.py` is not maintained.

### Method runs and evaluation -> `scripts/battery_multiple_scenarios/train_method_run.py`, `evaluate.py`

A method run is an MLflow child (`mlflow.parentRunId`) of a backbone run, tagged `role=method`, `method` and `seed` (`qmodem.battery.method_run`). It restores the backbone's finalized `mean`/`variance` checkpoints and adds the method's posterior stage (none for HNN). Methods are registered in `METHOD_CONFIGS`, keyed by `Method`.

Evaluation always happens after a method run exists, in a separate process:

1. The evaluate script selects the method run (`--run-id`, or `--method` + `--seed`), restores it and loads the scalers from its parent backbone run.
2. It reconstructs the simulator's reference RUL samples for each test history (`qmodem.battery.data_generation:reconstruct_true_rul_distribution`) at `n_timestamps` intermediate SoCs.
3. `qmodem.battery.evaluate` draws M = 100 predictive samples per timestamp and computes, per test case and with stagereg metric functions only, expected CRPS, RMSE, coverage and WSU. The first timestamp (t0) is dropped from every metric and is only WSU's origin. Metrics are logged into the method run per case (`evaluation/test/<case>/<metric>`), as unweighted means over cases (`evaluation/test/<metric>`) and per policy (`evaluation/test/policy_<p>/<metric>`).

### AVI (Q-AVI, C-AVI)

`qmodem.avi` holds the dataset-agnostic `AVI` inference method (stagereg's `InferenceMethod` seam), its implicit posterior and loss, the `LatentMap` protocol and the quantum latent map. A generator (latent map -> postprocessor) maps `z ~ U(0, 2π)` to one draw of the conv layer; it is fitted adversarially against a discriminator that judges (window, target) pairs, with `k` discriminator steps per generator step. The battery generator and discriminator live in `qmodem.battery.models`; the method configurations in `qmodem.battery.methods.avi`. Q-AVI and C-AVI differ only in their latent map.
