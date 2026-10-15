# QMoDeM: Quantum-aided Models for Decision-Making

QMoDeM compares quantum and classical inference methods for uncertainty-aware remaining-useful-life (RUL) regression on simulated battery discharges. Training runs are built on [stagereg](https://github.com/GroupiSP/stagereg) (stages, posterior stage, inference methods) and tracked in MLflow. The comparison is fixed by [docs/comparison-protocol.md](docs/comparison-protocol.md); the vocabulary is in [GLOSSARY.md](GLOSSARY.md).

## Installation

```bash
uv sync
# or, without uv:
pip install -e .
```

This installs `stagereg[jax]==0.2.0` together with a pinned JAX/Flax window:

| Pin | Why |
|---|---|
| `stagereg[jax]==0.2.0` | The posterior stage, early stopper, checkpoint cache and metrics qmodem relies on are those of 0.2.0. |
| `jax>=0.10,<0.11` | PennyLane (Q-AVI's quantum latent map) breaks on JAX 0.11. |
| `flax>=0.12.8,<0.12.9` | The tested Flax release for stagereg 0.2.0. Older Flax breaks stagereg's optimizer restore (this is why `jax-ai-stack`, which pins Flax 0.12.0, is no longer a dependency). |

## Configuration (`.env`)

Scripts read their configuration from the environment, falling back to a `.env` file (copy `.env.example`):

| Variable | Meaning |
|---|---|
| `RAW_DATA_DIR_MULTI` | Directory where the battery simulation data (`train.csv`, `test.csv`) is written and read. |
| `DATA_GEN_RUN_ID_MULTI` | MLflow run ID of the data-generation run. |
| `MLFLOW_EXPERIMENT_NAME` | MLflow experiment of your numerical campaign; run selection searches only this experiment. |
| `MLFLOW_BACKEND_STORE` | MLflow backend store URI (default `sqlite:///mlflow.db`). |
| `MLFLOW_ARTIFACT_STORE` | MLflow artifact root (default `mlruns/`). |

Except for data generation, values already set in the environment win over `.env`, so a run can be pointed at another backend, e.g. `MLFLOW_EXPERIMENT_NAME=smoke uv run python scripts/battery_multiple_scenarios/train_backbone.py --seed 0`.

## Scripts

All scripts live in `scripts/battery_multiple_scenarios/` and log to MLflow. Run `uv run python <script> --help` for every flag (not on `generate_discharge_histories.py`, which parses no arguments and would start generating).

| Script | Creates | Purpose |
|---|---|---|
| `generate_discharge_histories.py` | data-generation run | Simulates the train/test discharge histories. Takes no flags (it runs immediately, and `.env` overrides the environment). |
| `train_backbone.py` | backbone run (`role=backbone`) | Trains the mean and variance stages of one protocol seed. |
| `train_method_run.py` | method run (`role=method`) | Restores a backbone run and trains one method's posterior stage (none for HNN). |
| `evaluate.py` | metrics in the method run | Evaluates a method run on the test histories (CRPS, RMSE, coverage, WSU). |
| `run_mean_study.py` | HPO study (`role=hpo_study`, `stage=mean`) | Tunes the mean network and its learning rate. |
| `run_variance_study.py` | HPO study (`stage=variance`) | Tunes the variance learning rate on a backbone run's frozen mean. |
| `run_posterior_study.py` | HPO study (`stage=posterior`, `method=m`) | Tunes one method's posterior stage on a backbone run. |

Method tag values (`--method`): `hnn`, `mcd`, `bbb`, `psgld`, `q-avi`, `c-avi`. Posterior studies take all of them except `hnn`.

### Run layout

```
data-generation run
backbone run          role=backbone  seed=r  data_generation_run_id
│     checkpoints/mean/best, checkpoints/variance/best, x_scaler, y_scaler
└── method run        role=method    method=m  seed=r   (mlflow.parentRunId = backbone run)
      checkpoints/posterior/best, evaluation/test/... metrics
HPO study run         role=hpo_study stage=mean|variance|posterior [method, backbone_run_id, seed]
└── trial run         role=hpo_trial hpo.trial=n  hpo.state=complete|pruned|fail
      params best.<name> on the study run, e.g. best.mean/network/kernel_size
```

### Run selection

Scripts that consume a run take either an exact run ID or tag criteria:

| Script | Exact run | Tag criteria |
|---|---|---|
| `train_method_run.py`, `run_variance_study.py`, `run_posterior_study.py` | `--backbone-run-id <id>` | `--seed r` (backbone run of seed r) |
| `evaluate.py` | `--run-id <id>` | `--method m --seed r` (method run) |

Tag selection returns the newest `FINISHED` match in the configured experiment. It raises if nothing matches, if a criterion the role needs is missing, or if two matches share the newest start time. HPO runs are never selected by tags: pass study runs by ID (`--mean-study-run-id`, `--variance-study-run-id`, `--study-run-id`).

## Running the protocol

Set `S=scripts/battery_multiple_scenarios`. `<MEAN>`, `<VAR>`, `<POST_m>` stand for study run IDs printed by the scripts (and visible in MLflow).

1. **Data.** Set `RAW_DATA_DIR_MULTI` and `MLFLOW_EXPERIMENT_NAME`, run `uv run python $S/generate_discharge_histories.py`, then copy its run ID to `DATA_GEN_RUN_ID_MULTI`.
2. **HPO** (protocol §5), all on protocol seed 0, in this order: mean → variance → posterior studies.

   ```bash
   uv run python $S/run_mean_study.py                                # mean study <MEAN>
   uv run python $S/train_backbone.py --seed 0 --mean-study-run-id <MEAN> \
       --run-name hpo-backbone-seed-0                                # tuned mean, default variance LR
   uv run python $S/run_variance_study.py --seed 0                   # variance study <VAR>
   uv run python $S/run_posterior_study.py --method mcd --seed 0     # posterior study <POST_mcd>
   # ... likewise for bbb, psgld, q-avi and c-avi
   ```

   The variance study and the posterior studies select seed 0's newest finished backbone run; pass `--backbone-run-id` to pin one.
3. **Protocol runs**, for every protocol seed r (0 included), re-trained from the frozen `best.*` params of the studies:

   ```bash
   uv run python $S/train_backbone.py --seed r \
       --mean-study-run-id <MEAN> --variance-study-run-id <VAR>
   uv run python $S/train_method_run.py --method hnn --seed r
   uv run python $S/train_method_run.py --method m --seed r --study-run-id <POST_m>
   uv run python $S/evaluate.py --method m --seed r
   ```

Each study is one `role=hpo_study` run whose selected hyperparameters are params `best.<name>` (e.g. `best.mean/network/kernel_size`), with one `role=hpo_trial` child run per trial. Studies keep Optuna's state in memory unless `--optuna-storage` is given.

### Smoke tests

Shrink the budgets, never the code. Point `MLFLOW_BACKEND_STORE`, `MLFLOW_ARTIFACT_STORE` and `MLFLOW_EXPERIMENT_NAME` at a scratch store so the campaign's runs stay clean:

```bash
uv run python $S/run_mean_study.py --n-trials 2 --max-epochs 3
uv run python $S/train_backbone.py --seed 0 --mean-epochs 3 --variance-epochs 3 --mean-study-run-id <MEAN>
uv run python $S/run_variance_study.py --seed 0 --n-trials 2 --max-epochs 3
uv run python $S/run_posterior_study.py --method psgld --seed 0 --n-trials 2 --max-epochs 4 \
    --posterior-param burn_in_epochs=2
uv run python $S/train_method_run.py --method bbb --seed 0 --max-epochs 5
uv run python $S/evaluate.py --method bbb --seed 0
```

`--posterior-param NAME=VALUE` (repeatable, nested with `/`, e.g. `avi/n_qubits=3`) overrides one posterior hyperparameter; `--max-epochs` / `--patience` override the posterior stage's schedule.

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for full guidelines. Quick reference:

```bash
uv run pre-commit install            # install hooks (once, before contributing)
uv run pre-commit run --all-files    # lint and format
uv run pytest                        # run all tests
```
