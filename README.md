# QMoDeM: Quantum-aided Models for Decision-Making

## Installation

```bash
pip install -e .
# or with uv:
uv sync
```

## Running the scripts

> Scripts log to MLFlow, a machine learning experiment tracking software. This design choice allows to compare runs easily in the same UI.

### Data generation

Run `scripts/battery_multiple_scenarios/generate_discharge_histories.py`.

### Setup of the environment file (`.env`)

1. Specify `RAW_DATA_DIR_MULTI`, i.e. the location where to write the battery simulation data.
2. Copy-paste the MLFlow run ID of the data-generation run to `DATA_GEN_RUN_ID_MULTI`.
3. Set a meaningful `MLFLOW_EXPERIMENT_NAME` for your numerical campaign.

### Training/testing

1. Make sure the data generation run ID is the correct one.
2. Run `train_backbone.py --seed r` to train the backbone run of protocol seed r.
3. Run `train_method_run.py --method m --seed r` to train a method run on it.
4. Run `evaluate.py --method m --seed r` to evaluate it on the test histories.

### Hyperparameter tuning

Studies run on protocol seed 0, in order: `run_mean_study.py` (backbone sizes and mean LR), then `train_backbone.py --seed 0` with the selected sizes, then `run_variance_study.py --seed 0` (variance LR on that backbone's frozen mean). Then one `run_posterior_study.py --method m` per method with a posterior stage (MCD, BBB, pSGLD, Q-AVI, C-AVI) on seed 0's backbone. Protocol runs, seed 0 included, are re-trained from the frozen selections: `train_backbone.py --seed r --mean-study-run-id <id> --variance-study-run-id <id>` and `train_method_run.py --method m --seed r --study-run-id <id>`. Each study is one `role=hpo_study` MLflow run that logs the selected hyperparameters as `best.<name>` params (e.g. `best.mean/network/kernel_size`), with one `role=hpo_trial` child run per trial. Pass `--n-trials` / `--max-epochs` only for smoke tests.

### Run selection

Scripts that consume a run take either `--run-id <id>` (that exact run) or tag criteria: `--seed r` selects a backbone run (`role=backbone`), `--method m --seed r` a method run (`role=method`). Tag selection returns the newest `FINISHED` match. It raises if nothing matches, if a criterion the role needs is missing, or if two matches share the newest start time. HPO runs (`role=hpo_study`, `role=hpo_trial`) are never selected by tags.

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for full guidelines. Quick reference:

```bash
uv run pre-commit install            # install hooks (once, before contributing)
uv run pytest                        # run all tests
```
