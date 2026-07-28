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

Run one of the `generate_discharge_history.py` scripts.

### Setup of the environment file (`.env`)

1. Specify `RAW_DATA_DIR`, i.e. the location where to write the battery simulation data. For the multiple loading scenarios case, you can do the same with `RAW_DATA_DIR_MULTI`.
2. Copy-paste the MLFlow run ID to `DATA_GEN_RUN_ID` (`DATA_GEN_RUN_ID_MULTI` for the multiple scenarios case).
3. Set `MLFLOW_USE_LAST_TRAINED=true` if you plan to run the test script right after the corresponding training one. If `MLFLOW_USE_LAST_TRAINED=true`, the program will prompt you to provide the experiment name and run ID for MLFlow to retrieve the correct run.
4. Set a meaningful `MLFLOW_EXPERIMENT_NAME` for your numerical campaign.

### Training/testing

1. Make sure the data genration run ID is the correct one.
2. Run one of the `_train.py` scripts to train the network.
3. Run the corresponding test script.

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for full guidelines. Quick reference:

```bash
uv run pre-commit install            # install hooks (once, before contributing)
uv run pytest                        # run all tests
```
