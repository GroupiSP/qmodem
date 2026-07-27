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

1. Start by setting up your `.env` file. Specify `RAW_DATA_DIR`, i.e. the location where to write the battery simulation data.
2. Run one of the `generate_discharge_history.py` scripts.
3. Write the data generation run ID from mlflow to `.env`.

### Training/testing

1. Make sure the data genration run ID is the correct one.
2. Run one of the `_train.py` scripts to train the network.
3. Copy the training run ID to `TRAIN_RUN_ID` in the corresponding `_test.py` file.
4. Run the test script

The logic of steps 3 and 4 is to write the test results in the same training run entry, making it possible to compare different runs in the MLFlow UI.

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for full guidelines. Quick reference:

```bash
uv run pre-commit install            # install hooks (once, before contributing)
uv run pytest                        # run all tests
```
