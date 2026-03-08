# QMoDeM: Quantum-aided Models for Decision-Making

## Installation

```bash
pip install -e .
# or with uv:
uv pip install -e .
```

## CLI Usage

QMoDeM provides a unified command-line interface for data generation, training, and evaluation.

### Generate data

```bash
qmodem generate-data [OPTIONS]
```

| Option | Default | Description |
|---|---|---|
| `--n-histories-train` | 100 | Number of discharge histories for training |
| `--n-histories-val` | 20 | Number of discharge histories for validation |
| `--n-test-cases` | 1 | Number of test cases to generate |
| `--n-simu` | 500 | Stochastic simulations per intermediate SoC |
| `--n-intermediate-socs` | 200 | Intermediate SoC evaluation points per test case |
| `--seed-train` | — | Random seed for training data |
| `--seed-val` | — | Random seed for validation data |
| `--seed-test` | — | Random seed for test data |
| `--output-dir` | `data` | Output directory |

### Train a model

```bash
qmodem train METHOD [OPTIONS]
```

`METHOD` is one of: `bayes_cnn`, `het_cnn`, `mcd_cnn`, `qavi_cnn`.

| Option | Default | Description |
|---|---|---|
| `--n-epochs` | 500 | Maximum training epochs |
| `--lr` | 0.01 | Learning rate |
| `--batch-size` | 32 | Mini-batch size |
| `--patience` | 50 | Early-stopping patience |
| `--print-every` | 10 | Print interval (epochs) |
| `--n-filters` | 4 | Number of CNN filters |
| `--kernel-size` | 5 | CNN kernel size |
| `--window-size` | 20 | Input window size |
| `--stride` | 10 | Sliding-window stride |
| `--no-normalize` | — | Disable target normalisation |
| `--train-data-path` | `data/train.npz` | Path to training data |
| `--val-data-path` | `data/val.npz` | Path to validation data |
| `--dropout-rate` | 0.1 | Dropout rate (`mcd_cnn` only) |
| `--n-qubits` | 6 | Number of PQC qubits (`qavi_cnn` only) |
| `--n-pqc-layers` | 1 | Number of PQC layers (`qavi_cnn` only) |

### Evaluate a model

```bash
qmodem test METHOD [OPTIONS]
```

| Option | Default | Description |
|---|---|---|
| `--test-data-path` | `data/test_case_0.npz` | Path to test data |
| `--n-samples` | 500 | Forward passes / weight samples |
| `--output-dir` | `saved/<method>/test` | Output directory for plots |

### Compare methods

```bash
qmodem compare [OPTIONS]
```

| Option | Default | Description |
|---|---|---|
| `--methods` | all | Methods to compare (repeat for each) |
| `--test-data-path` | `data/test_case_0.npz` | Path to test data |
| `--n-samples` | 500 | Forward passes / weight samples |
| `--output-dir` | `saved/compare` | Output directory for the comparison plot |

### Examples

```bash
# Full workflow
qmodem generate-data --n-histories-train 200
qmodem train het_cnn --n-epochs 300 --lr 0.0005
qmodem test het_cnn --n-samples 1000

# Bayesian CNN with custom settings
qmodem train bayes_cnn --n-filters 32 --patience 100
qmodem test bayes_cnn

# QAVI CNN
qmodem train qavi_cnn --n-qubits 6 --n-pqc-layers 3
qmodem test qavi_cnn

# Compare two methods on the same test case
qmodem compare --methods het_cnn --methods mcd_cnn --n-samples 200
```

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for full guidelines. Quick reference:

```bash
uv run pre-commit install            # install hooks (once, before contributing)
uv run pytest                        # run all tests
```
