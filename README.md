# QMoDeM: Quantum-aided Models for Decision-Making

## Installation

```bash
pip install -e .
# or with uv:
uv sync
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
| `--output-dir` | `saved` | Base directory for checkpoints and metadata |

### Evaluate a model

```bash
qmodem test METHOD [OPTIONS]
```

| Option | Default | Description |
|---|---|---|
| `--test-data-path` | `data/test_case_0.npz` | Path to test data |
| `--n-samples` | 500 | Forward passes / weight samples |
| `--output-dir` | `saved/<method>/test` | Output directory for plots |
| `--trained-dir` | `saved` | Base directory with trained model artefacts |

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
| `--trained-dir` | `saved` | Base directory with trained model artefacts |

### Compare methods (box plot)

```bash
qmodem compare-box [OPTIONS]
```

| Option | Default | Description |
|---|---|---|
| `--methods` | all | Methods to compare (repeat for each) |
| `--test-case` | all `test_case_*.npz` in `--data-dir` | Test-case indices to include (repeat for each) |
| `--data-dir` | `data` | Directory containing test-case files |
| `--n-samples` | 500 | Forward passes / weight samples |
| `--output-dir` | `saved/compare` | Output directory for the box plot |
| `--trained-dir` | `saved` | Base directory with trained model artefacts |

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

# CRPS box plot across all test cases and methods
qmodem compare-box

# Box plot for specific methods and test cases
qmodem compare-box --methods het_cnn --methods bayes_cnn \
    --test-case 0 --test-case 1
```

## PHMe26 — Reproducing Conference Results

The results presented at the PHMe26 conference can be reproduced with the script(s) available in the `phme26/` folder.

```
phme26/
├── make_results.sh   # single entry-point script
├── data/             # generated train/val/test data
├── trained/          # model checkpoints
└── results/          # comparison plots
```

### Quick start

```bash
pip install uv                          # install uv (if not already installed)
uv sync                                 # install qmodem
bash phme26/make_results.sh --gen       # generate data + train + compare
```

After the first run the data is cached in `phme26/data/`, so subsequent runs
can skip data generation:

```bash
bash phme26/make_results.sh             # train + compare only
```

The script trains all four methods (`het_cnn`, `mcd_cnn`, `bayes_cnn`,
`qavi_cnn`) with default settings and produces a comparison plot for each of
the 4 test cases in `phme26/results/`.

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for full guidelines. Quick reference:

```bash
uv run pre-commit install            # install hooks (once, before contributing)
uv run pytest                        # run all tests
```
