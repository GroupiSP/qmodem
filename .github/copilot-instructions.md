# QMoDeM: Quantum-aided Models for Decision-Making

This project builds neural network models for battery health prediction using JAX, Flax NNX, and battery discharge simulation data from `lib-eod-simulation`.

## Development Tools

### Package Manager
This project uses **`uv`** for all dependency management. Always use `uv run` to execute commands in the project environment.

### Installation
```bash
# Install dependencies
uv sync --locked --all-extras --dev

# Install pre-commit hooks (required before contributing)
uv run pre-commit install
```

## Build, Test, and Lint Commands

### Testing
```bash
# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/test_model.py

# Run a specific test function
uv run pytest tests/test_model.py::test_hnn_v0_init

# Run tests with verbose output
uv run pytest -v
```

### Linting and Formatting
```bash
# Run all pre-commit hooks (ruff linting, ruff formatting, docformatter)
uv run pre-commit run --all-files

# Run ruff linter only
uv run ruff check

# Run ruff formatter only
uv run ruff format

# Run docformatter only
uv run docformatter --in-place --config ./pyproject.toml
```

### CI/CD
- **Linting**: `.github/workflows/lint.yml` runs pre-commit hooks on all files
- **Testing**: `.github/workflows/test.yml` runs pytest

## Architecture

### Core Components

1. **Data Pipeline** (`src/qmodem/data.py`)
   - `BatterySimulationSource`: Generates training data from battery discharge simulations
   - Uses `lib_eod_simulation` library for physics-based battery simulations
   - Flattens multi-simulation histories into time-step records with voltage features and RUL targets
   - Configuration file: `battery_config.json` (referenced via `BATT_CONFIG_PATH`)

2. **Neural Network Modules** (`src/qmodem/module.py`)
   - **Building blocks**: `GaussianBlock`, `ResNetBlockV0/V1`, `MLPBlockV0`
   - **Model variants**:
     - `HNNV0/V1`: Heteroscedastic NNs (predict mean + variance) for uncertainty quantification
     - `MCDNetV0/V1`: Monte Carlo Dropout networks for epistemic uncertainty
     - `NNEnsemble`: Ensemble wrapper for multiple models
   - Uses **Flax NNX** (not legacy Flax Linen)

3. **Loss Functions** (`src/qmodem/module.py`)
   - `nll_loss`: Negative log-likelihood for Gaussian outputs
   - `mse_loss`: Standard MSE loss

4. **Training Utilities** (`src/qmodem/train.py`)
   - `EarlyStopper`: Patience-based early stopping with `min_delta` threshold

5. **Metrics** (`src/qmodem/metrics.py`)
   - `cdf`: Cumulative distribution function from samples
   - `crps`: Continuous Ranked Probability Score for distribution comparison

### Workflow
Scripts in `scripts/` demonstrate complete training/evaluation pipelines:
- Load battery config → Run simulation → Create data source
- Build model → Train with optax optimizer → Save checkpoints with Orbax
- Evaluate with uncertainty metrics (CRPS, ensemble predictions)

## Key Conventions

### JAX and Flax NNX
- **Use Flax NNX**, not Linen (this is the new API as of 2024+)
- All modules inherit from `nnx.Module`
- RNGs passed via `rngs: nnx.Rngs` parameter (not separate RNG arguments)
- Modules store state directly (unlike Linen's functional approach)

### Model Initialization Patterns
- **Identity initialization for ResNets**: `ResNetBlockV0/V1` use zero-initialized layer norm scales to approximate identity mapping at initialization
- **Gaussian output layers**: Final layers in heteroscedastic models output `[mu, softplus(var)]` concatenated
- **RNG keys**: Models require named RNG streams (e.g., `rngs=nnx.Rngs(params=0, dropout=1)`)

### Code Style
- **Type hints**: Use `from __future__ import annotations` and full type annotations
- **Docstring style**: Google-style docstrings (enforced by docformatter)
- **Formatting**: Black-compatible via ruff (configured in pyproject.toml)
- **Array types**: Use `jax.Array` for type hints (not `np.ndarray` or `jnp.ndarray`)

### Testing
- Use pytest fixtures for model initialization (see `tests/test_model.py`)
- Parametrize tests with different input shapes/configurations
- Test both model structure (e.g., layer counts) and forward pass outputs

### Data Organization
- Simulation results saved to `saved/<experiment_name>/`
- Checkpoints in `saved/<experiment_name>/checkpoints/`
- Metadata (e.g., max RUL for unscaling) in `saved/<experiment_name>/metadata/` as JSON
- Use `qmodem.utils.mkdir_if_not_existent()` to create directory structures

### Dependencies
- **Local dependency**: `lib-eod-simulation` is an editable local dependency (see `pyproject.toml` `[tool.uv.sources]`)
- Must be available at `../lib_eod_simulation` relative to this repo

## Important Files
- `battery_config.json`: Battery simulation parameters (SOC range, diffusion coefficients, etc.)
- `pyproject.toml`: Project config, dependencies, tool settings (ruff, docformatter, pytest)
- `uv.lock`: Locked dependencies (do not edit manually)
