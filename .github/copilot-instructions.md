# Copilot Instructions for QMoDeM

QMoDeM (Quantum-aided Models for Decision-Making) is a JAX/Flax research library for **Remaining Useful Life (RUL) prediction** of batteries using Bayesian, heteroscedastic, Monte Carlo Dropout, and quantum-inspired deep learning models.

## Commands

```bash
uv run pytest                        # run all tests
uv run pytest tests/test_module.py::TestMLPBlockV0::test_forward_pass_shape -v  # single test
uv run ruff check --fix              # lint
uv run ruff format                   # format
uv run pre-commit install            # install hooks (once, before contributing)
```

## Architecture

```
src/qmodem/
├── module.py    # All NN architectures (SimpleCNN1D, BayesCNN1D, MCDCNN1D, HeteroscedasticCNN1D, QAVICNN1D, …) and loss functions
├── data.py      # BatterySimulationSource / BatterySimulationTimeWindowSource — wraps lib_eod_simulation
├── train.py     # train_loop(), EarlyStopper
├── metrics.py   # CRPS (Continuous Ranked Probability Score) for probabilistic evaluation
└── __init__.py  # Re-exports all public symbols

scripts/         # Standalone experiment scripts (bayes_cnn/, het_cnn/, mcd_cnn/, qavi_cnn/)
tests/           # pytest tests mirroring src/qmodem modules
battery_config.json  # Simulator config (referenced via BATT_CONFIG_PATH)
```

**Data flow:** `battery_config.json` → `lib_eod_simulation` → `BatterySimulationSource` → sliding windows of voltage time-series → CNN models → RUL prediction + uncertainty → CRPS evaluation.

## Key Conventions

### Flax nnx patterns
- All models inherit from `nnx.Module` and require `*, rngs: nnx.Rngs` as a keyword-only argument.
- `nnx.Rngs` must be created **outside** `@nnx.jit` and passed in; never mutate RNG state inside jit.
- When saving checkpoints for models with `nnx.Dropout`, split only `nnx.Param` state — Orbax cannot serialize JAX PRNGKey dtype:
  ```python
  graphdef, params, other = nnx.split(model, nnx.Param, ...)
  ```

### Model naming
- Architecture classes: `*CNN1D` (e.g. `BayesCNN1D`, `MCDCNN1D`).
- Versioned variants use `V0`, `V1` suffixes (e.g. `MCDNetV0`, `MCDNetV1`).

### Loss functions
Defined alongside models in `module.py` and exported from `__init__.py`: `mse_loss`, `nll_loss`, `nll_loss_mcd`, `elbo_nll_loss`.

### Input tensor convention
All models expect input shape `(batch, 1, window_size)` and internally transpose to `(batch, window_size, 1)` for Flax `Conv`.

### Type annotations
Use `from __future__ import annotations` at the top of every file. Full type hints are required throughout.

### Docstrings
Google-style format, enforced by `docformatter` (configured in `pyproject.toml`).
