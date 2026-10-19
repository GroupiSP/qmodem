# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-10-19

qmodem now runs on [stagereg](https://pypi.org/project/stagereg/) `0.2.0` and
follows the [comparison protocol](docs/comparison-protocol.md). This is a clean
break: runs, checkpoints and `.last_trained.json` files from `0.1.0` can no
longer be loaded or selected ([#103], [#118]).

### Added

- Dependency on `stagereg[jax]==0.2.0`. Every run records a `stagereg_version` tag.
- Backbone run (`role=backbone`, one per seed): a `MeanStage` followed by a
  `GammaVarianceStage`, logging finalized `mean` and `variance` checkpoints and
  the scalers.
- Method runs (`role=method`), MLflow children of a backbone run, that train only
  the posterior stage on the conv layer (kernel and bias): `hnn`, `mcd`, `bbb`,
  `psgld`, `q-avi` and `c-avi`, built through one method registry.
- Shared posterior problem: `IsotropicGaussianPrior(1.0)`, validation on sample
  CRPS over 100 draws with a fixed draw key, early stopping (cap 300, patience 60)
  for every method except pSGLD.
- pSGLD method run: 300 epochs, at least 100 burn-in epochs, thinning that keeps
  exactly M = 100 samples, and time-per-epoch logging.
- Dataset-agnostic AVI inference method (`qmodem.avi`) with a `LatentMap`
  protocol and two latent maps: a quantum one (Q-AVI) and a parameter-matched
  classical MLP (C-AVI, a new ablation arm).
- Cost metrics per method run: posterior-stage wall-clock time, trainable
  parameter count and draws per second.
- Evaluation driver logging per-test-case metrics as
  `evaluation/test/<case>/<metric>` and unweighted means as
  `evaluation/test/<metric>`, using stagereg's metric functions.
- `MLflowTracker` and `MLflowCheckpointStore` (Orbax arrays, only finalized
  checkpoints uploaded) in the new `qmodem.tracking` package.
- Run selection by `--run-id` or by tags (`role`, `method`, `seed`); it raises on
  no match or an ambiguous match and never returns HPO runs.
- HPO following protocol §5: mean, variance and per-method posterior studies
  (`role=hpo_study`, trials `role=hpo_trial`, selected params logged as `best.*`).
- Scripts in `scripts/battery_multiple_scenarios/`: `train_backbone`,
  `train_method_run`, `evaluate`, `run_mean_study`, `run_variance_study` and
  `run_posterior_study`.
- `protocol_overrides=true` tag on runs started with `--posterior-param`,
  `--max-epochs` or `--patience`; tag selection skips them unless
  `--include-overridden` is passed.

### Changed

- Method tag `qavi` is renamed to `q-avi`.
- README and glossary describe the new run layout, method names and scripts.
- Mean and variance networks are separate (`Conv1D → act → GAP → Linear` and the
  same backbone with stagereg's `GammaHead`), replacing the joint Gaussian head.
- MLflow run layout: `role`, `method` and `seed` tags, params grouped by stage
  (e.g. `mean/optimizer/learning_rate`), checkpoints under
  `checkpoints/<stage>/best`, and evaluation logged into the method run.
- `jax` is pinned to `>=0.10,<0.11` (PennyLane 0.45 breaks on newer jax) and
  `flax` to `>=0.12.8,<0.12.9` (stagereg 0.2.0 stage restore fails on older flax).

### Removed

- CMAPSS support (data loading, LSTM, scripts). Use `v0.1.0` to recover it.
- Single-scenario `scripts/battery` (to be rebuilt in [#98]).
- Home-grown training loops, callbacks, metrics, Gaussian head and Bayesian conv
  layers (including Flipout), replaced by stagereg.
- `.last_trained.json`, `MLFLOW_USE_LAST_TRAINED` and the old `run_train*`,
  `run_inference*` and `run_hpo*` scripts.
- `jax-ai-stack` dependency.
- Old `*_average` evaluation metric keys.

### Notes

- pSGLD is affordable at CNN size: about 2.6 s per burn-in epoch and 6 s per
  sampling epoch on CPU, roughly 25 minutes per run. No stagereg `0.2.1` is
  needed.

## [0.1.0] - 2026-10-15

Last pre-stagereg release, including CMAPSS. Tagged without a GitHub release.

[Unreleased]: https://github.com/GroupiSP/qmodem/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/GroupiSP/qmodem/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/GroupiSP/qmodem/releases/tag/v0.1.0
[#98]: https://github.com/GroupiSP/qmodem/issues/98
[#103]: https://github.com/GroupiSP/qmodem/issues/103
[#118]: https://github.com/GroupiSP/qmodem/pull/118
