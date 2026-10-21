# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Disjoint backbone and posterior validation sets, with fixed per-split policy
  counts and one random stream per split ([#121]).
- Backbones trained before the cosine learning-rate decay (constant-LR optimizer
  state) no longer restore and must be retrained ([#121]).
- The pickled test simulation config keeps the 70/30 Bernoulli policy prior only
  for the evaluation reference band; the histories use fixed policy counts
  ([#121]).
- Dependency bumped to `stagereg[jax]==0.4.0`. The posterior stage now validates
  under stagereg's `validation_key`, one per method run (ADR 0002), on the
  compiled path; qmodem's fixed-key wrappers are gone ([#135]).
- Posterior validation metrics, early-stopping decisions and posterior training
  randomness differ from older runs: validation keys each batch's predictive
  samples separately and no longer advances the run's key. Posterior HPO studies
  and method runs must be rerun; backbone runs are unaffected ([#135]).

## [0.2.1] - 2026-10-20

qmodem now runs on stagereg `0.3.0` ([#130]).

### Changed

- Dependency bumped to `stagereg[jax]==0.3.0` (exact pin).
- The `jax>=0.10,<0.11` and `flax>=0.12.8,<0.12.9` caps are kept, now driven by
  PennyLane: PennyLane 0.45.1 calls `jax.core.is_concrete`, removed in JAX 0.11
  ([PennyLaneAI/pennylane#9930]), and Flax 0.12.9 requires JAX 0.11. The caps will
  be lifted once PennyLane supports JAX 0.11.

### Notes

- Results are statistically equivalent to `0.2.0` runs. Bit-identity is not
  guaranteed, though the smoke-run q-avi, c-avi and pSGLD loss trajectories were
  bit-identical on the same JAX/Flax ([#132]). The `stagereg_version` tag
  (`0.2.0` vs `0.3.0`) tells the runs apart.
- pSGLD is about 30% faster (partial speed-up). Same machine and data (CPU, smoke
  dataset, 300 epochs, epoch 0 excluded), `0.2.0` vs `0.3.0`: burn-in 0.046 vs
  0.032 s/epoch, sampling 1.17 vs 0.83 s/epoch, wall 241 vs 171 s ([#132]).
  Sampling epochs still grow with the number of retained samples
  ([GroupiSP/stagereg#136]). Absolute times are on the small smoke dataset, so
  they are not comparable to the earlier ≈2.6 s / 6 s per-epoch figures; the
  ~30% is the same-machine relative gain.

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

[Unreleased]: https://github.com/GroupiSP/qmodem/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/GroupiSP/qmodem/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/GroupiSP/qmodem/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/GroupiSP/qmodem/releases/tag/v0.1.0
[#98]: https://github.com/GroupiSP/qmodem/issues/98
[#103]: https://github.com/GroupiSP/qmodem/issues/103
[#118]: https://github.com/GroupiSP/qmodem/pull/118
[#121]: https://github.com/GroupiSP/qmodem/issues/121
[#130]: https://github.com/GroupiSP/qmodem/issues/130
[#132]: https://github.com/GroupiSP/qmodem/issues/132
[#135]: https://github.com/GroupiSP/qmodem/issues/135
[PennyLaneAI/pennylane#9930]: https://github.com/PennyLaneAI/pennylane/issues/9930
[GroupiSP/stagereg#136]: https://github.com/GroupiSP/stagereg/issues/136
