# Posterior validation key derived per method run

**Status**: accepted

Every posterior stage is validated under one fixed key, `PosteriorStageOptions.validation_key` (stagereg ≥ 0.4.0), so every epoch scores the same draws and predictive samples. qmodem does **not** use stagereg's constant default (`jax.random.key(0)`). It derives the key from the method run's key (`PosteriorKeys.validation = fold_in(method_key, _VALIDATION_STREAM)`), so the key differs by protocol seed and by method ([#135](https://github.com/GroupiSP/qmodem/issues/135)).

## Considered options

| Option | Verdict |
|---|---|
| stagereg's constant default key | Rejected: every protocol seed would score the same validation draws and samples, so validation noise would be shared across seeds, while methods are compared paired by protocol seed as if the seeds were independent. |
| **Key derived per method run** | Chosen: validation noise is independent across protocol seeds. Trials of one HPO study still share the key, because they share protocol seed 0 and the method, so the pruner and trial ranking compare trials on the same draws. |

## Consequences

- `val_crps_baseline` must reproduce stagereg's key derivation exactly (`split` into draw and metric keys, then `fold_in(metric_key, batch_index)`), or the baseline and the epochs stop being comparable. A test pins this: epoch-0 validation CRPS with no update equals the baseline.
- Switching to the constant default, or changing the derivation, changes every posterior validation score, early-stopping decision and HPO result. Such a change must supersede this ADR, and posterior studies must be rerun.
