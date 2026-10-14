# Comparison protocol

The comparison protocol fixes the stages, data, seeds, tuning, metrics and outputs under which HNN, MCD, BBB, pSGLD and QAVI are compared on the battery multiple-scenarios case, so that **the methods differ only in the posterior stage**. It was decided in [Comparison protocol (#96)](https://github.com/GroupiSP/qmodem/issues/96) on the map [Move qmodem onto stagereg and isolate QAVI's epistemic UQ (#89)](https://github.com/GroupiSP/qmodem/issues/89). Vocabulary: [qmodem GLOSSARY](../GLOSSARY.md), [stagereg GLOSSARY](https://github.com/GroupiSP/stagereg/blob/main/GLOSSARY.md).

```
comparison protocol
├─ methods     core: HNN · MCD · BBB · pSGLD (conditional) · QAVI
│              arms: ablation (classical generator) · sensitivity (N_eff)
├─ backbone    one backbone run per protocol seed, shared; conv layer inferred
├─ seeds       R = 5, dataset fixed, keys derived from r, paired by seed
├─ posterior   N = #windows; M = 100 draws × 1 aleatoric noise sample
├─ tuning      equal Optuna budget per posterior stage on seed 0, HPs frozen
│              one shared validation criterion for all posterior stages
├─ metrics     expected CRPS (grid-free), RMSE, coverage@0.95, WSU per trajectory
│              sample-based; t0 dropped; mean over test cases (+ per policy)
├─ validity    a failed check invalidates the seed for all methods
└─ outputs     T1 · T2 · T3 · F1 · F2 · F3 · F4
```

## 1. Question and methods

**Headline question.** Does QAVI's implicit, adversarially fitted posterior give better-calibrated epistemic UQ than explicit-family VI (BBB), sampling (pSGLD) and MCD, on the same backbone? HNN is the reference with no epistemic UQ.

| Method | Posterior stage | Status |
|---|---|---|
| HNN | none (method run is evaluation-only) | core |
| MCD | mean network + elementwise input dropout before the conv | core |
| BBB | stagereg `BayesByBackprop` on the conv layer | core |
| pSGLD | stagereg pSGLD on the conv layer | core, **conditional**: dropped with a note if unaffordable at CNN size |
| QAVI | generator (PQC + postprocessor) on the conv layer, objective as in [#94](https://github.com/GroupiSP/qmodem/issues/94) | core |
| classical-generator QAVI | as QAVI, PQC replaced by a classical generator | **ablation arm**, only if [#100](https://github.com/GroupiSP/qmodem/issues/100) includes it |
| any core method under N_eff | as the core method, likelihood scaled by N_eff (§4) | **sensitivity arm**, optional |

Arms are reported against the core methods (one T2-style row each) and never enter T1.

## 2. Shared stages

Fixed by [#92](https://github.com/GroupiSP/qmodem/issues/92) and [#93](https://github.com/GroupiSP/qmodem/issues/93):

- For each protocol seed, one **backbone run** trains the mean stage and the iteration-0 variance stage. Every method run of that seed restores the same finalized `mean`/`variance` checkpoints.
- The inferred subnetwork is the **whole conv layer** (kernel + bias) for every method with a posterior stage.
- The mean and variance network sizes come from the mean-stage HPO ([#99](https://github.com/GroupiSP/qmodem/issues/99)) and are shared, so they cannot bias the comparison.

## 3. Seeds and repetitions

- **R = 5 protocol seeds**, r = 0…4. Increase to 10 if the across-seed std of a primary metric is comparable to the method differences.
- The dataset is **fixed**: one data-generation run (`DATA_GEN_RUN_ID_MULTI`), not resampled per seed.
- Every RNG key derives from r. The backbone uses r directly. Method runs use `fold_in(r, method)` for posterior init, draws and evaluation.
- Methods are compared **paired by seed**, because they share the backbone run within a seed.

## 4. Posterior problem

- **Likelihood scaling**: N = number of training windows (stride 1), the i.i.d. convention. Overlapping windows from the same history are correlated, so this over-concentrates every posterior equally. It is stated as a **limitation**.
- **Sensitivity arm (optional)**: N_eff = N / (1 + 2 Σ_k ρ_k), with ρ_k the autocorrelation of the mean stage's training residuals within histories. It is reported against the primary N, never in place of it.

## 5. Tuning

- **Equal budget**: each method's posterior stage gets the same Optuna budget (number of trials, TPE sampler, sampler seed, objective) on the backbone run of **seed 0**. The selected HPs are **frozen** across all R seeds. MCD's dropout rate and QAVI's λ, S, k and learning rates are all tuned within that budget. QAVI has more HPs, and the equal budget is a deliberate handicap.
- **One validation criterion** is shared by every method with a posterior stage, for best-checkpoint selection, early stopping and the HPO objective. Which criterion is decided by [#101](https://github.com/GroupiSP/qmodem/issues/101).
- The budget size and search spaces are decided per stage in the per-stage HPO ticket.

## 6. Evaluation

**Draws.** M = 100 posterior draws per window for every method, each with **one** aleatoric noise sample, giving 100 predictive samples. HNN gets 100 aleatoric samples from its single mean. M is never tuned per method.

**Test cases and timestamps.** All test cases of the fixed data-generation run, `test_n_soc0s = 20`, with 100 simulator reference samples per timestamp. The **t0** timestamp has no predictions. It is dropped from every metric and serves only as WSU's time origin.

**Metrics** (stagereg `core.metrics` functions; qmodem's per-test-case driver):

| Metric | Definition |
|---|---|
| CRPS | **expected CRPS**, grid-free: `mean_j sample_crps(pred_samples, y_ref_j)` over the simulator reference samples. This equals E_{Y∼true}[CRPS(F_pred, Y)]. Cramér-distance values are never reported or mixed in |
| RMSE | against the sample mean of the predictive samples |
| Coverage | empirical 2.5/97.5 percentiles of the predictive samples, inclusive; level **0.95** is an explicit protocol parameter |
| WSU | `core.metrics.wsu` **per trajectory**, t0 passed as origin. The epoch `WeightedSpread` adapter is never used |

All means and intervals are **sample-based** for every method. QAVI has no analytic form, and analytic intervals for the other methods would be an unequal treatment.

**Aggregation.** Each metric is computed per test case, then averaged with an **unweighted mean over test cases** (primary). Per current policy (policy 0 / policy 1 cases), the same mean is reported as secondary columns. Pooled values are not reported.

## 7. Validity

A seed enters the results only if all of these hold:

| Check | Rule |
|---|---|
| Shared backbone | every method run of seed r has the same `mlflow.parentRunId`; the backbone's `data_generation_run_id` tag equals the protocol's |
| Completeness | every run is `FINISHED`; every core method has a run for the seed (no ragged seeds) |
| Frozen HPs | method runs log the HPO-selected params, identical across seeds |
| Versions | one `stagereg_version` and one qmodem commit across all runs |
| Posterior health | QAVI: draws not collapsed (spread of conv weights across draws > 0; discriminator accuracy not stuck at 1.0). pSGLD: burn-in respected, ≥ M retained samples. Thresholds are an implementation detail; the values are reported |

A failed check **invalidates the seed for all methods**, and the seed is re-run. Methods are never dropped individually.

## 8. Reporting

**Statistics.** Per metric: the per-seed mean over test cases, then **mean ± std across seeds** per method. **Paired differences** vs BBB and vs HNN, per seed, with a 95% bootstrap CI over seeds. No p-values.

**Outputs.** T1, T2, F1 and F2 are the primary results.

| ID | Content | Answers |
|---|---|---|
| **T1** | RMSE, CRPS, coverage@0.95, WSU per method (mean ± std), plus per-policy columns | overall ranking |
| **T2** | paired Δ vs BBB and vs HNN (bootstrap CI), plus per-policy columns; one row per arm | is QAVI's gain real |
| **T3** | posterior-stage wall-clock, trainable params, draws/s per method | cost of each method |
| **F1** | reliability curve: empirical vs nominal coverage at levels 0.5…0.95 | calibration beyond one level |
| **F2** | epistemic / aleatoric variance decomposition over SoC0, per method, one representative case **per current policy**, event marked | does QAVI add epistemic spread, and where |
| **F3** | per-test-case CRPS bars, all methods, grouped by policy | is the gain uniform |
| **F4** | RUL predictive 95% bands vs the simulator reference band, all methods, one representative case **per current policy**, event marked | qualitative picture, incl. the bimodal pre-event reference |

**Representative case.** For each current policy, the test case whose HNN CRPS is the **median** among that policy's cases at protocol seed 0. The rule is fixed before any method results are seen.

## Limitations

- N counts overlapping windows (§4).
- R = 5 seeds on a single fixed dataset: the CIs cover training randomness, not dataset resampling.
- Iteration-0 variance only. Outer variance/posterior iterations would make variance checkpoints method-specific, and the protocol would then share the iteration schedule instead.
