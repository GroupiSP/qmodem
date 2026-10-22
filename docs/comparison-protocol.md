# Comparison protocol

The comparison protocol fixes the stages, data, seeds, tuning, metrics and outputs under which HNN, MCD, BBB, pSGLD and Q-AVI are compared on the battery multiple-scenarios case, so that **the methods differ only in the posterior stage**. It was decided in [Comparison protocol (#96)](https://github.com/GroupiSP/qmodem/issues/96) on the map [Move qmodem onto stagereg and isolate QAVI's epistemic UQ (#89)](https://github.com/GroupiSP/qmodem/issues/89). Vocabulary: [qmodem GLOSSARY](../GLOSSARY.md), [stagereg GLOSSARY](https://github.com/GroupiSP/stagereg/blob/main/GLOSSARY.md).

```
comparison protocol
├─ methods     core: HNN · MCD · BBB · pSGLD (conditional) · Q-AVI
│              arms: ablation (C-AVI) · sensitivity (N_eff)
├─ backbone    one backbone run per protocol seed, shared; conv layer inferred
├─ data        one data-gen run; train 10 · backbone val 20 · posterior val 20 · test 20
│              fixed 70/30 policy counts per split
├─ seeds       R = 5, dataset fixed, keys derived from r, paired by seed
├─ posterior   N = #windows; M = 100 draws × 1 aleatoric noise sample
├─ tuning      50 TPE trials per posterior stage on seed 0, HPs frozen; fixed prior
│              one shared validation criterion: validation CRPS, M_val = 100,
│              on the posterior validation set; mean/variance on the backbone validation set
├─ metrics     expected CRPS (grid-free), RMSE, coverage@0.95, WSU per trajectory
│              sample-based; t0 dropped; mean over test cases (+ per policy)
├─ validity    a failed check invalidates the seed for all methods
└─ outputs     T1 · T2 · T3 · F1 · F2 · F3 · F4
```

## 1. Question and methods

**Headline question.** Does Q-AVI's implicit, adversarially fitted posterior give better-calibrated epistemic UQ than explicit-family VI (BBB), sampling (pSGLD) and MCD, on the same backbone? HNN is the reference with no epistemic UQ.

| Method | Posterior stage | Status |
|---|---|---|
| HNN | none (method run is evaluation-only) | core |
| MCD | mean network + elementwise input dropout before the conv | core |
| BBB | stagereg `BayesByBackprop` on the conv layer | core |
| pSGLD | stagereg pSGLD on the conv layer | core, **conditional**: dropped with a note if unaffordable at CNN size |
| Q-AVI | generator (quantum latent map + postprocessor) on the conv layer, objective as in [#94](https://github.com/GroupiSP/qmodem/issues/94) | core |
| C-AVI | as Q-AVI, quantum latent map replaced by a parameter-matched classical latent map ([#100](https://github.com/GroupiSP/qmodem/issues/100)) | **ablation arm** |
| any core method under N_eff | as the core method, likelihood scaled by N_eff (§4) | **sensitivity arm**, optional |

Arms are reported against the core methods (one T2-style row each) and never enter T1.

## 2. Shared stages

Fixed by [#92](https://github.com/GroupiSP/qmodem/issues/92) and [#93](https://github.com/GroupiSP/qmodem/issues/93):

- For each protocol seed, one **backbone run** trains the mean stage and the iteration-0 variance stage. Every method run of that seed restores the same finalized `mean`/`variance` checkpoints.
- The inferred subnetwork is the **whole conv layer** (kernel + bias) for every method with a posterior stage.
- The mean and variance network sizes come from the mean-stage HPO ([#99](https://github.com/GroupiSP/qmodem/issues/99)) and are shared, so they cannot bias the comparison.

## 3. Seeds and repetitions

- **R = 5 protocol seeds**, r = 0…4. Increase to 10 if the across-seed std of a primary metric is comparable to the method differences.
- The dataset is **fixed**: one data-generation run (`DATA_GEN_RUN_ID_MULTI`), not resampled per seed. It is the run generated with the four splits below, not data-gen run `e52603fb…` behind the `stages` experiment ([#121](https://github.com/GroupiSP/qmodem/issues/121)).
- The data-generation run has **four splits**. Policies are assigned by **fixed counts** per split (70/30, in a seeded random order), never by i.i.d. draws, so every split has policy-1 histories:

  | Split | Histories | Policy 0 / policy 1 | Used by |
  |---|---|---|---|
  | train | 10 | 7 / 3 | every stage's training loss |
  | backbone validation set | 20 | 14 / 6 | mean and variance stages: early stopping, best checkpoint, HPO |
  | posterior validation set | 20 | 14 / 6 | the validation criterion (§5) |
  | test | 20 | 14 / 6 | evaluation only (§6) |

  Train, backbone validation set and posterior validation set are contiguous `run_id` blocks, in that order. Their sizes and policy counts are logged as params of the data-generation run, and data loading slices them by `run_id` range. The two validation sets are **disjoint**, so no posterior stage is judged on histories its backbone was selected on ([ADR 0001](adr/0001-disjoint-validation-sets.md)).
- Every RNG key derives from r. The backbone uses r directly. Method runs use `fold_in(r, method)` for posterior init, draws and evaluation.
- Methods are compared **paired by seed**, because they share the backbone run within a seed.

## 4. Posterior problem

- **Likelihood scaling**: N = number of training windows (stride 1), the i.i.d. convention. Overlapping windows from the same history are correlated, so this over-concentrates every posterior equally. It is stated as a **limitation**.
- **Sensitivity arm (optional)**: N_eff = N / (1 + 2 Σ_k ρ_k), with ρ_k the autocorrelation of the mean stage's training residuals within histories. It is reported against the primary N, never in place of it.

## 5. Tuning

- **Equal budget**: each method's posterior stage gets the same Optuna budget (number of trials, TPE sampler, sampler seed, objective) on the backbone run of **seed 0**. The selected HPs are **frozen** across all R seeds. MCD's dropout rate and Q-AVI's λ, S, k and learning rates are all tuned within that budget. Q-AVI has more HPs, and the equal budget is a deliberate handicap. C-AVI gets its own equal budget over Q-AVI's search space, latent-map width h derived from (q, L) ([#100](https://github.com/GroupiSP/qmodem/issues/100)).
- **One validation criterion** is shared by every method with a posterior stage: the **mean sample CRPS** (stagereg `crps`, energy form) of the posterior predictive against the observed validation target ([#101](https://github.com/GroupiSP/qmodem/issues/101)), computed on the **posterior validation set** (§3). Validation targets are single observations of held-out histories, so the window mean estimates the same expected CRPS as the test metric. Each posterior epoch scores CRPS only. `nll`, `rmse` and `coverage_95` are diagnostics: they are scored once, on the best checkpoint, from the same predictive samples (same validation key, `PREDICTIVE_SAMPLE_COUNT` draws), logged at the best epoch's step, and never select anything ([#139](https://github.com/GroupiSP/qmodem/issues/139)).
- **Validation draws**: M_val = M = 100 draws × 1 aleatoric noise sample (`num_draws = predictive_sample_count = 100`), with a **fixed draw key** per run. A smaller M_val is ruled out: the plug-in CRPS is biased by E|X − X'| / (2M), which favours narrow predictives. M_val is independent of Q-AVI's training-loss S.
- **Role of the criterion per method**:

  | Method | Best checkpoint | Early stopping | HPO objective |
  |---|---|---|---|
  | MCD, BBB, Q-AVI, C-AVI | ✅ | ✅ | best-epoch validation CRPS |
  | pSGLD | — (last epoch: a stopper would truncate the chain) | — | validation CRPS of the final retained chain |

- **Posterior baseline**: before posterior epoch 0, every method run with a posterior stage computes the validation criterion once (same draw key and M_val) and logs it as `val_crps_baseline`. It is diagnostic only: it never feeds early stopping, the best checkpoint or the HPO objective. HNN logs none.
- **Validation set per stage**: the mean and variance stages (early stopping, best checkpoint, mean and variance studies) use the **backbone validation set**. Every posterior stage, posterior study and the draws-per-second probe use the **posterior validation set**. The two selection steps never share a history ([ADR 0001](adr/0001-disjoint-validation-sets.md)).
- **Patience**: one shared patience for every method that early-stops, generous (≈ 20% of the max-epoch cap) so that it covers Q-AVI's adversarial oscillations, with `min_delta = 0`. It is fixed and never tuned. Method-specific patience is ruled out as unequal treatment.
- **Prior**: one fixed prior for every method, stagereg's `IsotropicGaussianPrior(precision=1.0)`. It is part of the shared posterior problem and is never tuned.
- **Budget**: **50 trials** per posterior study, `TPESampler` with a fixed sampler seed, and a `MedianPruner` with `n_startup_trials = 5` and `n_warmup_steps = 60`. Every method reports per-epoch validation CRPS. pSGLD reports the CRPS of its running chain after burn-in. HNN has no study.
- **Caps**: MCD, BBB, Q-AVI and C-AVI train for at most **300 epochs**, with a shared **patience of 60** and `min_delta = 0`. pSGLD runs a fixed **300 epochs**. The first 100 epochs are burn-in, and `thinning` is derived so that exactly M = 100 samples are retained.
- **Shared, untuned settings**: batch size 32. The posterior stage uses Adam at a constant learning rate, with no schedule. The mean and variance stages use Adam with **cosine LR decay** from `lr` to `0.01·lr` over the stage's `max_epochs`, stepped per epoch; the floor ratio 0.01 is fixed and never tuned ([#121](https://github.com/GroupiSP/qmodem/issues/121)).
- **Search spaces** (ranges on a log scale are marked *log*):

  | Method | Tuned HPs |
  |---|---|
  | MCD | dropout rate 0.01–0.5; LR 1e-5–1e-2 *log* |
  | BBB | LR 1e-5–1e-2 *log*; `initial_std` 1e-4–1e-1 *log* |
  | pSGLD | `step_size` 1e-6–1e-2 *log*; `decay` 0.99 and `stability` 1e-5 fixed |
  | Q-AVI | λ 1e-4–1 *log*; S ∈ {1, 4, 8, 16}; k ∈ {1, 2, 3, 5}; q 3–6; L 1–4; generator LR and discriminator LR, each 1e-5–1e-2 *log*; discriminator fixed (64-64 LeakyReLU MLP) |
  | C-AVI | as Q-AVI; h derived from (q, L) |

- **Variance stage**: a variance-only study (20 trials, TPE, median pruning) runs on seed 0's frozen mean checkpoint after the mean study ([#99](https://github.com/GroupiSP/qmodem/issues/99)). It tunes only the variance LR (1e-4–1e-2, *log*) and minimizes best-epoch validation Gamma NLL on the backbone validation set, the stage's own loss. Cap, patience and batch size are inherited from the mean study.
- **Study runs**: there is one top-level `role=hpo_study` run per (stage, method), tagged `stage`, `method`, `backbone_run_id` (the seed-0 backbone) and `n_trials`. It logs the selected params as `best.*`. Its trials are `role=hpo_trial` children. They restore that backbone and all use the same key, `fold_in(0, method)`. Method runs, seed 0 included, are always **re-trained** from the frozen HPs and never reuse a trial.

Decided in [How are the variance and posterior stages tuned under an equal budget? (#102)](https://github.com/GroupiSP/qmodem/issues/102). Validation sets per stage, posterior baseline and mean/variance LR decay decided in [Spec: disjoint, policy-stratified validation sets + backbone LR decay (#121)](https://github.com/GroupiSP/qmodem/issues/121).

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

All means and intervals are **sample-based** for every method. Q-AVI has no analytic form, and analytic intervals for the other methods would be an unequal treatment.

**Aggregation.** Each metric is computed per test case, then averaged with an **unweighted mean over test cases** (primary). Per current policy (policy 0 / policy 1 cases), the same mean is reported as secondary columns. Pooled values are not reported.

## 7. Validity

A seed enters the results only if all of these hold:

| Check | Rule |
|---|---|
| Shared backbone | every method run of seed r has the same `mlflow.parentRunId`; the backbone's `data_generation_run_id` tag equals the protocol's |
| Completeness | every run is `FINISHED`; every core method has a run for the seed (no ragged seeds) |
| Frozen HPs | method runs log the HPO-selected params, identical across seeds |
| Versions | one `stagereg_version` and one qmodem commit across all runs |
| Posterior health | Q-AVI: draws not collapsed (spread of conv weights across draws > 0; discriminator accuracy not stuck at 1.0). pSGLD: burn-in respected, ≥ M retained samples. Thresholds are an implementation detail; the values are reported |

A failed check **invalidates the seed for all methods**, and the seed is re-run. Methods are never dropped individually.

## 8. Reporting

**Statistics.** Per metric: the per-seed mean over test cases, then **mean ± std across seeds** per method. **Paired differences** vs BBB and vs HNN, per seed, with a 95% bootstrap CI over seeds. No p-values.

**Outputs.** T1, T2, F1 and F2 are the primary results.

| ID | Content | Answers |
|---|---|---|
| **T1** | RMSE, CRPS, coverage@0.95, WSU per method (mean ± std), plus per-policy columns | overall ranking |
| **T2** | paired Δ vs BBB and vs HNN (bootstrap CI), plus per-policy columns; one row per arm | is Q-AVI's gain real |
| **T3** | posterior-stage wall-clock, trainable params, draws/s per method (timed on the first 1024 posterior-validation windows, `DRAWS_PER_SECOND_WINDOWS`) | cost of each method |
| **F1** | reliability curve: empirical vs nominal coverage at levels 0.5…0.95 | calibration beyond one level |
| **F2** | epistemic / aleatoric variance decomposition over SoC0, per method, one representative case **per current policy**, event marked | does Q-AVI add epistemic spread, and where |
| **F3** | per-test-case CRPS bars, all methods, grouped by policy | is the gain uniform |
| **F4** | RUL predictive 95% bands vs the simulator reference band, all methods, one representative case **per current policy**, event marked | qualitative picture, incl. the bimodal pre-event reference |

**Representative case.** For each current policy, the test case whose HNN CRPS is the **median** among that policy's cases at protocol seed 0. The rule is fixed before any method results are seen.

## Limitations

- N counts overlapping windows (§4).
- R = 5 seeds on a single fixed dataset: the CIs cover training randomness, not dataset resampling.
- Iteration-0 variance only. Outer variance/posterior iterations would make variance checkpoints method-specific, and the protocol would then share the iteration schedule instead.
