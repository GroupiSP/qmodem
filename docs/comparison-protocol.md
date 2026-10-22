# Comparison protocol

Fixes the stages, data, seeds, tuning, metrics and outputs under which HNN, MCD, BBB, pSGLD and Q-AVI are compared on the battery multiple-scenarios case, so that **the methods differ only in the posterior stage**. Decided in [#96](https://github.com/GroupiSP/qmodem/issues/96) (map: [#89](https://github.com/GroupiSP/qmodem/issues/89)). Vocabulary: [qmodem GLOSSARY](../GLOSSARY.md), [stagereg GLOSSARY](https://github.com/GroupiSP/stagereg/blob/main/GLOSSARY.md).

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
| C-AVI | as Q-AVI, parameter-matched classical latent map ([#100](https://github.com/GroupiSP/qmodem/issues/100)) | **ablation arm** |
| any core method under N_eff | likelihood scaled by N_eff (§4) | **sensitivity arm**, optional |

Arms get one T2-style row each and never enter T1.

## 2. Shared stages ([#92](https://github.com/GroupiSP/qmodem/issues/92), [#93](https://github.com/GroupiSP/qmodem/issues/93))

- Per protocol seed, one **backbone run** trains the mean stage and the iteration-0 variance stage. Every method run of that seed restores its `mean`/`variance` checkpoints.
- Inferred subnetwork: the **whole conv layer** (kernel + bias), for every method with a posterior stage.
- Mean and variance network sizes come from the mean-stage HPO ([#99](https://github.com/GroupiSP/qmodem/issues/99)) and are shared.

## 3. Data and seeds

- **R = 5 protocol seeds**, r = 0…4. Raise to 10 if the across-seed std of a primary metric is comparable to the method differences.
- **Fixed dataset**: one data-generation run (`DATA_GEN_RUN_ID_MULTI`), the one with the four splits below (not `e52603fb…` behind the `stages` experiment, [#121](https://github.com/GroupiSP/qmodem/issues/121)).
- Policies are assigned by **fixed counts** per split (70/30, seeded random order), never i.i.d., so every split has policy-1 histories:

  | Split | Histories | Policy 0 / 1 | Used by |
  |---|---|---|---|
  | train | 10 | 7 / 3 | every stage's training loss |
  | backbone validation set | 20 | 14 / 6 | mean and variance stages: early stopping, best checkpoint, HPO |
  | posterior validation set | 20 | 14 / 6 | every posterior stage and study, the draws/s probe |
  | test | 20 | 14 / 6 | evaluation only (§6) |

  Train and the two validation sets are contiguous `run_id` blocks, in that order; sizes and policy counts are params of the data-generation run. The validation sets are **disjoint** ([ADR 0001](adr/0001-disjoint-validation-sets.md)).
- **Keys** derive from r: the backbone uses r, method runs `fold_in(r, method)` for posterior init, draws, validation ([ADR 0002](adr/0002-posterior-validation-key-per-method-run.md)) and evaluation.
- Methods are compared **paired by seed** (shared backbone run).

## 4. Posterior problem

- **Likelihood scaling**: N = number of stride-1 training windows (i.i.d. convention). Overlapping windows are correlated, so every posterior is over-concentrated equally: a **limitation**.
- **Sensitivity arm** (optional): N_eff = N / (1 + 2 Σ_k ρ_k), ρ_k the within-history autocorrelation of the mean stage's training residuals. Reported next to N, never in place of it.

## 5. Tuning ([#101](https://github.com/GroupiSP/qmodem/issues/101), [#102](https://github.com/GroupiSP/qmodem/issues/102), [#121](https://github.com/GroupiSP/qmodem/issues/121), [#139](https://github.com/GroupiSP/qmodem/issues/139))

Every method with a posterior stage gets the **same Optuna budget** on seed 0's backbone run; the selected HPs are **frozen** across all seeds. Q-AVI has the most HPs, so the equal budget handicaps it. HNN has no study.

| Setting | Value | Applies to |
|---|---|---|
| Study budget | 50 trials, `TPESampler` (fixed seed), `MedianPruner(n_startup_trials=5, n_warmup_steps=60)` | every posterior study |
| Validation criterion | mean sample CRPS (stagereg `crps`, energy form) on the posterior validation set; the only per-epoch score | every posterior stage |
| Validation draws | M_val = 100 draws × 1 noise sample, one validation key per method run ([ADR 0002](adr/0002-posterior-validation-key-per-method-run.md)); smaller M_val ruled out (plug-in CRPS bias favours narrow predictives); independent of Q-AVI's S | every posterior stage |
| Validation timestamps | K = 50 evenly spaced windows per history (min(K, n), ends included); training stays stride-1 ([ADR 0003](adr/0003-validation-at-fixed-timestamps.md)) | both validation sets |
| Diagnostics | `nll`, `rmse`, `coverage_95` scored once on the best checkpoint, same samples, logged at the best epoch; never select | every posterior stage |
| Baseline | `val_crps_baseline`, criterion before epoch 0, same key; diagnostic only | every posterior stage |
| Cap / patience | 300 epochs / 60 (≈ 20% of cap, covers Q-AVI's oscillations), `min_delta = 0`; shared, never tuned | MCD, BBB, Q-AVI, C-AVI |
| Chain | fixed 300 epochs, first 100 burn-in, `thinning` set so M = 100 samples are retained | pSGLD |
| Prior | `IsotropicGaussianPrior(precision=1.0)`, never tuned | every posterior stage |
| Optimizer | Adam, constant LR, batch size 32 | posterior stages |
| Optimizer | Adam, cosine decay `lr` → `0.01·lr` over `max_epochs`, stepped per epoch | mean and variance stages |

**Role of the criterion.**

| Method | Best checkpoint | Early stopping | HPO objective (pruner reports) |
|---|---|---|---|
| MCD, BBB, Q-AVI, C-AVI | ✅ | ✅ | best-epoch validation CRPS (per-epoch CRPS) |
| pSGLD | — last epoch | — would truncate the chain | final retained chain's CRPS (running chain's CRPS after burn-in) |

**Search spaces** (*log* = log scale):

| Method | Tuned HPs |
|---|---|
| MCD | dropout rate 0.01–0.5; LR 1e-5–1e-2 *log* |
| BBB | LR 1e-5–1e-2 *log*; `initial_std` 1e-4–1e-1 *log* |
| pSGLD | `step_size` 1e-6–1e-2 *log*; `decay` 0.99 and `stability` 1e-5 fixed |
| Q-AVI | λ 1e-4–1 *log*; S ∈ {1, 4, 8, 16}; k ∈ {1, 2, 3, 5}; q 3–6; L 1–4; generator and discriminator LR 1e-5–1e-2 *log*; discriminator fixed (64-64 LeakyReLU MLP) |
| C-AVI | as Q-AVI, own budget; h derived from (q, L) |
| variance stage | LR 1e-4–1e-2 *log*; 20 trials, TPE, median pruning, on seed 0's frozen mean checkpoint; minimizes best-epoch Gamma NLL on the backbone validation set; cap, patience, batch size from the mean study ([#99](https://github.com/GroupiSP/qmodem/issues/99)) |

**Study runs.** One top-level `role=hpo_study` run per (stage, method), tagged `stage`, `method`, `backbone_run_id` (seed 0) and `n_trials`, logging `best.*`. Trials are `role=hpo_trial` children that restore that backbone with the same key `fold_in(0, method)`. Method runs, seed 0 included, are always **re-trained** from the frozen HPs.

## 6. Evaluation

- **Draws**: M = 100 posterior draws per window × **one** aleatoric noise sample each; HNN gets 100 aleatoric samples from its single mean. M is never tuned per method.
- **Test cases**: all 20 test cases of the fixed run, with 100 simulator reference samples per timestamp. **t0** has no predictions: dropped from every metric, used only as WSU's time origin.
- **Sample-based** means and intervals for every method (Q-AVI has no analytic form).

| Metric (stagereg `core.metrics`) | Definition |
|---|---|
| CRPS | **expected CRPS**, grid-free: `mean_j sample_crps(pred_samples, y_ref_j)` over reference samples = E_{Y∼true}[CRPS(F_pred, Y)]. Never Cramér distance |
| RMSE | vs the predictive sample mean |
| Coverage | empirical 2.5/97.5 percentiles, inclusive; level **0.95** is a protocol parameter |
| WSU | `core.metrics.wsu` **per trajectory**, t0 as origin; never the epoch `WeightedSpread` adapter |

**Aggregation**: per test case, then **unweighted mean over test cases** (primary), plus the same mean per current policy (secondary). No pooled values.

## 7. Validity

A seed enters the results only if all checks hold. A failure **invalidates the seed for all methods** and the seed is re-run; methods are never dropped individually.

| Check | Rule |
|---|---|
| Shared backbone | every method run of seed r has the same `mlflow.parentRunId`; the backbone's `data_generation_run_id` tag equals the protocol's |
| Completeness | every run `FINISHED`; every core method has a run for the seed |
| Frozen HPs | method runs log the HPO-selected params, identical across seeds |
| Versions | one `stagereg_version` and one qmodem commit across all runs |
| Posterior health | Q-AVI: draws not collapsed (conv-weight spread > 0; discriminator accuracy not stuck at 1.0). pSGLD: burn-in respected, ≥ M samples. Values reported |

## 8. Reporting

**Statistics**: per metric, per-seed mean over test cases, then **mean ± std across seeds**. **Paired differences** vs BBB and vs HNN per seed, 95% bootstrap CI over seeds. No p-values.

**Outputs** (T1, T2, F1, F2 primary):

| ID | Content | Answers |
|---|---|---|
| **T1** | RMSE, CRPS, coverage@0.95, WSU per method (mean ± std) + per-policy columns | overall ranking |
| **T2** | paired Δ vs BBB and HNN (bootstrap CI) + per-policy columns; one row per arm | is Q-AVI's gain real |
| **T3** | posterior-stage wall-clock, trainable params, draws/s (first 1024 posterior-validation windows, `DRAWS_PER_SECOND_WINDOWS`) | cost |
| **F1** | reliability curve: empirical vs nominal coverage, levels 0.5…0.95 | calibration beyond one level |
| **F2** | epistemic / aleatoric variance over SoC0, per method, representative case per policy, event marked | where Q-AVI adds epistemic spread |
| **F3** | per-test-case CRPS bars, all methods, grouped by policy | is the gain uniform |
| **F4** | RUL 95% bands vs simulator reference band, all methods, representative case per policy, event marked | qualitative picture, incl. bimodal pre-event reference |

**Representative case**: per current policy, the test case with the **median** HNN CRPS at seed 0. Fixed before any method results are seen.

## Limitations

- N counts overlapping windows (§4).
- R = 5 seeds on one fixed dataset: CIs cover training randomness, not dataset resampling.
- Iteration-0 variance only. Outer variance/posterior iterations would make variance checkpoints method-specific, and the protocol would then share the iteration schedule instead.
