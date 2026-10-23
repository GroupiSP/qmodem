# Validation at fixed timestamps

**Status**: accepted

Both validation sets (backbone and posterior) are scored only at their **validation timestamps**: for a history with n stride-1 windows, the min(K, n) windows at `linspace(0, n − 1, min(K, n))`, first and last included (the same rule as a test case's evaluation timestamps). **K = 50**, one protocol constant (`VALIDATION_TIMESTAMP_COUNT`) shared by both sets. The selection happens when the windows are prepared, so every consumer (backbone stages, backbone HPO, posterior stages, posterior HPO, `val_crps_baseline`, the once-only diagnostics) gets the reduced sets with no change of its own. Training windows stay stride-1, so the posterior's dataset size N is unchanged ([#138](https://github.com/GroupiSP/qmodem/issues/138), [#142](https://github.com/GroupiSP/qmodem/issues/142)).

## Context

Validating every stride-1 window (~14k windows per set of 20 histories, 100 predictive samples each for a posterior) dominates the cost of posterior stages and their HPO studies. Neighbouring windows share 9 of 10 steps, so the effective sample size is near the number of histories, not the number of windows. Selection compares epochs and trials on the same windows under the same validation key ([ADR 0002](0002-posterior-validation-key-per-method-run.md)), so subsampling noise is shared and largely cancels in the differences that matter.

## Considered options

| Option | Verdict |
|---|---|
| Full stride-1 validation (status quo) | Rejected: slow, and weights histories by their window count (485–842), so long histories dominate. |
| Fewer predictive samples | Rejected: breaks the "validation and evaluation alike" invariant of `PREDICTIVE_SAMPLE_COUNT`; the plug-in CRPS bias favours narrow predictives. *Note*: the bias rationale is superseded by [ADR 0004](0004-closed-form-crps.md) (validation draws no predictive samples); M_val is kept for validation/evaluation parity. |
| Random window subsample | Rejected: does not guarantee every degradation stage of a history is represented. |
| Separate K per validation set | Rejected: "validation timestamps" should have one meaning. |
| **K evenly spaced windows per history, K from a study** | Chosen: deterministic, equal weight per history (keeps the policy stratification of [ADR 0001](0001-disjoint-validation-sets.md)), mirrors the test set's evaluation timestamps. |

## K study ([#141](https://github.com/GroupiSP/qmodem/issues/141))

Data-gen run `b5e7394d5c7140b0ac5c827099879f2a`, protocol seed 0, code-default hyperparameters, full stride-1 validation; throwaway experiment `k_study_138`: backbone `b61eb543…` (mean best epoch 27 of 79, variance 73 of 125), BBB `977c4e45…` (10 of 72), Q-AVI `de66db95…` (201 of 263). No LR was raised by hand: neither posterior curve was flat. Per-window validation scores (stage loss for the backbone set, CRPS for the posterior set) were recorded at every epoch. Variance components come from a one-way random-effects ANOVA (history vs. window within history); SE(Δ), Δ = criterion(best epoch) − criterion(e), from a bootstrap over histories (B = 5000).

**Rule change** (decided after a first pass, deviating from #138):

- The reference is the **equal-weight full set** (every window, each history weighted equally), not the stride-1 full set: K vs. stride-1 mixes subsampling with a change of weighting (on the mean stage it levels off at ≈1.12–1.15 even at K = 400).
- The decision uses **near-best pairs**: the 10 epochs whose full-set score is closest to the best epoch's. Pooled over all pairs, the ratio is dominated by far-from-best epochs and is non-monotone in K.
- Per set: the smallest K with SE(Δ) ≤ 1.05× the reference SE(Δ), max over the set's stages or methods. K = the max over both sets.

**Variance components** (at the best epoch; ESS(K) = 20·K / (1 + (K − 1)·ICC)):

| series | ICC | design effect, full set | ESS, full set | ESS(10/20/50/100) |
|---|---|---|---|---|
| backbone / mean MSE | 0.20 | 145 | 98 | 71 / 83 / 92 / 95 |
| backbone / variance Gamma NLL | 0.12 | 86 | 166 | 96 / 122 / 146 / 156 |
| posterior / BBB CRPS | 0.24 | 170 | 82 | 63 / 72 / 78 / 80 |
| posterior / Q-AVI CRPS | 0.23 | 160 | 88 | 66 / 75 / 82 / 85 |

**SE(Δ) ratio on near-best pairs**, against the equal-weight full set (against stride-1 in brackets):

| K | mean MSE | variance NLL | BBB CRPS | Q-AVI CRPS |
|---|---|---|---|---|
| 10 | 1.289 (1.472) | 0.983 (0.895) | 1.037 (1.084) | 1.010 (1.057) |
| 20 | 1.147 (1.311) | 0.966 (0.880) | 0.899 (0.940) | 0.981 (1.027) |
| **50** | **1.041** (1.189) | 0.967 (0.881) | 0.961 (1.004) | 0.988 (1.034) |
| 100 | 1.026 (1.172) | 0.995 (0.906) | 0.984 (1.029) | 1.007 (1.055) |
| winner | 50 | 10 | 10 | 10 |

Backbone set: max(50, 10) = 50; posterior set: 10; **K = 50**.

**Best-epoch check** (the full-set score at the epoch chosen under K within one full-set SE of the full-set best): passes for every K ∈ {10, 20, 50, 100}; at K = 50 the gaps are 0.05 (mean), 0.001 (variance), 0.06 (BBB) and 0.12 (Q-AVI) full-set SEs.

**Caveats**: ratios rest on 10 near-best pairs (noise ≈ ±2–3%, hence non-monotone in K); the mean stage at K = 50 (1.041) is close to the 1.05 threshold, K = 20 fails clearly; one run per stage or method, seed 0, default HPs. The analysis code is one-off and not committed.

## Consequences

- Any change of K, or a return to full-set validation, changes every validation score and selection. Such a change must supersede this ADR and requires rerunning the backbone and posterior studies.
- `val_crps_baseline` uses the validation timestamps (it reads the reduced posterior validation set), so it still equals epoch 0's score ([ADR 0002](0002-posterior-validation-key-per-method-run.md)).
- K is logged on every run as the `validation_timestamp_count` param, and `windows_backbone_val` / `windows_posterior_val` count the reduced sets.
- No existing run is invalidated: `stages_disjoint_val` holds only data generation.
- ADRs 0001 and 0002 stay in force.
