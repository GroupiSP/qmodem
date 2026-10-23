# Closed-form CRPS for validation and test scoring

**Status**: accepted

The CRPS of a posterior is computed from the **posterior predictive** itself, the exact mixture `F = (1/M) Σ_m N(μ_m, s²)` over the M posterior draws, with the closed-form mixture CRPS of stagereg ≥ 0.5.0 (`expected_gaussian_mixture_crps`, Grimit et al. 2006), instead of the sample CRPS of noisy predictive samples drawn from F ([#144](https://github.com/GroupiSP/qmodem/issues/144)).

## Validation ([#145](https://github.com/GroupiSP/qmodem/issues/145))

The **validation criterion** of every posterior stage is the mean closed-form CRPS of F against the posterior validation targets (`ClosedFormContinuousRankedProbabilityScore(name="crps")`, R = 1 reference sample per window: the target). Early stopping, checkpoint selection, `val_crps_baseline` and posterior HPO read it. The diagnostics (NLL, RMSE, coverage@0.95) are exact for F as well, so validation draws no predictive samples. M_val stays at 100 posterior draws.

### Considered options

| Option | Verdict |
|---|---|
| Sample CRPS of M_val noisy predictive samples (status quo) | Rejected: applies to any predictive, but carries Monte Carlo noise from the aleatoric samples and a plug-in bias that favours narrow predictives; epochs and trials are compared through that noise. |
| **Closed-form mixture CRPS** | Chosen: deterministic given the draws, unbiased for F, cheaper (no samples, `O(M²)` per window). Every posterior here is a Gaussian mixture over draws with a shared aleatoric variance, so the generality of the sample CRPS is not needed. |

### Consequences

- **Validation CRPS is not comparable with earlier runs** (different estimator). Every posterior HPO study must be rerun; early-stopping decisions and best epochs change.
- Validation needs only the draw key of `split(validation_key)` ([ADR 0002](0002-posterior-validation-key-per-method-run.md)); qmodem's `score_validation` refuses a suite that draws predictive samples.
- M_val is no longer constrained by the plug-in bias ([ADR 0003](0003-validation-at-fixed-timestamps.md)); it is kept at 100 for validation/evaluation parity.

## Test scoring ([#147](https://github.com/GroupiSP/qmodem/issues/147))

Each test case's posterior predictive is held in RUL units as `CasePredictive` (`draw_means (T−1, M)`, `aleatoric_variance (T−1,)`; affine unscale `a·μ_m + b`, `a²·s²`). Every test metric is exact given the draws: CRPS is the mean closed-form mixture CRPS against the simulator reference samples (t0 dropped); RMSE is scored on the mixture mean; coverage and WSU on the central mixture interval (stagereg's mixture quantile). The RUL-over-time figure draws the same mean and interval; the reference band stays the empirical interval of the reference samples. Both bands go through one `IntervalForecast` interface. No aleatoric noise is sampled: the evaluation key only selects the draws.

### Considered options

| Option | Verdict |
|---|---|
| Sample CRPS, sample mean and empirical percentiles of M noisy predictive samples (status quo) | Rejected: MC noise in every metric and in the plot; each 95% tail rests on ~2.5 samples; the plug-in CRPS bias favours narrow predictives. |
| **Exact mixture CRPS, mean and quantiles** | Chosen: deterministic given the draws, and figure and metrics cannot disagree. Any number of draws (pSGLD's retained chain) goes through the same path. |

### Consequences

- **Test metrics are not comparable with earlier runs** (different estimators).
- The `evaluation/test/<case>/predictive` artifact stores the posterior predictive (`draw_means`, `aleatoric_variance`) instead of noisy samples and a variance split; rebuilding a case from it reproduces every logged metric. Older artifacts are not readable.
- `EvaluationParameters.num_samples` is `num_draws` (`--num-draws` in `evaluate.py`).
