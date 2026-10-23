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
