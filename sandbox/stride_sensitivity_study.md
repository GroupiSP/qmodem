# Stride sensitivity study
What is the effect on the posterior stage training of reducing the time-window stride to 50%, 20% and 10% overlap?

## Specifics
- Create three different MLFlow experiments for 50%, 20% and 10% overlap.
- Fit mean, variance (HNN part) and a single Bayesian method -> BBB
- Tracks the ratio of epistemic to aleatoric variance (Var_m(μ_m) / s²) on posterior validation.
- Compute the expected CRPS floor (½·E|Y−Y′|) of the simulator's own RUL distribution, which is bimodal before the event. Report all CRPS scores relative to this floor.
- Quantify (relative) CRPS, RMSE and coverage@0.95 against HNN on the same backbone for a single test case from the shorter-RUL load policy.
- [to refine] Quantify the noise of the CRPS throughout epochs.
 