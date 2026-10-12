# Do stagereg's metrics match qmodem's numerically? (GroupiSP/qmodem#90)

**Verdict.** On identical inputs, RMSE, coverage, WSU and point-CRPS match qmodem bit-for-bit. One condition: WSU must be called per trajectory, with the t0 coordinate included. **Grid CRPS does not match.** stagereg's `crps` is the *expected* CRPS under the reference samples. qmodem's `crps` is the squared CDF distance (Cramér). They differ by exactly `∫F_true(1−F_true) dx`. Most remaining gaps come from the pipeline, not the formulas: how means and intervals are made, and how units are aggregated.

Sources: qmodem `fe4a733` (`src/qmodem/metrics.py`, `src/qmodem/battery/scoring.py`, `src/qmodem/battery/evaluate.py`). stagereg `3d7af91` (`src/stagereg/core/metrics.py`, `core/metric_registry.py`, `jax/metrics.py`).

## Verdict table

| Metric | qmodem def | stagereg def | Numerically equal? | Matters for comparison protocol? |
|---|---|---|---|---|
| RMSE | `sqrt(mean((rul − mean(samples_pred))²))` over timestamps 1..N-1 of one test case (`scoring.py:33,94`). `rmse_average` = **mean of per-case RMSEs** (`evaluate.py:204`) | `rmse(targets, mean)` (`core/metrics.py:132`). The epoch adapter **pools all scoring units** (`metric_registry.py:312`). The JAX predictor's `mean` is the **analytic** mean (Gaussian `loc` / mixture mean, `jax/metrics.py:351`), not a sample mean | **Yes** (0 diff) for the same `y` and `mean` | **Yes.** Mean of per-case RMSEs ≠ pooled RMSE (301.2 vs 303.7 in the check). Sample mean ≠ analytic mean (MC noise). Fix one aggregation and one mean source |
| Coverage | `np.percentile(samples_pred,[2.5,97.5])`. Inclusive `lo ≤ y ≤ hi`. Mean over ts 1..N-1, then mean over cases (`scoring.py:29,37,63`) | `coverage(y, lo, hi)`, inclusive (`core/metrics.py:138`). The interval comes from `PredictionInterval(level)`. In the JAX pipeline it is the **exact Gaussian** `μ±z·σ` or the **mixture quantile** (`jax/metrics.py:479-497`), not the sample percentile | **Yes** (0 diff) for the same bounds | **Yes.** Sample-percentile and analytic intervals differ: 95% width median −4.5%, up to 23% with N=100. Use one interval construction and the same level (0.95) for every method |
| WSU | `Σ_{i=1..N-2} ½(w_i + w_{i+1})·(t_i − t_0) / (t_{N-1} − t_0)²` with `w = hi − lo` (95% percentile). **t_0 (no prediction) is the time origin** (`scoring.py:67`) | Same formula on generic `(lower, upper, x)`. Stagereg says it "preserves QMoDeM's established WSU definition" (`core/metrics.py:148`). The adapter **sorts all units by coordinate** (`metric_registry.py:428`) | **Yes** (0 diff) when called per trajectory with `x = times incl. t0` and a dummy bound at index 0. **No** (8.6% off) when called on units 1..N-1 only, because the origin shifts to t_1 | **Yes.** WSU is a per-trajectory metric. The epoch `WeightedSpread` adapter pools and sorts units from different test cases, so it gives a meaningless value. Call `core.metrics.wsu` per test case and pass t0 |
| Grid CRPS (vs reference samples) | `∫(F_true − F_pred)² dx` with empirical CDFs, trapezoid on the grid (`metrics.py:17`). This is the squared Cramér distance | `∫[F_true(1−F_pred)² + (1−F_true)F_pred²] dx` = E_{Y∼true}[CRPS(F_pred, Y)] (`core/metrics.py:55`) | **No.** Analytically and numerically, `stagereg − qmodem = ∫F_true(1−F_true) dx` (max diff 8e-16). The gap was a median 64% of stagereg's value | **Absolute values: yes. Rankings: no.** The offset depends only on the reference samples and grid, not on the model, so method rankings are the same on a shared reference set. Choose one definition and do not mix numbers from both |
| `average_crps` aggregation | Mean over **all N timestamps, including t0**, where `samples_pred=[]`, so `F_pred≡0` and the term is `∫F_true²` (`scoring.py:97`, `evaluate.py:147`). That term was about 829 in the check | One score per scoring unit. Empty samples **raise** (`core/metrics.py:14`). Adapter mean over units (`metric_registry.py:527`) | **No** (65% off in the check, combined with the definition gap above) | **Yes.** The t0 term is the same for every model, so it only dilutes the score, but it is a qmodem artefact. Drop t0 in the protocol |
| Point CRPS (grid) | `∫(1{x≥y} − F_pred)² dx` on the grid (`metrics.py:26`). Not used by `evaluate.py` | Identical (`core/metrics.py:80`) | **Yes** (0 diff) | Low |
| Sample CRPS (exact, energy form) | Not in qmodem | `E|X−y| − ½E|X−X′|`, exact, no grid (`core/metrics.py:103`) | n/a. Grid point-CRPS on qmodem's grid is off by a median of −2% in the interior and up to −90% when the target is near 0 or above 4950 (grid truncation). It matches on a wide, fine grid (54.35 vs 54.40) | **Yes.** qmodem's grid `np.arange(0, 5000, 50)` **excludes 5000** and truncates mass below 0. Prefer `sample_crps`, or use a grid that covers the support |
| Empirical CDF | `count(samples ≤ x)/n`. Empty → 0 | `mean(samples ≤ x)`. Empty → 0 | **Yes** | No |

## Other differences

- **Input shape.** qmodem's `samples_pred` has shape `(n, 1)` (`mc_sample`). stagereg needs 1-D vectors, or `(units, samples)` matrices in the adapters. Call `.ravel()` first.
- **Validation.** stagereg rejects NaN/inf, empty inputs, a non-increasing grid and `lo > hi`. qmodem rejects none of these, and `np.percentile([])` gives NaN at t0.
- **Coverage level.** Both default to 0.95 (qmodem hard-codes `[2.5, 97.5]`). stagereg names each interval by `level`, so make the level an explicit protocol parameter.

## Implications for the comparison protocol

1. Pick one **CRPS definition**: stagereg's expected CRPS (a proper score vs the simulator reference) or qmodem's Cramér distance. Report only that one. Rankings agree on a shared reference set.
2. Score timestamps **1..N-1 only**. Drop qmodem's t0 term.
3. Fix the **aggregation**: per-case then mean over cases (qmodem) vs pooled units (stagereg adapters). Compute WSU per trajectory either way.
4. Fix the **interval and mean source** for all UQ methods: MC sample percentiles/means vs analytic quantiles/means.
5. Fix the **CRPS grid** (cover the full RUL support and include the endpoint), or use the grid-free `sample_crps` for point targets.

## Reproduction

A throwaway script was run with `uv run --with /Users/gbltostibalduc/Repos/stagereg python parity.py` inside qmodem's env (not committed). It builds `TestCaseResults` objects (10 cases × 20 timestamps, 100 true and 100 predicted samples, grid `arange(0, 5000, 50)`), feeds the same arrays to `stagereg.core.metrics` and the `metric_registry` adapters, and reports the maximum relative difference per metric (the numbers above).
