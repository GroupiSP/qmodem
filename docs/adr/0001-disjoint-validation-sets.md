# Disjoint validation sets per selection step

**Status**: accepted

The backbone (mean and variance stages) and the posterior stage are selected on two **disjoint, policy-stratified validation sets**: the **backbone validation set** stops, checkpoints and tunes the mean and variance stages, and the **posterior validation set** carries the validation criterion (posterior early stopping, best checkpoint, posterior HPO). The data was regenerated to make this possible: train 10 (7/3), backbone validation set 20 (14/6), posterior validation set 20 (14/6), test 20 (14/6), with policies assigned by fixed counts.

## Context

In the `stages` experiment every posterior stage regressed from epoch 0: its best epoch was always 0 and posterior HPO always picked the 1e-5 LR floor ([#120](https://github.com/GroupiSP/qmodem/issues/120)). The cause is a **winner's curse**. The mean stage early-stops at the argmin of a noisy validation curve, so the backbone's score on those histories is biased low. The posterior stage was then stopped and tuned on the **same** histories, so its epoch 0 inherits that bias and any weight update looks like a regression. The old validation set (5 histories, i.i.d. Bernoulli policy choice) was too small to split, and a split could contain no policy-1 history.

## Considered options

| Option | Verdict |
|---|---|
| Shared validation set (status quo) | Rejected: causes the winner's curse above; the comparison protocol cannot tell whether any method improves on its backbone. |
| k-fold / cross-validated selection | Rejected: multiplies backbone runs and posterior studies by k for every method and seed, and the selection steps would still need to be disjoint within each fold. Out of budget. |
| Fresh posterior validation set appended to the existing data, reusing the existing backbones | Rejected: cheapest, but the backbones stay selected on 5 histories with a random policy mix, the test set stays at 10 histories with only 3 policy-1 cases, and mixing two data-generation runs complicates provenance. |
| **Regenerate with two disjoint, stratified validation sets** | Chosen: each validation set (20 histories) is low-noise on its own; fixed 70/30 counts guarantee policy-1 histories in every split and match the test mixture; one data-generation run documents the whole split. |

## Consequences

- Backbones selected on the old data cannot be reused: every backbone run, study and method run is redone on the new data-generation run, in a fresh MLflow experiment. The `stages` experiment and data-gen run `e52603fb…` stay as the "before" reference.
- Merging the two validation sets back into one (e.g. to "use more data") silently reintroduces the winner's curse. Any such change must supersede this ADR.
- Every consumer names the validation set it uses: backbone training and backbone HPO use the backbone validation set; posterior stages, method runs and posterior HPO use the posterior validation set; evaluation uses the test set only.
- Backbones trained before the cosine learning-rate decay (constant-LR optimizer state in their checkpoints) no longer restore: their optimizer state does not match the decayed schedule, so they must be retrained.
- The pickled test simulation config keeps the 70/30 Bernoulli policy prior, but only for the evaluation reference band (the policy is unknown before the event). The histories themselves, test included, use fixed policy counts, each split drawing from its own random stream spawned from `train_seed` or `test_seed`.
