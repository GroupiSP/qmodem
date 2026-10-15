---
applyTo: "Makefile,scripts/battery_multiple_scenarios/run_hpo.py,scripts/battery_multiple_scenarios/run_train.py"
---

The files in this directory are not under active maintenance and can be disregarded when implementing changes.

- `scripts/battery_multiple_scenarios/run_hpo.py` is the legacy posterior HPO script (`run_hpo_qavi.py` was deleted in #112). It no longer imports (its HPO containers were removed in #114) and is replaced by the per-method posterior studies (#115).
- `scripts/battery_multiple_scenarios/run_train.py` is the legacy joint-training script (`run_train_qavi.py` was deleted in #112, superseded by `train_method_run.py --method q-avi`). It no longer imports (the CNN with its Gaussian head, `ConvType` and the Bayesian conv layers were removed in #109) and is replaced by `train_backbone.py` + `train_method_run.py` (HNN, MCD, BBB, pSGLD, Q-AVI; C-AVI #113).
