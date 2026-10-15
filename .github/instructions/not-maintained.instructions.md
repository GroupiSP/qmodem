---
applyTo: "Makefile,scripts/battery_multiple_scenarios/run_hpo.py,scripts/battery_multiple_scenarios/run_hpo_qavi.py,scripts/battery_multiple_scenarios/run_train.py,scripts/battery_multiple_scenarios/run_train_qavi.py"
---

The files in this directory are not under active maintenance and can be disregarded when implementing changes.

- `scripts/battery_multiple_scenarios/run_hpo.py` and `run_hpo_qavi.py` are the legacy posterior HPO scripts. They no longer import (their HPO containers were removed in #114) and are replaced by the per-method posterior studies (#115).
- `scripts/battery_multiple_scenarios/run_train.py` and `run_train_qavi.py` are the legacy joint-training scripts. They no longer import (the CNN with its Gaussian head, `ConvType` and the Bayesian conv layers were removed in #109) and are replaced by `train_backbone.py` + `train_method_run.py` (HNN, BBB; MCD #111, pSGLD #110, Q-AVI/C-AVI #112/#113).
