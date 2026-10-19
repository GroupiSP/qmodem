---
applyTo: "Makefile"
---

The files in this directory are not under active maintenance and can be disregarded when implementing changes.

Every script in `scripts/battery_multiple_scenarios/` is maintained. The legacy `run_hpo.py` and `run_train.py` were deleted in #115, replaced by `run_{mean,variance,posterior}_study.py`, `train_backbone.py` and `train_method_run.py`.
