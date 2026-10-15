---
applyTo: "Makefile,scripts/battery_multiple_scenarios/run_hpo.py,scripts/battery_multiple_scenarios/run_hpo_qavi.py"
---

The files in this directory are not under active maintenance and can be disregarded when implementing changes.

- `scripts/battery_multiple_scenarios/run_hpo.py` and `run_hpo_qavi.py` are the legacy posterior HPO scripts. They no longer import (their HPO containers were removed in #114) and are replaced by the per-method posterior studies (#115).
