"""DEPRECATED: Shared utilities have moved to ``qmodem.utils``.

Use ``from qmodem.utils import ...`` instead.  This module re-exports the
symbols for backward compatibility and will be removed in a future release.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "scripts/_shared.py is deprecated. Use qmodem.utils instead.",
    DeprecationWarning,
    stacklevel=2,
)

from qmodem.utils import (  # noqa: F401, E402
    SHARED_PARAMS,
    TEST_SEED,
    TRAIN_SEED,
    create_battery_and_policy,
    get_run_dirs,
    load_battery_config,
    make_simulator_config,
    read_json,
    restore_model_from_checkpoint,
    restore_model_state,
    write_json,
)
