"""The method registry: everything the pipeline needs to know about each method.

Adding a method means one ``Method`` tag value, one module under
``qmodem.battery.methods`` holding its configuration, and one ``METHODS`` entry
here (configuration type, key index and posterior search space). The method-run
pipeline, the posterior studies and the scripts' ``--method`` choices all derive
from this registry.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping

from qmodem.hpo import Categorical, Dimension, FloatRange, IntRange

from ..train import Method
from .avi import CAVIConfig, QAVIConfig
from .base import MethodConfig
from .bbb import BBBConfig
from .hnn import HNNConfig
from .mcd import MCDConfig
from .psgld import PSGLDConfig


@dataclasses.dataclass(frozen=True)
class MethodEntry:
    """One registered method.

    Attributes:
        config: The method's configuration type.
        key_index: The fixed integer folded into the protocol seed's key
            (``method_key``). Never derive it from ``hash()``, which is salted per
            process.
        search_dimensions: The posterior HPs its study tunes (protocol §5), named
            after the flattened method-run params; empty if the method has no
            posterior stage to tune (HNN).
    """

    config: type[MethodConfig]
    key_index: int
    search_dimensions: tuple[Dimension, ...] = ()


_LEARNING_RATE = FloatRange("posterior/optimizer/learning_rate", 1e-5, 1e-2, log=True)

_AVI_DIMENSIONS: tuple[Dimension, ...] = (
    FloatRange("posterior/avi/adversarial_weight", 1e-4, 1.0, log=True),
    Categorical("posterior/avi/num_draws", (1, 4, 8, 16)),
    Categorical("posterior/avi/discriminator_steps", (1, 2, 3, 5)),
    IntRange("posterior/avi/n_qubits", 3, 6),
    IntRange("posterior/avi/n_layers", 1, 4),
    FloatRange("posterior/avi/generator/learning_rate", 1e-5, 1e-2, log=True),
    FloatRange("posterior/avi/discriminator/learning_rate", 1e-5, 1e-2, log=True),
)
"""Q-AVI's space, reused by C-AVI (its hidden width h derives from q and L)."""

METHODS: Mapping[Method, MethodEntry] = {
    Method.HNN: MethodEntry(HNNConfig, 0),
    Method.MCD: MethodEntry(
        MCDConfig,
        1,
        (FloatRange("posterior/dropout_rate", 0.01, 0.5), _LEARNING_RATE),
    ),
    Method.BBB: MethodEntry(
        BBBConfig,
        2,
        (_LEARNING_RATE, FloatRange("posterior/initial_std", 1e-4, 1e-1, log=True)),
    ),
    Method.PSGLD: MethodEntry(
        PSGLDConfig, 3, (FloatRange("posterior/step_size", 1e-6, 1e-2, log=True),)
    ),
    Method.Q_AVI: MethodEntry(QAVIConfig, 4, _AVI_DIMENSIONS),
    Method.C_AVI: MethodEntry(CAVIConfig, 5, _AVI_DIMENSIONS),
}
"""Every implemented method."""
