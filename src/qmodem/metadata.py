"""Typed metadata structures and JSON persistence helpers for QMoDeM training runs.

This module defines :class:`TypedDict` classes that represent the metadata saved
alongside model checkpoints, and provides :func:`save_metadata` /
:func:`load_metadata` helpers that serialise and deserialise that metadata as
human-readable JSON.

Types:
    SimulatorConfig: Primitive battery-simulation parameters.
    TrainingParams: Data windowing and dataset size parameters.
    QAVITrainingParams: :class:`TrainingParams` extended with ``batch_w``.
    BaseModelParams: Shared CNN architecture parameters.
    MCDModelParams: :class:`BaseModelParams` extended with ``dropout_rate``.
    ScalingParams: Target-normalisation parameters.
    PQCParams: Quantum circuit parameters (QAVI method only).
    TrainingMetadata: Top-level metadata common to all methods.
    QAVITrainingMetadata: :class:`TrainingMetadata` extended with ``pqc_params``.

Functions:
    save_metadata: Serialise a metadata dict to ``metadata.json``.
    load_metadata: Deserialise ``metadata.json`` back into a dict.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict


class SimulatorConfig(TypedDict):
    """Primitive battery-simulation parameters (JSON-serialisable subset).

    Intentionally excludes non-serialisable runtime objects such as
    ``battery`` and ``discharge_policy``.
    """

    n_simu: int
    v_cut: float
    soc_0: float
    dt: float
    omega_std: float
    eta_std: float


class TrainingParams(TypedDict):
    """Data windowing and dataset size parameters."""

    window_size: int
    stride: int
    n_histories_train: int
    n_histories_val: int
    soc_range: list[float]


class QAVITrainingParams(TrainingParams):
    """Training params extended with QAVI-specific batch width."""

    batch_w: int


class BaseModelParams(TypedDict):
    """Shared CNN architecture parameters."""

    n_filters: int
    kernel_size: int


class MCDModelParams(BaseModelParams):
    """CNN parameters for the MC-Dropout method."""

    dropout_rate: float


class ScalingParams(TypedDict):
    """Target normalisation parameters."""

    normalize: bool
    y_max: float


class PQCParams(TypedDict):
    """Quantum circuit parameters (QAVI method only)."""

    n_qubits: int
    n_pqc_layers: int


class TrainingMetadata(TypedDict):
    """Top-level metadata common to all training methods.

    Attributes:
        method: Tag identifying the training method (e.g. ``"het_cnn"``).
        simulator_config: Primitive battery-simulation parameters.
        training_params: Data windowing and dataset size parameters (base or
            method-specific; see :class:`TrainingParams` and
            :class:`QAVITrainingParams`).
        model_params: CNN architecture parameters (base or method-specific;
            see :class:`BaseModelParams` and :class:`MCDModelParams`).
        scaling_params: Target normalisation parameters.
    """

    method: str
    simulator_config: SimulatorConfig
    training_params: TrainingParams | QAVITrainingParams
    model_params: BaseModelParams | MCDModelParams
    scaling_params: ScalingParams


class QAVITrainingMetadata(TrainingMetadata):
    """Training metadata for the QAVI method, including quantum-circuit parameters.

    Attributes:
        pqc_params: Quantum circuit parameters.
    """

    pqc_params: PQCParams


def save_metadata(metadata_dir: Path, metadata: TrainingMetadata) -> None:
    """Serialise *metadata* to ``metadata.json`` inside *metadata_dir*.

    Args:
        metadata_dir: Directory in which to write ``metadata.json``.
        metadata: Metadata dict to serialise.
    """
    with open(metadata_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def load_metadata(metadata_dir: Path) -> TrainingMetadata:
    """Deserialise ``metadata.json`` from *metadata_dir*.

    Args:
        metadata_dir: Directory containing ``metadata.json``.

    Returns:
        The deserialised metadata dict.
    """
    with open(metadata_dir / "metadata.json", encoding="utf-8") as f:
        return json.load(f)
