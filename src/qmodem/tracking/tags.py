"""The tags of protocol runs: roles, stages, provenance and protocol overrides."""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum

from .tracker import stagereg_version


class Role(StrEnum):
    """The ``role`` tag of every run the protocol creates."""

    BACKBONE = "backbone"
    METHOD = "method"
    HPO_STUDY = "hpo_study"
    HPO_TRIAL = "hpo_trial"


class Stage(StrEnum):
    """The training stages (``stage`` tag of HPO runs, prefix of logged params)."""

    MEAN = "mean"
    VARIANCE = "variance"
    POSTERIOR = "posterior"


MEAN_STUDY_TAG = "mean_study_run_id"
"""Backbone-run tag: the mean study whose ``best.*`` configured the mean stage."""

VARIANCE_STUDY_TAG = "variance_study_run_id"
"""Backbone-run tag: the variance study whose ``best.*`` set the variance LR."""


def backbone_tags(
    *,
    seed: int,
    data_generation_run_id: str,
    mean_study_run_id: str | None = None,
    variance_study_run_id: str | None = None,
) -> dict[str, str]:
    """Return the tags of a backbone run.

    Args:
        seed: The protocol seed r.
        data_generation_run_id: MLflow run ID of the data-generation run.
        mean_study_run_id: The mean study the mean HPs come from, or ``None``.
        variance_study_run_id: The variance study the variance LR comes from, or
            ``None``.

    Returns:
        The ``role``, ``seed``, ``data_generation_run_id`` and ``stagereg_version``
        tags, plus ``MEAN_STUDY_TAG`` / ``VARIANCE_STUDY_TAG`` when given.
    """
    studies = {
        MEAN_STUDY_TAG: mean_study_run_id,
        VARIANCE_STUDY_TAG: variance_study_run_id,
    }
    return {
        "role": Role.BACKBONE,
        "seed": str(seed),
        "data_generation_run_id": data_generation_run_id,
        "stagereg_version": stagereg_version(),
        **{key: value for key, value in studies.items() if value is not None},
    }


PROTOCOL_OVERRIDES_TAG = "protocol_overrides"
"""Tag ``"true"`` on a run whose protocol-fixed settings were overridden."""

PROTOCOL_OVERRIDES_PARAMS_TAG = "protocol_overrides.params"
"""Tag listing the overridden settings (flattened names, comma-separated)."""


def protocol_override_tags(names: Sequence[str]) -> dict[str, str]:
    """Return the tags marking a run whose protocol-fixed settings were overridden.

    Such runs (e.g. smoke runs with a shrunk epoch cap) are skipped by tag
    selection unless explicitly included (``RunCriteria.include_overridden``).

    Args:
        names: The overridden settings, e.g. ``["stage/max_epochs"]``.

    Returns:
        ``{}`` if nothing was overridden, else ``PROTOCOL_OVERRIDES_TAG = "true"``
        and ``PROTOCOL_OVERRIDES_PARAMS_TAG`` with the sorted names.
    """
    if not names:
        return {}
    return {
        PROTOCOL_OVERRIDES_TAG: "true",
        PROTOCOL_OVERRIDES_PARAMS_TAG: ",".join(sorted(names)),
    }


HPO_STUDY_ROLE = Role.HPO_STUDY
HPO_TRIAL_ROLE = Role.HPO_TRIAL
HPO_TRIAL_TAG = "hpo.trial"
PARENT_RUN_TAG = "mlflow.parentRunId"


def hpo_study_tags(
    *,
    stage: str,
    n_trials: int,
    method: str | None = None,
    backbone_run_id: str | None = None,
    seed: int | None = None,
) -> dict[str, str]:
    """Return the tags of an HPO study run.

    Args:
        stage: The tuned stage, e.g. ``"mean"``, ``"variance"`` or ``"posterior"``.
        n_trials: The study's trial budget.
        method: The tuned method (posterior studies), or ``None`` for a backbone
            stage, in which case no ``method`` tag is set.
        backbone_run_id: The backbone run the trials restore, or ``None`` if they
            restore none (mean study), in which case no tag is set.
        seed: The protocol seed the trials' keys derive from, or ``None`` to set
            no ``seed`` tag.

    Returns:
        The ``role``, ``stage``, ``n_trials``, ``stagereg_version`` and, where
        applicable, ``method``, ``backbone_run_id`` and ``seed`` tags.
    """
    optional = {
        "method": method,
        "backbone_run_id": backbone_run_id,
        "seed": None if seed is None else str(seed),
    }
    return {
        "role": HPO_STUDY_ROLE,
        "stage": stage,
        "n_trials": str(n_trials),
        "stagereg_version": stagereg_version(),
        **{key: value for key, value in optional.items() if value is not None},
    }


def hpo_trial_tags(
    *,
    study_run_id: str,
    number: int,
    stage: str,
    method: str | None = None,
    backbone_run_id: str | None = None,
    seed: int | None = None,
) -> dict[str, str]:
    """Return the tags of an HPO trial run, a child of its study run.

    Args:
        study_run_id: The parent study run.
        number: The Optuna trial number.
        stage: The tuned stage.
        method: The tuned method, or ``None`` for a backbone stage.
        backbone_run_id: The backbone run the trial restores, or ``None``.
        seed: The protocol seed the trial's key derives from, or ``None``.

    Returns:
        The ``role``, ``hpo.trial``, ``stage``, parent-run and, where applicable,
        ``method``, ``backbone_run_id`` and ``seed`` tags.
    """
    optional = {
        "method": method,
        "backbone_run_id": backbone_run_id,
        "seed": None if seed is None else str(seed),
    }
    return {
        "role": HPO_TRIAL_ROLE,
        HPO_TRIAL_TAG: str(number),
        "stage": stage,
        PARENT_RUN_TAG: study_run_id,
        **{key: value for key, value in optional.items() if value is not None},
    }
