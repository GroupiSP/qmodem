"""MLflow tracking of the protocol runs.

Modules: ``tracker`` (stagereg tracker, run setup), ``checkpoints`` (stagereg
checkpoints as run artifacts), ``tags`` (roles, stages, run tags), ``selection``
(choosing a consumed run) and ``scalers`` (fitted scalers as pyfunc models). The
public API is re-exported here.
"""

from .checkpoints import (
    CHECKPOINT_ARTIFACT_ROOT,
    MLflowCheckpointStore,
    _import_attribute,
    mlflow_checkpoint_store,
)
from .scalers import (
    LoadedScaler,
    ScalerArtifact,
    _ScalerWrapper,
    load_fitted_scaler,
    load_scaler,
    log_scaler,
)
from .selection import (
    IDENTITY_TAGS,
    RunCriteria,
    RunSelectionError,
    RunSelector,
    resolve_run_id,
    select_run,
)
from .tags import (
    HPO_STUDY_ROLE,
    HPO_TRIAL_ROLE,
    HPO_TRIAL_TAG,
    MEAN_STUDY_TAG,
    PARENT_RUN_TAG,
    PROTOCOL_OVERRIDES_PARAMS_TAG,
    PROTOCOL_OVERRIDES_TAG,
    VARIANCE_STUDY_TAG,
    Role,
    Stage,
    backbone_tags,
    hpo_study_tags,
    hpo_trial_tags,
    protocol_override_tags,
)
from .tracker import (
    MLFlowSetup,
    MLflowTracker,
    stagereg_version,
    track_dataframe,
    track_mlflow,
    unflatten_parameters,
)

# ``_import_attribute`` and ``_ScalerWrapper`` are pickled by reference as
# ``qmodem.tracking.<name>`` into checkpoints and scalers logged before the split
# into a package; they stay importable here so those runs still load.
__all__ = [
    "CHECKPOINT_ARTIFACT_ROOT",
    "HPO_STUDY_ROLE",
    "HPO_TRIAL_ROLE",
    "HPO_TRIAL_TAG",
    "IDENTITY_TAGS",
    "MEAN_STUDY_TAG",
    "PARENT_RUN_TAG",
    "PROTOCOL_OVERRIDES_PARAMS_TAG",
    "PROTOCOL_OVERRIDES_TAG",
    "VARIANCE_STUDY_TAG",
    "LoadedScaler",
    "MLFlowSetup",
    "MLflowCheckpointStore",
    "MLflowTracker",
    "Role",
    "RunCriteria",
    "RunSelectionError",
    "RunSelector",
    "ScalerArtifact",
    "Stage",
    "_ScalerWrapper",
    "_import_attribute",
    "backbone_tags",
    "hpo_study_tags",
    "hpo_trial_tags",
    "load_fitted_scaler",
    "load_scaler",
    "log_scaler",
    "mlflow_checkpoint_store",
    "protocol_override_tags",
    "resolve_run_id",
    "select_run",
    "stagereg_version",
    "track_dataframe",
    "track_mlflow",
    "unflatten_parameters",
]
