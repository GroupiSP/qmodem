"""Fitted data scalers logged as MLflow pyfunc models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlflow
import numpy as np
from mlflow.pyfunc import PyFuncModel

from ..data import DataScaler


@dataclass(frozen=True)
class ScalerArtifact:
    """A fitted data scaler to log as an MLflow pyfunc model.

    Attributes:
        name: Name of the logged model, e.g. ``"x_scaler"``.
        scaler: The fitted scaler.
        input_example: Example input, shaped like the data the scaler was fitted on.
        predict_method: Scaler method the pyfunc's ``predict`` calls, e.g.
            ``"inverse_transform"`` for the target scaler.
    """

    name: str
    scaler: DataScaler
    input_example: np.ndarray
    predict_method: str = "transform"


class _ScalerWrapper(mlflow.pyfunc.PythonModel):
    """Pyfunc model whose ``predict`` calls one method of a fitted scaler."""

    def __init__(self, scaler: DataScaler, predict_method: str = "transform") -> None:
        self.scaler = scaler
        self.predict_method = predict_method

    def predict(self, model_input: np.ndarray, params: Any = None) -> np.ndarray:
        return getattr(self.scaler, self.predict_method)(model_input)


class LoadedScaler:
    """Scikit-learn-like view of a scaler logged with ``log_scaler``.

    Both ``transform`` and ``inverse_transform`` call the pyfunc's ``predict``, whose
    behaviour was fixed at logging time by ``ScalerArtifact.predict_method``.

    Attributes:
        pyfunc: The loaded pyfunc model.
    """

    def __init__(self, pyfunc: PyFuncModel) -> None:
        """Wrap a loaded pyfunc model.

        Args:
            pyfunc: The pyfunc model logged by ``log_scaler``.
        """
        self.pyfunc = pyfunc

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply the logged scaler method.

        Args:
            X: Input data.

        Returns:
            The scaled data.
        """
        return self.pyfunc.predict(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Apply the logged scaler method.

        Args:
            X: Input data.

        Returns:
            The (inverse-)scaled data.
        """
        return self.pyfunc.predict(X)


def log_scaler(artifact: ScalerArtifact) -> None:
    """Log a fitted scaler as a pyfunc model of the active MLflow run.

    Args:
        artifact: The scaler and how to log it.
    """
    mlflow.pyfunc.log_model(
        name=artifact.name,
        python_model=_ScalerWrapper(
            artifact.scaler, predict_method=artifact.predict_method
        ),
        input_example=artifact.input_example,
    )


def load_scaler(run_id: str, name: str) -> LoadedScaler:
    """Load a scaler logged with ``log_scaler``.

    Method runs pass their parent backbone run's ID, where the scalers are logged
    once.

    Args:
        run_id: The run the scaler was logged into.
        name: The scaler's model name, e.g. ``"x_scaler"``.

    Returns:
        The loaded scaler.
    """
    return LoadedScaler(mlflow.pyfunc.load_model(f"runs:/{run_id}/{name}"))


def load_fitted_scaler(run_id: str, name: str) -> DataScaler:
    """Load the fitted scaler object logged with ``log_scaler``.

    Unlike ``load_scaler``, the result has the scaler's own ``transform`` and
    ``inverse_transform``, so a run consuming a backbone can window its data with
    exactly the backbone's scaling, without refitting.

    Args:
        run_id: The run the scaler was logged into.
        name: The scaler's model name, e.g. ``"x_scaler"``.

    Returns:
        The fitted scaler.
    """
    model = mlflow.pyfunc.load_model(f"runs:/{run_id}/{name}")
    return model.unwrap_python_model().scaler
