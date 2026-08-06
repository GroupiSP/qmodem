from __future__ import annotations

import io
from typing import Any

import mlflow
import numpy as np
from mlflow.pyfunc import PyFuncModel

from qmodem.data import DataScaler

type ScalersDescription = dict[
    str, tuple[DataScaler, str, np.ndarray]
]  # (scaler, predict_method, input_example)


class _ScalerWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, scaler: DataScaler, predict_method: str = "transform"):
        self.scaler = scaler
        self.predict_method = predict_method

    def predict(self, model_input: np.ndarray, params=None):
        return getattr(self.scaler, self.predict_method)(model_input)


class _ScalerInverseWrapper:
    """Wraps around a loaded mlflow.pyfunc.PyFuncModel logged as
    `src/qmodem/battery/data_processing.py:_ScalerWrapper` to provide a scikit-learn-
    like interface with ``transform`` and ``inverse_transform`` methods.

    This is useful for loading the scaler from MLflow and using it to inverse-transform predictions back to the original scale.
    The behaviour of the `predict` method of the pyfunc is determined at logging time. For example, if the scaler normalized the labels,
    the `predict` method will perform the inverse transformation to return the original labels.
    """

    def __init__(self, pyfunc: PyFuncModel):
        self.pyfunc = pyfunc

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.pyfunc.predict(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return self.pyfunc.predict(X)


def mlflow_log_scaler(
    scaler: DataScaler,
    name: str,
    input_example: np.ndarray,
    predict_method: str = "transform",
) -> None:
    """Wrapper around the mlflow sklearn model logging functionality. If the scaler is
    not an sklearn estimator, it does not log anything.

    Args:
        scaler: The scaler to log.
        name: The name of the MLFlow model for the scaler.
        predict_method: The method of the scaler to use for prediction (e.g., "inverse_transform").
    """
    mlflow.pyfunc.log_model(
        name=name,
        python_model=_ScalerWrapper(scaler, predict_method=predict_method),
        input_example=input_example,
    )


def mlflow_load_scaler(model_uri: str) -> PyFuncModel:
    loaded_pyfunc = mlflow.pyfunc.load_model(model_uri)
    return _ScalerInverseWrapper(loaded_pyfunc)


def log_general(
    num_model_params: int,
    hyperparameters: dict[str, Any],
    scalers: ScalersDescription,
    log_stream: io.StringIO,
) -> None:
    mlflow.log_param("n_params", num_model_params)
    mlflow.log_params(hyperparameters)

    for name, (scaler, predict_method, input_example) in scalers.items():
        mlflow_log_scaler(
            scaler,
            name=name,
            input_example=input_example,
            predict_method=predict_method,
        )

    mlflow.log_text(log_stream.getvalue(), "training_log.txt")
