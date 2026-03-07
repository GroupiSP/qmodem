"""Tests for qmodem.metadata — save_metadata / load_metadata persistence."""

from __future__ import annotations

from pathlib import Path

import pytest

from qmodem.metadata import (
    BaseModelParams,
    MCDModelParams,
    PQCParams,
    QAVITrainingMetadata,
    QAVITrainingParams,
    ScalingParams,
    SimulatorConfig,
    TrainingMetadata,
    TrainingParams,
    load_metadata,
    save_metadata,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

SIM_CONFIG: SimulatorConfig = {
    "n_simu": 1,
    "v_cut": 3.0,
    "soc_0": 1.0,
    "dt": 1.0,
    "omega_std": 0.01,
    "eta_std": 0.01,
}

BASE_TRAINING_PARAMS: TrainingParams = {
    "window_size": 50,
    "stride": 5,
    "n_histories_train": 100,
    "n_histories_val": 20,
    "soc_range": [0.05, 1.0],
}

BASE_MODEL_PARAMS: BaseModelParams = {
    "n_filters": 16,
    "kernel_size": 3,
}

SCALING_PARAMS: ScalingParams = {
    "normalize": True,
    "y_max": 500.0,
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSaveLoadMetadata:
    """Tests for save_metadata / load_metadata JSON round-trip."""

    def test_het_cnn_metadata(self, tmp_path: Path) -> None:
        """het_cnn / bayes_cnn style: base fields only."""
        metadata: TrainingMetadata = {
            "method": "het_cnn",
            "simulator_config": SIM_CONFIG,
            "training_params": BASE_TRAINING_PARAMS,
            "model_params": BASE_MODEL_PARAMS,
            "scaling_params": SCALING_PARAMS,
        }
        save_metadata(tmp_path, metadata)
        loaded = load_metadata(tmp_path)

        assert loaded == metadata

    def test_bayes_cnn_metadata(self, tmp_path: Path) -> None:
        """bayes_cnn style: same structure as het_cnn."""
        metadata: TrainingMetadata = {
            "method": "bayes_cnn",
            "simulator_config": SIM_CONFIG,
            "training_params": BASE_TRAINING_PARAMS,
            "model_params": BASE_MODEL_PARAMS,
            "scaling_params": SCALING_PARAMS,
        }
        save_metadata(tmp_path, metadata)
        loaded = load_metadata(tmp_path)

        assert loaded == metadata

    def test_mcd_cnn_metadata(self, tmp_path: Path) -> None:
        """mcd_cnn style: model_params includes dropout_rate."""
        mcd_model_params: MCDModelParams = {
            "n_filters": 32,
            "kernel_size": 5,
            "dropout_rate": 0.2,
        }
        metadata: TrainingMetadata = {
            "method": "mcd_cnn",
            "simulator_config": SIM_CONFIG,
            "training_params": BASE_TRAINING_PARAMS,
            "model_params": mcd_model_params,
            "scaling_params": SCALING_PARAMS,
        }
        save_metadata(tmp_path, metadata)
        loaded = load_metadata(tmp_path)

        assert loaded == metadata
        assert loaded["model_params"]["dropout_rate"] == pytest.approx(0.2)

    def test_qavi_cnn_metadata(self, tmp_path: Path) -> None:
        """qavi_cnn style: training_params includes batch_w and pqc_params is present."""
        qavi_training_params: QAVITrainingParams = {
            "window_size": 50,
            "stride": 5,
            "n_histories_train": 100,
            "n_histories_val": 20,
            "soc_range": [0.05, 1.0],
            "batch_w": 10,
        }
        pqc_params: PQCParams = {
            "n_qubits": 4,
            "n_pqc_layers": 2,
        }
        metadata: QAVITrainingMetadata = {
            "method": "qavi_cnn",
            "simulator_config": SIM_CONFIG,
            "training_params": qavi_training_params,
            "model_params": BASE_MODEL_PARAMS,
            "scaling_params": SCALING_PARAMS,
            "pqc_params": pqc_params,
        }
        save_metadata(tmp_path, metadata)
        loaded = load_metadata(tmp_path)

        assert loaded == metadata
        assert loaded["training_params"]["batch_w"] == 10
        assert loaded["pqc_params"]["n_qubits"] == 4
        assert loaded["pqc_params"]["n_pqc_layers"] == 2

    def test_metadata_file_is_json(self, tmp_path: Path) -> None:
        """The saved file is valid JSON and named metadata.json."""
        import json

        metadata: TrainingMetadata = {
            "method": "het_cnn",
            "simulator_config": SIM_CONFIG,
            "training_params": BASE_TRAINING_PARAMS,
            "model_params": BASE_MODEL_PARAMS,
            "scaling_params": SCALING_PARAMS,
        }
        save_metadata(tmp_path, metadata)

        json_file = tmp_path / "metadata.json"
        assert json_file.exists()
        parsed = json.loads(json_file.read_text(encoding="utf-8"))
        assert parsed["method"] == "het_cnn"

    def test_normalize_false_y_max_is_one(self, tmp_path: Path) -> None:
        """When normalize=False, y_max is stored as 1.0 and restored correctly."""
        scaling: ScalingParams = {"normalize": False, "y_max": 1.0}
        metadata: TrainingMetadata = {
            "method": "het_cnn",
            "simulator_config": SIM_CONFIG,
            "training_params": BASE_TRAINING_PARAMS,
            "model_params": BASE_MODEL_PARAMS,
            "scaling_params": scaling,
        }
        save_metadata(tmp_path, metadata)
        loaded = load_metadata(tmp_path)

        assert loaded["scaling_params"]["normalize"] is False
        assert loaded["scaling_params"]["y_max"] == pytest.approx(1.0)
