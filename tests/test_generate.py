"""Tests for qmodem.generate and BatterySimulationTimeWindowSource.from_file."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from qmodem.generate import generate_test_data, generate_train_data

# ---------------------------------------------------------------------------
# Helpers — mock _run_discharge and _run_stochastic_sims
# ---------------------------------------------------------------------------


def _mock_run_discharge(_config, soc_0):
    """Deterministic stub: returns a voltage array of length 20 and t_eod based on soc_0."""
    n_t = 20
    voltage = np.linspace(4.2, 3.0, n_t)
    t_eod = 5.0 * soc_0
    return voltage, t_eod


def _mock_run_stochastic_sims(_config, soc_0, n_simu):
    """Deterministic stub: returns t_eods of shape (n_simu,)."""
    rng = np.random.default_rng(42)
    return rng.normal(loc=5.0 * soc_0, scale=0.1, size=n_simu)


# ---------------------------------------------------------------------------
# Tests for generate_train_data
# ---------------------------------------------------------------------------


class TestGenerateTrainData:
    @pytest.fixture
    def train_npz(self, tmp_path: Path) -> Path:
        """Generate a train data file in a temp directory."""
        with patch("qmodem.generate._run_discharge", side_effect=_mock_run_discharge):
            path = generate_train_data(
                simulator_config={"dt": 20.0},
                n_histories=5,
                soc_range=(0.2, 0.9),
                seed=42,
                output_path=tmp_path / "train.npz",
            )
        return path

    def test_returns_path(self, train_npz: Path) -> None:
        assert train_npz.exists()
        assert train_npz.suffix == ".npz"

    def test_npz_keys(self, train_npz: Path) -> None:
        data = np.load(train_npz, allow_pickle=True)
        assert set(data.files) >= {"voltages", "t_eods", "soc_0s", "config_json"}

    def test_n_histories(self, train_npz: Path) -> None:
        data = np.load(train_npz, allow_pickle=True)
        assert len(data["voltages"]) == 5
        assert len(data["t_eods"]) == 5
        assert len(data["soc_0s"]) == 5

    def test_soc_range(self, train_npz: Path) -> None:
        data = np.load(train_npz, allow_pickle=True)
        soc_0s = data["soc_0s"]
        assert all(0.2 <= s <= 0.9 for s in soc_0s)

    def test_config_json_is_valid(self, train_npz: Path) -> None:
        data = np.load(train_npz, allow_pickle=True)
        config = json.loads(str(data["config_json"]))
        assert isinstance(config, dict)

    def test_reproducible(self, tmp_path: Path) -> None:
        """Same seed produces identical data."""
        with patch("qmodem.generate._run_discharge", side_effect=_mock_run_discharge):
            p1 = generate_train_data(
                simulator_config={},
                n_histories=3,
                seed=99,
                output_path=tmp_path / "a.npz",
            )
            p2 = generate_train_data(
                simulator_config={},
                n_histories=3,
                seed=99,
                output_path=tmp_path / "b.npz",
            )
        d1 = np.load(p1, allow_pickle=True)
        d2 = np.load(p2, allow_pickle=True)
        np.testing.assert_array_equal(d1["soc_0s"], d2["soc_0s"])
        np.testing.assert_array_equal(d1["t_eods"], d2["t_eods"])


# ---------------------------------------------------------------------------
# Tests for generate_test_data
# ---------------------------------------------------------------------------


class TestGenerateTestData:
    @pytest.fixture
    def test_npz_paths(self, tmp_path: Path) -> list[Path]:
        """Generate test case files in a temp directory."""
        with (
            patch("qmodem.generate.les.SimulatorSimple") as MockSim,
            patch(
                "qmodem.generate._run_stochastic_sims",
                side_effect=_mock_run_stochastic_sims,
            ),
        ):
            # Configure the mock simulator for the single trajectory
            mock_instance = MockSim.return_value
            n_t = 30
            mock_instance.v_memo = np.linspace(4.2, 3.0, n_t).reshape(n_t, 1)
            mock_instance.soc_memo = np.linspace(1.0, 0.1, n_t).reshape(n_t, 1)
            mock_instance.t_eods = [5.0]

            paths = generate_test_data(
                simulator_config={"dt": 20.0},
                n_test_cases=2,
                n_simu=10,
                n_intermediate_socs=5,
                seed=123,
                output_dir=tmp_path,
            )
        return paths

    def test_returns_correct_count(self, test_npz_paths: list[Path]) -> None:
        assert len(test_npz_paths) == 2

    def test_files_exist(self, test_npz_paths: list[Path]) -> None:
        for p in test_npz_paths:
            assert p.exists()

    def test_npz_keys(self, test_npz_paths: list[Path]) -> None:
        data = np.load(test_npz_paths[0])
        expected_keys = {
            "voltage",
            "soc_history",
            "t_eod",
            "dt",
            "eval_indices",
            "ref_t_eods",
            "config_json",
        }
        assert set(data.files) >= expected_keys

    def test_ref_t_eods_shape(self, test_npz_paths: list[Path]) -> None:
        data = np.load(test_npz_paths[0])
        assert data["ref_t_eods"].shape == (5, 10)  # (n_intermediate_socs, n_simu)

    def test_eval_indices_shape(self, test_npz_paths: list[Path]) -> None:
        data = np.load(test_npz_paths[0])
        assert data["eval_indices"].shape == (5,)
