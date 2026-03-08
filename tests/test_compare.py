"""Tests for the compare feature and decoupled plotting helpers."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from qmodem.application import (
    TestResult,
    compare,
    populate_crps_ax,
    populate_rul_ax,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_test_result(label: str = "Test Method") -> TestResult:
    """Create a minimal ``TestResult`` for testing."""
    return TestResult(
        method_label=label,
        ts_rul_true=[0.0, 1.0, 2.0, 3.0],
        ruls_true=[3.0, 2.0, 1.0, 0.0],
        ruls_true_lowers=[2.5, 1.5, 0.5, 0.0],
        ruls_true_uppers=[3.5, 2.5, 1.5, 0.5],
        ts_pred=[0.5, 1.5, 2.5],
        pred_means=[2.8, 1.8, 0.8],
        pred_lowers=[2.3, 1.3, 0.3],
        pred_uppers=[3.3, 2.3, 1.3],
        ts_eval=[1.0, 2.0],
        crps_values=[0.1, 0.2],
    )


# ---------------------------------------------------------------------------
# TestResult dataclass
# ---------------------------------------------------------------------------


class TestTestResult:
    """Tests for the ``TestResult`` dataclass."""

    def test_creation(self) -> None:
        result = _make_test_result()
        assert result.method_label == "Test Method"
        assert len(result.ts_pred) == 3
        assert len(result.crps_values) == 2

    def test_fields(self) -> None:
        result = _make_test_result()
        expected_fields = {
            "method_label",
            "ts_rul_true",
            "ruls_true",
            "ruls_true_lowers",
            "ruls_true_uppers",
            "ts_pred",
            "pred_means",
            "pred_lowers",
            "pred_uppers",
            "ts_eval",
            "crps_values",
        }
        actual_fields = {f.name for f in result.__dataclass_fields__.values()}
        assert actual_fields == expected_fields


# ---------------------------------------------------------------------------
# populate_rul_ax
# ---------------------------------------------------------------------------


class TestPopulateRulAx:
    """Tests for ``populate_rul_ax``."""

    def test_draws_lines_and_fills(self) -> None:
        fig, ax = plt.subplots()
        result = _make_test_result()
        populate_rul_ax(
            ax,
            result.ts_rul_true,
            result.ruls_true,
            result.ruls_true_lowers,
            result.ruls_true_uppers,
            result.ts_pred,
            result.pred_means,
            result.pred_lowers,
            result.pred_uppers,
            result.method_label,
        )
        assert len(ax.lines) >= 2
        assert len(ax.collections) >= 2  # fill_between creates PolyCollections
        assert ax.get_ylabel() == "RUL [s]"
        assert ax.get_xlabel() == "Time [s]"
        plt.close(fig)

    def test_legend_contains_labels(self) -> None:
        fig, ax = plt.subplots()
        result = _make_test_result()
        populate_rul_ax(
            ax,
            result.ts_rul_true,
            result.ruls_true,
            result.ruls_true_lowers,
            result.ruls_true_uppers,
            result.ts_pred,
            result.pred_means,
            result.pred_lowers,
            result.pred_uppers,
            result.method_label,
        )
        legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
        assert any("True RUL" in t for t in legend_texts)
        assert any("Test Method" in t for t in legend_texts)
        plt.close(fig)


# ---------------------------------------------------------------------------
# populate_crps_ax
# ---------------------------------------------------------------------------


class TestPopulateCrpsAx:
    """Tests for ``populate_crps_ax``."""

    def test_draws_line(self) -> None:
        fig, ax = plt.subplots()
        populate_crps_ax(ax, [1.0, 2.0], [0.1, 0.2], "My Method")
        assert len(ax.lines) == 1
        assert ax.get_ylabel() == "CRPS [s]"
        plt.close(fig)

    def test_multiple_methods_overlay(self) -> None:
        fig, ax = plt.subplots()
        populate_crps_ax(ax, [1.0, 2.0], [0.1, 0.2], "Method A")
        populate_crps_ax(ax, [1.0, 2.0], [0.3, 0.15], "Method B")
        assert len(ax.lines) == 2
        legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
        assert "Method A" in legend_texts
        assert "Method B" in legend_texts
        plt.close(fig)


# ---------------------------------------------------------------------------
# compare() — integration-level
# ---------------------------------------------------------------------------


class TestCompare:
    """Tests for the ``compare()`` function."""

    def test_rejects_invalid_method(self) -> None:
        with pytest.raises(ValueError, match="Unknown method"):
            compare(methods=["not_a_method"])

    def test_no_results_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """If every method is skipped, a RuntimeError is raised."""
        from qmodem import application

        def _always_fail(**kw: object) -> None:
            raise FileNotFoundError("no checkpoint")

        monkeypatch.setattr(
            application,
            "_PREDICT_DISPATCH",
            {m: _always_fail for m in application.METHODS},
        )
        with pytest.raises(RuntimeError, match="No methods produced results"):
            compare(methods=["het_cnn"])

    def test_compare_produces_correct_subplot_count(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: object,
    ) -> None:
        """Patch _PREDICT_DISPATCH so compare runs without trained models."""
        from qmodem import application

        fake_results = {
            "het_cnn": _make_test_result("Heteroscedastic CNN"),
            "mcd_cnn": _make_test_result("MC Dropout CNN"),
        }
        monkeypatch.setattr(
            application,
            "_PREDICT_DISPATCH",
            {m: lambda m=m, **kw: fake_results[m] for m in fake_results},
        )

        fig = compare(
            methods=["het_cnn", "mcd_cnn"],
            output_dir=str(tmp_path),
        )
        axes = fig.get_axes()
        # 2 RUL subplots + 1 CRPS overlay = 3
        assert len(axes) == 3
        plt.close(fig)

    def test_compare_skips_failing_method_with_warning(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: object,
    ) -> None:
        """Methods whose checkpoints are missing are skipped with a warning."""
        from qmodem import application

        def _fail(**kw: object) -> None:
            raise FileNotFoundError("no checkpoint")

        monkeypatch.setattr(
            application,
            "_PREDICT_DISPATCH",
            {
                "het_cnn": lambda **kw: _make_test_result("Het CNN"),
                "mcd_cnn": _fail,
            },
        )

        with pytest.warns(UserWarning, match="Skipping mcd_cnn"):
            fig = compare(
                methods=["het_cnn", "mcd_cnn"],
                output_dir=str(tmp_path),
            )
        axes = fig.get_axes()
        # Only 1 successful method → 1 RUL + 1 CRPS = 2
        assert len(axes) == 2
        plt.close(fig)
