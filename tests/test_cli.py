"""Smoke tests for the qmodem CLI entry points."""

from __future__ import annotations

from click.testing import CliRunner

from qmodem.cli import cli


class TestCLIHelp:
    """Verify that all CLI commands respond to ``--help``."""

    def test_root_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "QMoDeM" in result.output

    def test_train_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["train", "--help"])
        assert result.exit_code == 0
        assert "METHOD" in result.output
        assert "bayes_cnn" in result.output

    def test_test_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["test", "--help"])
        assert result.exit_code == 0
        assert "bayes_cnn" in result.output

    def test_generate_data_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["generate-data", "--help"])
        assert result.exit_code == 0
        assert "n-histories-train" in result.output

    def test_compare_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["compare", "--help"])
        assert result.exit_code == 0
        assert "--methods" in result.output
        assert "--n-samples" in result.output


class TestCLIValidation:
    """Verify that invalid inputs are rejected."""

    def test_train_invalid_method(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["train", "invalid_method"])
        assert result.exit_code != 0
        assert "Invalid value" in result.output or "invalid_method" in result.output

    def test_test_invalid_method(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["test", "invalid_method"])
        assert result.exit_code != 0

    def test_train_requires_method(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["train"])
        assert result.exit_code != 0
