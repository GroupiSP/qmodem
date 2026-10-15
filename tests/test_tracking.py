"""Tests for qmodem.tracking."""

from __future__ import annotations

import importlib.metadata

from qmodem.tracking import stagereg_version


def test_stagereg_version_is_the_installed_distribution_version() -> None:
    assert stagereg_version() == importlib.metadata.version("stagereg") == "0.2.0"
