"""Tests for the installed qmodem distribution metadata."""

import importlib.metadata


def test_qmodem_distribution_version_is_the_release_version() -> None:
    """The installed qmodem distribution reports the 0.2.1 release version."""
    assert importlib.metadata.version("qmodem") == "0.2.1"
