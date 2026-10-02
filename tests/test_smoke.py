"""Smoke test for the package skeleton."""

import bgremover


def test_version_is_set() -> None:
    """The package must expose the initial version string."""
    assert bgremover.__version__ == "0.0.1"
