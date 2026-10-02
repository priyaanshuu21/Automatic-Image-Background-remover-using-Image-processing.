"""Smoke test for the package skeleton."""

import re

import bgremover

_VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")


def test_version_is_semver() -> None:
    """The package must expose a dotted numeric version string."""
    assert isinstance(bgremover.__version__, str)
    assert _VERSION_RE.match(bgremover.__version__), bgremover.__version__


def test_version_matches_init_file() -> None:
    """The declared version must not be the placeholder ``0.0.0``."""
    assert bgremover.__version__ != "0.0.0"
