"""Permanent architecture guard for the project.

The mini project must stay a *classical* image processing implementation: no
deep-learning or cloud packages, no web frameworks, no interactive calls and
no interactive plots.  These tests walk every module of the package with the
``ast`` and ``tokenize`` modules so a violation cannot be hidden inside a
string, a comment or a nested import.
"""

from __future__ import annotations

import ast
import io
import tokenize
from pathlib import Path

import pytest

PACKAGE_DIR = Path(__file__).resolve().parent.parent / "bgremover"

FORBIDDEN_IMPORT_ROOTS = {
    "rembg",
    "torch",
    "torchvision",
    "tensorflow",
    "keras",
    "onnx",
    "onnxruntime",
    "mediapipe",
    "segment_anything",
    "transformers",
    "detectron2",
    "requests",
    "urllib3",
    "boto3",
    "google",
    "openai",
    "anthropic",
    "flask",
    "fastapi",
    "django",
    "sqlalchemy",
    "sqlite3",
    "docker",
}

FORBIDDEN_TEXT = ("cv2.grabCut", "cv2.dnn", "plt.show(")

TKINTER_ALLOWED_MODULES = {"cli.py", "gui.py", "gui_controller.py",
                           "transforms.py"}


def python_modules() -> list[Path]:
    """Return every Python module inside the package."""
    return sorted(PACKAGE_DIR.rglob("*.py"))


def code_without_comments_and_strings(source: str) -> str:
    """Strip comments and string literals from Python source code.

    Parameters
    ----------
    source:
        Full source of a Python module.

    Returns
    -------
    str
        The same source with comment and string tokens removed, so a plain
        substring search cannot be fooled by documentation.
    """
    kept: list[str] = []
    reader = io.StringIO(source)
    for token in tokenize.generate_tokens(reader.readline):
        if token.type in (
            tokenize.COMMENT,
            tokenize.STRING,
        ):
            continue
        kept.append(token.string)
    return " ".join(kept)


@pytest.mark.parametrize("module", python_modules(),
                         ids=lambda p: p.name)
def test_no_forbidden_imports(module: Path) -> None:
    """No deep-learning, cloud, database or web dependency may be used."""
    tree = ast.parse(module.read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                roots.add(node.module.split(".")[0])
    assert not roots & FORBIDDEN_IMPORT_ROOTS, (
        f"{module.name} imports forbidden roots: "
        f"{sorted(roots & FORBIDDEN_IMPORT_ROOTS)}"
    )


@pytest.mark.parametrize("module", python_modules(),
                         ids=lambda p: p.name)
def test_no_forbidden_library_calls(module: Path) -> None:
    """``cv2.grabCut``, ``cv2.dnn`` and ``plt.show`` are banned."""
    code = code_without_comments_and_strings(
        module.read_text(encoding="utf-8")
    )
    compact = code.replace(" ", "")
    for needle in FORBIDDEN_TEXT:
        assert needle.replace(" ", "") not in compact, (
            f"{module.name} uses forbidden construct {needle!r}"
        )


@pytest.mark.parametrize("module", python_modules(),
                         ids=lambda p: p.name)
def test_no_interactive_input(module: Path) -> None:
    """``input()`` would block the agent; it must never appear."""
    code = code_without_comments_and_strings(
        module.read_text(encoding="utf-8")
    )
    assert "input(" not in code.replace(" ", ""), (
        f"{module.name} calls input()"
    )


@pytest.mark.parametrize("module", python_modules(),
                         ids=lambda p: p.name)
def test_tkinter_is_confined(module: Path) -> None:
    """Only the front-end modules may touch tkinter."""
    tree = ast.parse(module.read_text(encoding="utf-8"))
    imports_tk = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name.split(".")[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [(node.module or "").split(".")[0]]
        else:
            continue
        if "tkinter" in names:
            imports_tk = True
    if imports_tk:
        assert module.name in TKINTER_ALLOWED_MODULES, (
            f"{module.name} imports tkinter, which is only allowed in "
            f"{sorted(TKINTER_ALLOWED_MODULES)}"
        )


def test_guard_finds_a_planted_violation(tmp_path: Path) -> None:
    """Sanity check: the detector really sees a forbidden import."""
    module = tmp_path / "bad.py"
    module.write_text("import torch\n", encoding="utf-8")
    tree = ast.parse(module.read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
    assert "torch" in roots & FORBIDDEN_IMPORT_ROOTS
