"""Tests for the GUI controller and the Tk view (never opens windows)."""

from __future__ import annotations

import tkinter as tk

import numpy as np
import pytest

import bgremover.gui as gui_module
from bgremover.gui_controller import GuiController
from bgremover.io import save_image
from bgremover.pipeline import STAGE_KEYS


def _saved(rect_image, tmp_path, name="gui.png"):
    """Write a fixture image and return its path as a string."""
    path = tmp_path / name
    save_image(path, rect_image)
    return str(path)


def test_gui_module_creates_no_windows() -> None:
    """Importing the view leaves the Tk interpreter untouched."""
    assert callable(gui_module.launch)
    assert hasattr(gui_module, "BackgroundRemoverApp")
    assert tk._default_root is None


def test_controller_load_and_summary(rect_image, tmp_path) -> None:
    """Loading sets the image, the source and a summary line."""
    controller = GuiController()
    assert not controller.has_image()
    assert controller.summary() == "no image loaded"
    with pytest.raises(RuntimeError):
        controller.run()
    path = _saved(rect_image, tmp_path)
    image = controller.load(path)
    assert controller.has_image()
    assert not controller.has_result()
    assert image.shape == rect_image.shape
    summary = controller.summary()
    assert "128x128" in summary and "not processed" in summary
    with pytest.raises(FileNotFoundError):
        controller.load(str(tmp_path / "missing.png"))


def test_controller_run_and_previews(rect_image, tmp_path) -> None:
    """Running fills every preview with correctly shaped output."""
    controller = GuiController()
    controller.load(_saved(rect_image, tmp_path))
    result = controller.run()
    assert controller.has_result()
    assert set(result.stages) == set(STAGE_KEYS)
    preview = controller.preview_rgba()
    assert preview.shape == rect_image.shape
    assert preview.dtype == np.uint8
    edges = controller.preview_stage("edges")
    assert edges.shape == rect_image.shape[:2] + (3,)
    assert edges.dtype == np.uint8
    with pytest.raises(KeyError):
        controller.preview_stage("nope")
    summary = controller.summary()
    assert "foreground" in summary


def test_controller_save_and_reset(rect_image, tmp_path) -> None:
    """Results save to disk and reset clears the whole session."""
    controller = GuiController()
    with pytest.raises(RuntimeError):
        controller.preview_rgba()
    with pytest.raises(RuntimeError):
        controller.save_rgba(str(tmp_path / "out.png"))
    controller.load(_saved(rect_image, tmp_path))
    controller.run()
    target = tmp_path / "result.png"
    controller.save_rgba(str(target))
    assert target.is_file() and target.stat().st_size > 0
    controller.reset()
    assert not controller.has_image()
    assert not controller.has_result()
    assert controller.summary() == "no image loaded"
