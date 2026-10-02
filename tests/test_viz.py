"""Tests for headless stage visualisation."""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

from bgremover.viz import (
    checkerboard,
    composite_over_checkerboard,
    save_stage_grid,
)


def test_matplotlib_uses_agg() -> None:
    """Figures never open a window, even in the test session."""
    assert matplotlib.get_backend().lower() == "agg"


def test_checkerboard_pattern() -> None:
    """Squares alternate between 192 and 255 gray."""
    board = checkerboard((16, 16), size=8)
    assert board.shape == (16, 16, 3)
    assert board.dtype == np.uint8
    assert set(np.unique(board).tolist()) == {192, 255}
    assert (board[0, 0] == 255).all()
    assert (board[0, 8] == 192).all()
    assert (board[8, 0] == 192).all()
    assert np.array_equal(board[0, 0], board[7, 7])
    assert np.array_equal(board[:, :, 0], board[:, :, 1])
    with pytest.raises(ValueError):
        checkerboard((0, 8))
    with pytest.raises(ValueError):
        checkerboard((8, 8), size=0)


def test_composite_over_checkerboard() -> None:
    """Opaque pixels cover the board, transparent ones reveal it."""
    rgba = np.zeros((16, 16, 4), dtype=np.uint8)
    rgba[..., 0] = 200
    rgba[4:12, 4:12, 3] = 255
    preview = composite_over_checkerboard(rgba, size=8)
    assert preview.shape == (16, 16, 3)
    assert preview.dtype == np.uint8
    assert (preview[8, 8] == (200, 0, 0)).all()
    assert np.array_equal(preview[0, 0], checkerboard((16, 16))[0, 0])
    with pytest.raises(ValueError):
        composite_over_checkerboard(rgba[..., :3])
    with pytest.raises(ValueError):
        composite_over_checkerboard(rgba.astype(np.float32))


def test_save_stage_grid_writes_a_figure(tmp_path) -> None:
    """Mixed stage kinds render into a non-empty PNG file."""
    stages = {
        "rgb": np.full((24, 32, 3), (10, 200, 30), dtype=np.uint8),
        "gray": np.full((24, 32), 90, dtype=np.uint8),
        "mask": np.zeros((24, 32), dtype=bool),
        "float": np.linspace(0.0, 1.0, 24 * 32).reshape(24, 32),
        "rgba": np.full((24, 32, 4), (200, 30, 30, 128), np.uint8),
    }
    stages["mask"][6:18, 8:24] = True
    destination = tmp_path / "nested" / "grid.png"
    saved = save_stage_grid(destination, stages, columns=3)
    assert saved == destination
    assert destination.is_file()
    assert destination.stat().st_size > 0
    with pytest.raises(ValueError):
        save_stage_grid(tmp_path / "empty.png", {})
    with pytest.raises(ValueError):
        save_stage_grid(tmp_path / "bad.png", stages, columns=0)
    with pytest.raises(ValueError):
        save_stage_grid(
            tmp_path / "bad.png", {"oops": np.zeros((2, 2, 5))}
        )
