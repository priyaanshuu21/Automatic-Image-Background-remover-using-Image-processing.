"""Headless visualisation of pipeline stages.

Everything is rendered with the ``Agg`` backend, so figures are written
straight to disk and no window can ever pop up — the test suite stays
headless.  RGBA stages are composited over a checkerboard, exactly like
image editors preview transparency.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

__all__ = [
    "checkerboard",
    "composite_over_checkerboard",
    "stage_to_rgb",
    "save_stage_grid",
]


def checkerboard(
    shape: tuple[int, int],
    size: int = 8,
) -> np.ndarray:
    """Return a light-gray/white checkerboard preview background.

    Parameters
    ----------
    shape:
        ``(height, width)`` with both values at least 1.
    size:
        Square side length in pixels, at least 1.

    Returns
    -------
    numpy.ndarray
        ``uint8`` RGB array of shape ``(H, W, 3)``.

    Raises
    ------
    ValueError
        For a bad shape or a bad square size.
    """
    height, width = int(shape[0]), int(shape[1])
    cell = int(size)
    if height < 1 or width < 1:
        raise ValueError(f"shape must hold values >= 1, got {shape!r}")
    if cell < 1:
        raise ValueError(f"size must be >= 1, got {size!r}")
    rows = (np.arange(height) // cell)[:, None]
    cols = (np.arange(width) // cell)[None, :]
    light = ((rows + cols) % 2 == 0).astype(np.uint8) * 63 + 192
    return np.repeat(light[:, :, None], 3, axis=2)


def composite_over_checkerboard(
    rgba: np.ndarray,
    size: int = 8,
) -> np.ndarray:
    """Composite an RGBA image over a checkerboard for preview.

    Parameters
    ----------
    rgba:
        ``uint8`` array of shape ``(H, W, 4)`` in RGBA order.
    size:
        Checkerboard square side length in pixels, at least 1.

    Returns
    -------
    numpy.ndarray
        ``uint8`` RGB array of shape ``(H, W, 3)``.

    Raises
    ------
    ValueError
        For a non-RGBA image or a bad square size.
    """
    array = np.asarray(rgba)
    if array.ndim != 3 or array.shape[2] != 4:
        raise ValueError(
            f"expected an RGBA image of shape (H, W, 4), "
            f"got {array.shape}"
        )
    if array.dtype != np.uint8:
        raise ValueError(f"image must be uint8, got {array.dtype}")
    board = checkerboard(array.shape[:2], size).astype(np.float32)
    weight = (array[..., 3:4].astype(np.float32) / 255.0)
    mixed = array[..., :3].astype(np.float32) * weight
    mixed += board * (1.0 - weight)
    return np.clip(np.rint(mixed), 0, 255).astype(np.uint8)


def stage_to_rgb(stage: np.ndarray) -> np.ndarray:
    """Render any pipeline stage as a ``uint8`` RGB preview.

    Boolean masks become black/white, float images are scaled from
    ``[0, 1]``, gray and RGB images pass through and RGBA images are
    composited over a checkerboard.

    Parameters
    ----------
    stage:
        A ``bool``, float, gray, RGB or RGBA array.

    Returns
    -------
    numpy.ndarray
        ``uint8`` RGB array of the same height and width.

    Raises
    ------
    ValueError
        For an unrenderable shape or dtype.
    """
    array = np.asarray(stage)
    if array.ndim == 3 and array.shape[2] == 4 and array.dtype == np.uint8:
        return composite_over_checkerboard(array)
    if array.ndim == 3 and array.shape[2] == 3 and array.dtype == np.uint8:
        return array.copy()
    if array.ndim != 2:
        raise ValueError(
            f"cannot preview array of shape {array.shape} and "
            f"dtype {array.dtype}"
        )
    if array.dtype == np.bool_:
        gray = array.astype(np.uint8) * 255
    elif np.issubdtype(array.dtype, np.floating):
        gray = np.clip(np.rint(array * 255.0), 0, 255).astype(np.uint8)
    elif array.dtype == np.uint8:
        gray = array.copy()
    else:
        raise ValueError(
            f"cannot preview array of shape {array.shape} and "
            f"dtype {array.dtype}"
        )
    return np.repeat(gray[:, :, None], 3, axis=2)


def save_stage_grid(
    path: str | Path,
    stages: dict[str, np.ndarray],
    columns: int = 5,
) -> Path:
    """Save every pipeline stage into a labelled figure grid.

    Parameters
    ----------
    path:
        Destination figure file (``.png`` recommended); parent folders
        are created as needed.
    stages:
        Mapping of stage name to stage array (``bool``, float, gray,
        RGB or RGBA).
    columns:
        Number of grid columns, at least 1.

    Returns
    -------
    pathlib.Path
        The resolved destination path.

    Raises
    ------
    ValueError
        For an empty mapping, a bad column count or an
        unrenderable stage.
    """
    if not stages:
        raise ValueError("stages must not be empty")
    if int(columns) < 1:
        raise ValueError(
            f"columns must be >= 1, got {columns!r}"
        )
    names = list(stages)
    count = len(names)
    rows = (count + int(columns) - 1) // int(columns)
    figure, axes = plt.subplots(rows, int(columns), squeeze=False)
    figure.set_size_inches(int(columns) * 2.4, rows * 2.4)
    for index, axis in enumerate(axes.ravel()):
        axis.axis("off")
        if index >= count:
            continue
        axis.imshow(stage_to_rgb(stages[names[index]]))
        axis.set_title(names[index], fontsize=9)
    figure.tight_layout()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=100)
    plt.close(figure)
    return destination
