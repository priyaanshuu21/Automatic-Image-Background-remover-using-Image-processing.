"""Alpha mattes and RGBA export.

A binary foreground mask becomes a soft alpha matte that is only feathered
in a narrow band around the boundary, colour halos are pushed out by
propagating the nearest opaque colours, and the result is composited and
resized back to the original resolution for export.
"""

from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from bgremover.filters import correlate2d, gaussian_kernel
from bgremover.morphology import dilate, erode, structuring_element

__all__ = [
    "alpha_from_mask",
    "decontaminate_edges",
    "compose_rgba",
    "upscale_mask",
]


def _as_bool_mask(mask: np.ndarray) -> np.ndarray:
    """Validate a 2-D boolean (or 0/1) mask."""
    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError(
            f"expected a 2-D mask, got shape {array.shape}"
        )
    if array.size == 0:
        raise ValueError("mask must not be empty")
    if array.dtype == np.bool_:
        return array.copy()
    unique = np.unique(array)
    if not set(unique.tolist()) <= {0, 1}:
        raise ValueError("mask must be boolean or hold only 0 and 1")
    return array.astype(bool)


def _as_rgb(image: np.ndarray) -> np.ndarray:
    """Validate a ``uint8`` RGB image."""
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(
            f"expected an RGB image of shape (H, W, 3), got {array.shape}"
        )
    if array.dtype != np.uint8:
        raise ValueError(f"image must be uint8, got {array.dtype}")
    return array


def _as_alpha(alpha: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Validate a float alpha matte of the given shape."""
    array = np.asarray(alpha, dtype=np.float32)
    if array.shape != shape:
        raise ValueError(
            f"alpha shape {array.shape} != image shape {shape}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError("alpha must be finite")
    return np.clip(array, 0.0, 1.0).astype(np.float32)


def alpha_from_mask(
    mask: np.ndarray,
    feather_sigma: float = 1.5,
    band_width: int = 4,
) -> np.ndarray:
    """Turn a binary foreground mask into a soft alpha matte.

    The mask is blurred with a Gaussian and the blur is kept only in a
    morphological band of ``band_width`` pixels around the boundary, so
    solid foreground stays fully opaque and solid background fully
    transparent while the cut-out edge turns soft.

    Parameters
    ----------
    mask:
        Boolean (or 0/1) array of shape ``(H, W)``.
    feather_sigma:
        Gaussian sigma in pixels; ``0`` disables feathering.
    band_width:
        Half-width of the feathered band in pixels; ``0`` disables
        feathering.

    Returns
    -------
    numpy.ndarray
        ``float32`` array of shape ``(H, W)`` in ``[0, 1]``.

    Raises
    ------
    ValueError
        For a bad mask, a negative sigma or a negative band width.
    """
    image = _as_bool_mask(mask)
    sigma = float(feather_sigma)
    if sigma < 0.0:
        raise ValueError(
            f"feather_sigma must be >= 0, got {feather_sigma!r}"
        )
    width = int(band_width)
    if width < 0:
        raise ValueError(
            f"band_width must be >= 0, got {band_width!r}"
        )
    hard = image.astype(np.float32)
    if sigma <= 0.0 or width <= 0:
        return hard
    side = max(2 * int(np.ceil(3.0 * sigma)) + 1, 3)
    blurred = correlate2d(hard, gaussian_kernel(side, sigma))
    blurred = np.clip(blurred, 0.0, 1.0).astype(np.float32)
    element = structuring_element("square", width)
    band = dilate(image, kernel=element) & ~erode(image, kernel=element)
    return np.where(band, blurred, hard).astype(np.float32)


def decontaminate_edges(
    rgb: np.ndarray,
    alpha: np.ndarray,
) -> np.ndarray:
    """Push background halos out of semi-transparent edge pixels.

    Every pixel with ``0 < alpha < 1`` takes the colour of the nearest
    fully opaque pixel, found with a multi-source breadth-first search
    over the 4-neighbourhood, so the exported fringe shows foreground
    colours instead of a background halo.

    Parameters
    ----------
    rgb:
        ``uint8`` RGB array of shape ``(H, W, 3)``.
    alpha:
        ``float32`` alpha matte of shape ``(H, W)`` in ``[0, 1]``.

    Returns
    -------
    numpy.ndarray
        New ``uint8`` RGB array; the input is not modified.

    Raises
    ------
    ValueError
        For a bad image or a mismatched alpha matte.
    """
    image = _as_rgb(rgb)
    matte = _as_alpha(alpha, image.shape[:2])
    opaque = matte >= 1.0
    semi = (matte > 0.0) & ~opaque
    if not semi.any() or not opaque.any():
        return image.copy()
    height, width = matte.shape
    visited = opaque.copy()
    result = image.copy()
    queue: deque[tuple[int, int]] = deque(
        map(tuple, np.argwhere(opaque).tolist())  # type: ignore[arg-type]
    )
    while queue:
        row, col = queue.popleft()
        for delta_row, delta_col in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            next_row, next_col = row + delta_row, col + delta_col
            if not 0 <= next_row < height or not 0 <= next_col < width:
                continue
            if visited[next_row, next_col] or not semi[next_row, next_col]:
                continue
            visited[next_row, next_col] = True
            result[next_row, next_col] = result[row, col]
            queue.append((next_row, next_col))
    return result


def compose_rgba(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Composite an RGB image with an alpha matte into RGBA.

    Parameters
    ----------
    rgb:
        ``uint8`` RGB array of shape ``(H, W, 3)``.
    alpha:
        Float array of shape ``(H, W)`` in ``[0, 1]``.

    Returns
    -------
    numpy.ndarray
        ``uint8`` array of shape ``(H, W, 4)`` in RGBA order.

    Raises
    ------
    ValueError
        For a bad image or a mismatched alpha matte.
    """
    image = _as_rgb(rgb)
    matte = _as_alpha(alpha, image.shape[:2])
    flat = np.rint(matte * 255.0).astype(np.uint8)
    return np.dstack([image, flat])


def upscale_mask(
    mask: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    """Resize a boolean mask or float matte to ``(height, width)``.

    Boolean masks use nearest-neighbour interpolation and stay boolean;
    float mattes use bilinear interpolation and stay in ``[0, 1]``.
    ``cv2.resize`` is used on purpose: it is the one sanctioned OpenCV
    call of the project (proper area averaging when minifying).

    Parameters
    ----------
    mask:
        Boolean array or float array of shape ``(H, W)``.
    shape:
        Target ``(height, width)``; both values at least 1.

    Returns
    -------
    numpy.ndarray
        Resized array of shape ``shape`` (a copy when already there).

    Raises
    ------
    ValueError
        For a bad mask, a bad shape or an unsupported dtype.
    """
    height, width = int(shape[0]), int(shape[1])
    if height < 1 or width < 1:
        raise ValueError(f"shape must hold values >= 1, got {shape!r}")
    array = np.asarray(mask)
    if array.ndim != 2 or array.size == 0:
        raise ValueError(
            f"expected a non-empty 2-D mask, got shape {array.shape}"
        )
    if array.shape == (height, width):
        return array.copy()
    if array.dtype == np.bool_:
        resized = cv2.resize(
            array.astype(np.uint8),
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )
        return resized.astype(bool)
    if np.issubdtype(array.dtype, np.floating):
        resized = cv2.resize(
            array.astype(np.float32),
            (width, height),
            interpolation=cv2.INTER_LINEAR,
        )
        return np.clip(resized, 0.0, 1.0).astype(np.float32)
    raise ValueError(
        f"mask must be boolean or floating, got {array.dtype}"
    )
