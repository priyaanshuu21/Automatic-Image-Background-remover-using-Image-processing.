"""Deterministic synthetic test fixtures.

No file is downloaded and no binary image is stored in git: every fixture is
generated from closed-form formulas with a fixed ``np.random.default_rng(42)``
seed, so the whole suite is reproducible on any machine.
"""

from __future__ import annotations

import numpy as np
import pytest

RADIUS = 35
_CIRCLE_SHAPE = (120, 160)
_RECT_SHAPE = (128, 128)
_RECT_SLICE = (slice(34, 94), slice(44, 92))
# Background gradient endpoints: smooth blue (top-left) to teal (bottom-right).
_GRADIENT_START = np.array([30.0, 100.0, 200.0], dtype=np.float32)
_GRADIENT_END = np.array([45.0, 135.0, 195.0], dtype=np.float32)


def make_circle_on_gradient() -> tuple[np.ndarray, np.ndarray]:
    """Return a red disc on a smooth blue-to-teal gradient.

    Returns
    -------
    tuple
        ``(image, mask)`` where ``image`` is ``uint8`` of shape
        ``(120, 160, 3)`` in RGB order and ``mask`` is a ``bool`` array of
        shape ``(120, 160)`` that is true inside the disc of radius
        :data:`RADIUS` centred in the image.
    """
    height, width = _CIRCLE_SHAPE
    rows = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    cols = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
    blend = np.clip(0.5 * (rows + cols), 0.0, 1.0)[..., None]
    background = _GRADIENT_START + (_GRADIENT_END - _GRADIENT_START) * blend
    image = np.repeat(background, 1, axis=2).astype(np.float32)

    grid_y = np.arange(height, dtype=np.float32)[:, None]
    grid_x = np.arange(width, dtype=np.float32)[None, :]
    centre_y = (height - 1) / 2.0
    centre_x = (width - 1) / 2.0
    squared = (grid_y - centre_y) ** 2 + (grid_x - centre_x) ** 2
    mask = squared <= float(RADIUS) ** 2

    image = np.clip(image, 0.0, 255.0)
    image[mask] = np.array([220.0, 35.0, 35.0], dtype=np.float32)
    return image.astype(np.uint8), mask


def make_rect_on_flat() -> tuple[np.ndarray, np.ndarray]:
    """Return a green rectangle on a flat light-gray background.

    Returns
    -------
    tuple
        ``(image, mask)`` with a ``uint8`` ``(128, 128, 3)`` image and the
        ``bool`` ``(128, 128)`` rectangle mask.
    """
    height, width = _RECT_SHAPE
    image = np.full((height, width, 3), 225, dtype=np.uint8)
    mask = np.zeros((height, width), dtype=bool)
    mask[_RECT_SLICE] = True
    image[_RECT_SLICE] = np.array([30, 160, 60], dtype=np.uint8)
    return image, mask


def noisy_variant(image: np.ndarray, sigma: float = 8.0) -> np.ndarray:
    """Add reproducible Gaussian noise and clip the result to ``uint8``.

    Parameters
    ----------
    image:
        ``uint8`` image of any supported shape.
    sigma:
        Standard deviation of the additive noise, in gray levels.

    Returns
    -------
    numpy.ndarray
        A new ``uint8`` image; the input is not modified.
    """
    array = np.asarray(image)
    rng = np.random.default_rng(42)
    noise = rng.normal(0.0, float(sigma), array.shape).astype(np.float32)
    noisy = np.clip(array.astype(np.float32) + noise, 0.0, 255.0)
    return np.rint(noisy).astype(np.uint8)


def ground_truth_mask(shape: tuple[int, int], radius: float) -> np.ndarray:
    """Return a boolean disc mask of the given shape and radius.

    Parameters
    ----------
    shape:
        ``(height, width)`` of the returned mask.
    radius:
        Radius of the disc, in pixels.

    Returns
    -------
    numpy.ndarray
        ``bool`` array of shape ``shape``.
    """
    height, width = shape
    grid_y = np.arange(height, dtype=np.float32)[:, None]
    grid_x = np.arange(width, dtype=np.float32)[None, :]
    centre_y = (height - 1) / 2.0
    centre_x = (width - 1) / 2.0
    squared = (grid_y - centre_y) ** 2 + (grid_x - centre_x) ** 2
    return squared <= float(radius) ** 2


def iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Return the intersection over union of two boolean masks.

    Parameters
    ----------
    mask_a, mask_b:
        Boolean arrays of identical shape.

    Returns
    -------
    float
        ``|A & B| / |A | B|``; ``1.0`` when both masks are empty and ``0.0``
        when exactly one of them is empty.
    """
    first = np.asarray(mask_a, dtype=bool)
    second = np.asarray(mask_b, dtype=bool)
    union = int(np.count_nonzero(first | second))
    if union == 0:
        return 1.0
    intersection = int(np.count_nonzero(first & second))
    return intersection / union


@pytest.fixture
def circle_on_gradient() -> tuple[np.ndarray, np.ndarray]:
    """Red disc on a blue-to-teal gradient with its ground-truth mask."""
    return make_circle_on_gradient()


@pytest.fixture
def circle_image(circle_on_gradient: tuple[np.ndarray, np.ndarray]
                 ) -> np.ndarray:
    """The image part of the :func:`circle_on_gradient` fixture."""
    image, _ = circle_on_gradient
    return image


@pytest.fixture
def circle_mask(circle_on_gradient: tuple[np.ndarray, np.ndarray]
               ) -> np.ndarray:
    """The ground-truth mask part of the circle fixture."""
    _, mask = circle_on_gradient
    return mask


@pytest.fixture
def rect_on_flat() -> tuple[np.ndarray, np.ndarray]:
    """Green rectangle on a flat gray background with its ground truth."""
    return make_rect_on_flat()


@pytest.fixture
def rect_image(rect_on_flat: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """The image part of the :func:`rect_on_flat` fixture."""
    image, _ = rect_on_flat
    return image


@pytest.fixture
def rect_mask(rect_on_flat: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """The ground-truth mask part of the rectangle fixture."""
    _, mask = rect_on_flat
    return mask


@pytest.fixture
def noisy_circle(circle_image: np.ndarray) -> np.ndarray:
    """The circle fixture degraded with reproducible Gaussian noise."""
    return noisy_variant(circle_image, 8.0)


@pytest.fixture
def random_rgb() -> np.ndarray:
    """A deterministic random ``uint8`` RGB image of shape (64, 80, 3)."""
    rng = np.random.default_rng(7)
    return rng.integers(0, 256, size=(64, 80, 3), dtype=np.uint8)
