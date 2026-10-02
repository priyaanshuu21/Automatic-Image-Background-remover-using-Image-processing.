"""Neighbourhood processing (spatial filtering).

The operators are written from scratch on top of
``numpy.lib.stride_tricks.sliding_window_view``: no Python loop ever touches
an individual pixel, so a 3x3 filter over a 1920x1080 image is a handful of
vectorised operations.

Image conventions: gray input ``uint8`` of shape ``(H, W)``, colour input
``uint8`` of shape ``(H, W, C)``; every public function returns ``uint8``
unless its name ends in ``_float``.
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from bgremover.config import PipelineConfig

LOGGER = logging.getLogger(__name__)

BORDER_MODES = ("reflect", "edge", "zero")
_NP_PAD_MODE = {"reflect": "reflect", "edge": "edge", "zero": "constant"}
MAX_MEDIAN_KSIZE = 15

LAPLACIAN_4 = np.array(
    [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
    dtype=np.float32,
)
LAPLACIAN_8 = np.array(
    [[1.0, 1.0, 1.0], [1.0, -8.0, 1.0], [1.0, 1.0, 1.0]],
    dtype=np.float32,
)


def _as_filterable(image: np.ndarray) -> np.ndarray:
    """Validate a 2-D or 3-D numeric image."""
    array = np.asarray(image)
    if array.ndim not in (2, 3):
        raise ValueError(
            f"image must have 2 or 3 dimensions, got shape {array.shape}"
        )
    if array.size == 0:
        raise ValueError("image must not be empty")
    return array


def _as_kernel(kernel: np.ndarray) -> np.ndarray:
    """Validate a 2-D kernel with odd side lengths."""
    array = np.asarray(kernel, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(
            f"kernel must be 2-D, got shape {array.shape}"
        )
    if array.shape[0] % 2 == 0 or array.shape[1] % 2 == 0:
        raise ValueError(
            f"kernel sides must be odd, got shape {array.shape}"
        )
    return array


def _check_border(border: str) -> str:
    """Validate the border mode name."""
    if border not in BORDER_MODES:
        raise ValueError(
            f"border must be one of {BORDER_MODES}, got {border!r}"
        )
    return border


def pad_image(
    image: np.ndarray,
    pad: int | tuple[int, int],
    mode: str = "reflect",
) -> np.ndarray:
    """Pad an image on all sides.

    Parameters
    ----------
    image:
        Array of shape ``(H, W)`` or ``(H, W, C)``.
    pad:
        Number of rows and columns to add, either a single integer or a
        ``(rows, columns)`` pair.
    mode:
        ``"reflect"`` (mirror without repeating the border sample),
        ``"edge"`` (replicate the border sample) or ``"zero"``.

    Returns
    -------
    numpy.ndarray
        Padded array of the same dtype.

    Raises
    ------
    ValueError
        For an unknown mode, a negative pad or a bad array rank.
    """
    array = _as_filterable(image)
    _check_border(mode)
    if isinstance(pad, (tuple, list)):
        rows, cols = int(pad[0]), int(pad[1])
    else:
        rows = cols = int(pad)
    if rows < 0 or cols < 0:
        raise ValueError(f"pad must be non-negative, got {pad!r}")
    pad_width = [(rows, rows), (cols, cols)]
    while len(pad_width) < array.ndim:
        pad_width.append((0, 0))
    if mode == "zero":
        return np.pad(array, pad_width, mode="constant", constant_values=0)
    return np.pad(array, pad_width, mode=_NP_PAD_MODE[mode])


def correlate2d(
    image: np.ndarray,
    kernel: np.ndarray,
    border: str = "reflect",
) -> np.ndarray:
    """Correlate an image with a 2-D kernel (no kernel flip).

    The result at ``(y, x)`` is the sum of ``kernel * image`` over the window
    centred on ``(y, x)``.  Colour images are filtered channel by channel.

    Parameters
    ----------
    image:
        Array of shape ``(H, W)`` or ``(H, W, C)``.
    kernel:
        2-D kernel with odd side lengths.
    border:
        Border handling, see :func:`pad_image`.

    Returns
    -------
    numpy.ndarray
        ``float32`` array of the same shape as the input.

    Raises
    ------
    ValueError
        For a bad kernel, border mode or array rank.
    """
    array = _as_filterable(image)
    weights = _as_kernel(kernel)
    _check_border(border)
    radius_y = weights.shape[0] // 2
    radius_x = weights.shape[1] // 2
    padded = pad_image(array, (radius_y, radius_x), border)
    windows = sliding_window_view(
        padded, (weights.shape[0], weights.shape[1]), axis=(0, 1)
    )
    return np.tensordot(windows, weights, axes=([-2, -1], [0, 1]))


def convolve2d(
    image: np.ndarray,
    kernel: np.ndarray,
    border: str = "reflect",
) -> np.ndarray:
    """Convolve an image with a 2-D kernel (true convolution).

    Convolution is correlation with the kernel flipped in both directions,
    ``kernel[::-1, ::-1]``; for a symmetric kernel the two are identical.

    Parameters
    ----------
    image:
        Array of shape ``(H, W)`` or ``(H, W, C)``.
    kernel:
        2-D kernel with odd side lengths.
    border:
        Border handling, see :func:`pad_image`.

    Returns
    -------
    numpy.ndarray
        ``float32`` array of the same shape as the input.

    Raises
    ------
    ValueError
        For a bad kernel, border mode or array rank.
    """
    weights = _as_kernel(kernel)
    return correlate2d(image, weights[::-1, ::-1], border)


def _filter_1d(
    image: np.ndarray,
    kernel_1d: np.ndarray,
    axis: int,
    border: str = "reflect",
) -> np.ndarray:
    """Correlate along a single axis with a 1-D kernel."""
    array = _as_filterable(image)
    _check_border(border)
    weights = np.asarray(kernel_1d, dtype=np.float32)
    if weights.ndim != 1 or weights.size % 2 == 0:
        raise ValueError(
            f"1-D kernel must have an odd length, got shape {weights.shape}"
        )
    radius = weights.size // 2
    pad_width = [(0, 0)] * array.ndim
    pad_width[axis] = (radius, radius)
    if border == "zero":
        padded = np.pad(
            array, pad_width, mode="constant", constant_values=0
        )
    else:
        padded = np.pad(array, pad_width, mode=_NP_PAD_MODE[border])
    windows = sliding_window_view(padded, weights.size, axis=axis)
    return np.tensordot(windows, weights, axes=([-1], [0]))


def box_kernel(k: int) -> np.ndarray:
    """Return the ``k x k`` averaging kernel.

    Parameters
    ----------
    k:
        Odd side length, at least 1.

    Returns
    -------
    numpy.ndarray
        ``float32`` array of shape ``(k, k)`` summing to 1.

    Raises
    ------
    ValueError
        If ``k`` is even or smaller than 1.
    """
    size = int(k)
    if size < 1 or size % 2 == 0:
        raise ValueError(f"k must be a positive odd integer, got {k!r}")
    return np.full((size, size), 1.0 / float(size * size), dtype=np.float32)


def _gaussian_1d(size: int, sigma: float | None = None) -> np.ndarray:
    """Return a normalised 1-D Gaussian of the given odd length."""
    side = int(size)
    if side < 1 or side % 2 == 0:
        raise ValueError(f"size must be a positive odd integer, got {size!r}")
    if sigma is None:
        deviation = 0.3 * ((side - 1) * 0.5 - 1.0) + 0.8
    else:
        deviation = float(sigma)
        if deviation <= 0.0:
            raise ValueError(f"sigma must be > 0, got {sigma!r}")
    axis = np.arange(side, dtype=np.float64) - (side - 1) / 2.0
    weights = np.exp(-(axis ** 2) / (2.0 * deviation ** 2))
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError("degenerate Gaussian kernel")
    return weights / total


def gaussian_kernel(
    size: int,
    sigma: float | None = None,
) -> np.ndarray:
    """Return a normalised 2-D Gaussian kernel.

    ``g(x, y) = exp(-(x^2 + y^2) / (2 sigma^2))`` normalised so that the
    kernel sums to 1.  When ``sigma`` is ``None`` the usual empirical rule
    ``sigma = 0.3 * ((size - 1) * 0.5 - 1) + 0.8`` is used.  The kernel is
    the outer product of a normalised 1-D Gaussian with itself, which is
    what makes the filter separable.

    Parameters
    ----------
    size:
        Odd side length, at least 1.
    sigma:
        Standard deviation in pixels; must be positive when given.

    Returns
    -------
    numpy.ndarray
        ``float32`` array of shape ``(size, size)`` summing to 1.

    Raises
    ------
    ValueError
        If ``size`` is even or ``sigma`` is not positive.
    """
    weights = _gaussian_1d(size, sigma)
    return np.outer(weights, weights).astype(np.float32)


def _to_uint8(values: np.ndarray) -> np.ndarray:
    """Round, clip and cast a float image to ``uint8``."""
    return np.clip(np.rint(values), 0, 255).astype(np.uint8)


def mean_filter(image: np.ndarray, k: int) -> np.ndarray:
    """Smooth an image with a ``k x k`` box filter.

    Parameters
    ----------
    image:
        ``uint8`` array of shape ``(H, W)`` or ``(H, W, C)``.
    k:
        Odd window side length.

    Returns
    -------
    numpy.ndarray
        ``uint8`` array of the same shape.

    Raises
    ------
    ValueError
        For an even or non-positive window.
    """
    array = _as_filterable(image)
    return _to_uint8(correlate2d(array, box_kernel(k)))


def gaussian_filter(
    image: np.ndarray,
    size: int = 5,
    sigma: float | None = None,
) -> np.ndarray:
    """Smooth an image with a separable Gaussian filter.

    A 2-D convolution is factorised into two 1-D passes, so the cost is
    ``O(size)`` per pixel instead of ``O(size^2)``.

    Parameters
    ----------
    image:
        ``uint8`` array of shape ``(H, W)`` or ``(H, W, C)``.
    size:
        Odd side length of the kernel.
    sigma:
        Standard deviation; derived from ``size`` when ``None``.

    Returns
    -------
    numpy.ndarray
        ``uint8`` array of the same shape.

    Raises
    ------
    ValueError
        For an even size or a non-positive sigma.
    """
    array = _as_filterable(image)
    weights_1d = np.ascontiguousarray(
        _gaussian_1d(size, sigma), dtype=np.float32
    )
    data = array.astype(np.float32)
    data = _filter_1d(data, weights_1d, axis=0)
    data = _filter_1d(data, weights_1d, axis=1)
    return _to_uint8(data)


def median_filter(image: np.ndarray, k: int) -> np.ndarray:
    """Remove salt-and-pepper noise with a median filter.

    The cost is ``O(k^2 log k)`` per pixel because a median of a window of
    ``k^2`` samples has to be found, which is why ``k`` is capped at
    :data:`MAX_MEDIAN_KSIZE`.

    Parameters
    ----------
    image:
        ``uint8`` array of shape ``(H, W)`` or ``(H, W, C)``.
    k:
        Odd window side length, at most 15.

    Returns
    -------
    numpy.ndarray
        ``uint8`` array of the same shape.

    Raises
    ------
    ValueError
        For an even, non-positive or excessive window size.
    """
    array = _as_filterable(image)
    side = int(k)
    if side < 1 or side % 2 == 0:
        raise ValueError(f"k must be a positive odd integer, got {k!r}")
    if side > MAX_MEDIAN_KSIZE:
        raise ValueError(
            f"k must not exceed {MAX_MEDIAN_KSIZE}, got {k!r}"
        )
    radius = side // 2
    padded = pad_image(array, radius, "reflect")
    windows = sliding_window_view(padded, (side, side), axis=(0, 1))
    return _to_uint8(np.median(windows, axis=(-2, -1)))


def unsharp_mask(
    image: np.ndarray,
    size: int = 5,
    sigma: float | None = None,
    amount: float = 1.0,
) -> np.ndarray:
    """Sharpen an image with the unsharp masking formula.

    ``g = f + amount * (f - gaussian(f))``.

    Parameters
    ----------
    image:
        ``uint8`` array of shape ``(H, W)`` or ``(H, W, C)``.
    size:
        Odd side length of the Gaussian kernel.
    sigma:
        Standard deviation; derived from ``size`` when ``None``.
    amount:
        Sharpening strength; ``0`` returns the input unchanged because
        ``g = f + 0 * (f - blur(f)) = f``.

    Returns
    -------
    numpy.ndarray
        ``uint8`` array of the same shape.

    Raises
    ------
    ValueError
        For a negative ``amount`` or a bad kernel size.
    """
    array = _as_filterable(image)
    if float(amount) < 0.0:
        raise ValueError(f"amount must be >= 0, got {amount!r}")
    blurred = gaussian_filter(array, size, sigma).astype(np.float32)
    data = array.astype(np.float32)
    return _to_uint8(data + float(amount) * (data - blurred))


def laplacian_sharpen(image: np.ndarray, variant: int = 4) -> np.ndarray:
    """Sharpen an image with the discrete Laplacian.

    ``g = f - laplacian(f)`` using the 4-neighbour kernel (the default) or
    the 8-neighbour kernel.

    Parameters
    ----------
    image:
        ``uint8`` array of shape ``(H, W)`` or ``(H, W, C)``.
    variant:
        4 or 8.

    Returns
    -------
    numpy.ndarray
        ``uint8`` array of the same shape.

    Raises
    ------
    ValueError
        If ``variant`` is neither 4 nor 8.
    """
    array = _as_filterable(image)
    if int(variant) == 4:
        kernel = LAPLACIAN_4
    elif int(variant) == 8:
        kernel = LAPLACIAN_8
    else:
        raise ValueError(f"variant must be 4 or 8, got {variant!r}")
    data = array.astype(np.float32)
    return _to_uint8(data - convolve2d(data, kernel))


def denoise(image: np.ndarray, cfg: PipelineConfig) -> np.ndarray:
    """Apply the pre-filter selected by the pipeline configuration.

    Parameters
    ----------
    image:
        ``uint8`` array of shape ``(H, W)`` or ``(H, W, C)``.
    cfg:
        Validated :class:`~bgremover.config.PipelineConfig`; ``cfg.denoise``
        selects ``"gaussian"``, ``"median"`` or ``"none"``.

    Returns
    -------
    numpy.ndarray
        ``uint8`` array of the same shape.  ``"none"`` returns a copy, so
        the result never aliases the input.

    Raises
    ------
    ValueError
        If the image is invalid or the configuration is not validated.
    """
    array = _as_filterable(image)
    cfg.validate()
    if cfg.denoise == "none":
        return array.copy()
    if cfg.denoise == "median":
        return median_filter(array, cfg.denoise_ksize)
    return gaussian_filter(array, cfg.denoise_ksize, cfg.gaussian_sigma)
