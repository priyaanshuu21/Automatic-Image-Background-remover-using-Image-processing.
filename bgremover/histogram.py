"""Histogram processing.

The histogram is the discrete distribution of gray levels; its cumulative
distribution is the basis of intensity equalisation.  For colour images the
value channel ``V = max(R, G, B)`` is equalised instead of the three channels
independently: stretching R, G and B separately changes their ratios and
therefore shifts the hue and the saturation, whereas scaling all three by the
same factor keeps both exactly intact.
"""

from __future__ import annotations

import logging

import numpy as np

LOGGER = logging.getLogger(__name__)


def _as_gray(gray: np.ndarray) -> np.ndarray:
    """Validate a 2-D ``uint8`` image."""
    array = np.asarray(gray)
    if array.ndim != 2:
        raise ValueError(
            f"expected a 2-D gray image, got shape {array.shape}"
        )
    if array.dtype != np.uint8:
        raise ValueError(f"expected a uint8 image, got dtype {array.dtype}")
    if array.size == 0:
        raise ValueError("image must not be empty")
    return array


def _as_rgb(rgb: np.ndarray) -> np.ndarray:
    """Validate a 3-D ``uint8`` RGB image."""
    array = np.asarray(rgb)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(
            f"expected an (H, W, 3) RGB image, got {array.shape}"
        )
    if array.dtype != np.uint8:
        raise ValueError(f"expected a uint8 image, got dtype {array.dtype}")
    if array.size == 0:
        raise ValueError("image must not be empty")
    return array


def compute_histogram(gray: np.ndarray, bins: int = 256) -> np.ndarray:
    """Count how often each gray level occurs.

    For the default ``bins = 256`` the result is exactly
    ``numpy.bincount(image.ravel(), minlength=256)``.  For a different number
    of bins the levels are quantised with ``index = (level * bins) // 256``,
    so every input sample is counted exactly once.

    Parameters
    ----------
    gray:
        2-D ``uint8`` image.
    bins:
        Number of bins, from 1 to 256.

    Returns
    -------
    numpy.ndarray
        ``int64`` array of length ``bins`` whose entries sum to the number
        of pixels.

    Raises
    ------
    ValueError
        For a bad image, dtype or bin count.
    """
    array = _as_gray(gray)
    count = int(bins)
    if not 1 <= count <= 256:
        raise ValueError(f"bins must lie in [1, 256], got {bins!r}")
    if count == 256:
        return np.bincount(array.ravel(), minlength=256).astype(np.int64)
    indices = (array.astype(np.int64) * count) // 256
    return np.bincount(indices.ravel(), minlength=count).astype(np.int64)


def normalized_histogram(gray: np.ndarray) -> np.ndarray:
    """Return the histogram as probabilities that sum to 1.

    Parameters
    ----------
    gray:
        2-D ``uint8`` image.

    Returns
    -------
    numpy.ndarray
        ``float64`` array of length 256.
    """
    hist = compute_histogram(gray).astype(np.float64)
    total = float(hist.sum())
    if total <= 0.0:
        raise ValueError("image must contain at least one pixel")
    return hist / total


def cumulative_distribution(hist: np.ndarray) -> np.ndarray:
    """Return the cumulative distribution function of a histogram.

    Parameters
    ----------
    hist:
        Non-negative histogram, e.g. the output of
        :func:`compute_histogram`.

    Returns
    -------
    numpy.ndarray
        ``float64`` array in ``[0, 1]`` whose last entry is 1.

    Raises
    ------
    ValueError
        For an empty or negative histogram.
    """
    values = np.asarray(hist, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("histogram must be a non-empty 1-D array")
    if np.any(values < 0):
        raise ValueError("histogram entries must be non-negative")
    total = float(values.sum())
    if total <= 0.0:
        raise ValueError("histogram must not be all zeros")
    return np.clip(np.cumsum(values) / total, 0.0, 1.0)


def equalize_gray(gray: np.ndarray) -> np.ndarray:
    """Equalise the contrast of a gray image.

    With ``N`` the number of pixels, ``cdf_k`` the cumulative count of level
    ``k`` and ``cdf_min`` the first non-zero cumulative count, the mapping
    is ``s_k = round(255 * (cdf_k - cdf_min) / (N - cdf_min))``.  A constant
    image has ``N == cdf_min`` and is returned unchanged.

    Parameters
    ----------
    gray:
        2-D ``uint8`` image.

    Returns
    -------
    numpy.ndarray
        ``uint8`` image of the same shape.

    Raises
    ------
    ValueError
        For a bad image or dtype.
    """
    array = _as_gray(gray)
    hist = compute_histogram(array)
    total = int(hist.sum())
    cumulative = np.cumsum(hist)
    non_zero = cumulative[cumulative > 0]
    if non_zero.size == 0:
        return array.copy()
    cdf_min = int(non_zero[0])
    denominator = total - cdf_min
    if denominator <= 0:
        LOGGER.debug("constant image, equalisation skipped")
        return array.copy()
    lut = np.rint(255.0 * (cumulative.astype(np.float64) - cdf_min)
                  / float(denominator))
    return np.clip(lut, 0, 255).astype(np.uint8)[array]


def color_histograms(rgb: np.ndarray) -> dict[str, np.ndarray]:
    """Return one 256-bin histogram per colour channel.

    Parameters
    ----------
    rgb:
        ``uint8`` array of shape ``(H, W, 3)``.

    Returns
    -------
    dict
        Keys ``"R"``, ``"G"`` and ``"B"`` mapped to ``int64`` arrays.

    Raises
    ------
    ValueError
        For a bad image or dtype.
    """
    array = _as_rgb(rgb)
    return {
        name: compute_histogram(array[:, :, index])
        for index, name in enumerate(("R", "G", "B"))
    }


def _rgb_to_hsv_v_only(rgb: np.ndarray) -> np.ndarray:
    """Return only the value component ``V = max(R, G, B)``.

    A private helper used by :func:`equalize_color_value` so that the
    histogram module does not depend on :mod:`bgremover.color`.
    """
    return rgb.astype(np.float32).max(axis=2)


def equalize_color_value(rgb: np.ndarray) -> np.ndarray:
    """Equalise the contrast of a colour image through its value channel.

    The value channel ``V = max(R, G, B)`` is equalised and the three colour
    channels are then rescaled by the common factor ``V_new / V``, which is
    exactly the value-channel equalisation of HSV.  Hue and saturation are
    preserved by construction, unlike per-channel RGB equalisation, which
    changes the ratios between the channels and therefore shifts colours.

    Parameters
    ----------
    rgb:
        ``uint8`` array of shape ``(H, W, 3)``.

    Returns
    -------
    numpy.ndarray
        New ``uint8`` array of shape ``(H, W, 3)``.

    Raises
    ------
    ValueError
        For a bad image or dtype.
    """
    array = _as_rgb(rgb)
    value = _rgb_to_hsv_v_only(array)
    value_u8 = np.clip(np.rint(value), 0, 255).astype(np.uint8)
    new_value = equalize_gray(value_u8).astype(np.float32)
    safe_value = np.maximum(value, 1.0)
    scale = (new_value / safe_value)[..., None]
    return np.clip(np.rint(array.astype(np.float32) * scale), 0, 255
                   ).astype(np.uint8)


def histogram_stats(gray: np.ndarray) -> dict[str, float]:
    """Return descriptive statistics of a gray image.

    Parameters
    ----------
    gray:
        2-D ``uint8`` image.

    Returns
    -------
    dict
        Keys ``mean``, ``std``, ``median``, ``mode`` (the most frequent
        level) and ``entropy`` in bits per pixel,
        ``H = -sum p log2 p`` over the non-empty bins.

    Raises
    ------
    ValueError
        For a bad image or dtype.
    """
    array = _as_gray(gray)
    hist = compute_histogram(array).astype(np.float64)
    total = float(hist.sum())
    levels = np.arange(256, dtype=np.float64)
    probabilities = hist / total
    positive = probabilities[probabilities > 0.0]
    entropy = float(-np.sum(positive * np.log2(positive)))
    return {
        "mean": float(np.sum(levels * probabilities)),
        "std": float(
            np.sqrt(
                np.sum(probabilities * (levels -
                                        np.sum(levels * probabilities)) ** 2)
            )
        ),
        "median": float(np.median(array.astype(np.float64))),
        "mode": float(int(np.argmax(hist))),
        "entropy": entropy,
    }
