"""Point processing (intensity mapping) operations.

Every operator here maps gray levels independently, so the whole family is
implemented with a 256-entry lookup table indexed by the ``uint8`` input.
That keeps the operations exact, fast and free of any per-pixel Python loop,
and it works unchanged on a gray image or on each channel of a colour image.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np

LOGGER = logging.getLogger(__name__)

SLICE_MODES = ("highlight", "preserve")
_LEVELS = np.arange(256, dtype=np.float64)


def _checked_image(image: np.ndarray) -> np.ndarray:
    """Validate a uint8 image of rank 2 or 3 and return it as an array."""
    array = np.asarray(image)
    if array.dtype != np.uint8:
        raise ValueError(f"expected a uint8 image, got dtype {array.dtype}")
    if array.ndim not in (2, 3):
        raise ValueError(
            f"expected an (H, W) or (H, W, C) image, got {array.shape}"
        )
    return array


def _checked_lut(lut: np.ndarray) -> np.ndarray:
    """Validate a 256-entry ``uint8`` lookup table."""
    table = np.asarray(lut)
    if table.dtype != np.uint8:
        raise ValueError(
            f"lookup table must have dtype uint8, got {table.dtype}"
        )
    if table.shape != (256,):
        raise ValueError(
            f"lookup table must have shape (256,), got {table.shape}"
        )
    return table


def apply_lut(image: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Map gray levels through a 256-entry lookup table.

    Parameters
    ----------
    image:
        ``uint8`` array of shape ``(H, W)`` or ``(H, W, C)``.
    lut:
        ``uint8`` array of shape ``(256,)`` holding the output level of
        every input level.

    Returns
    -------
    numpy.ndarray
        A new array of the same shape and dtype as ``image``.

    Raises
    ------
    ValueError
        If the image or the lookup table has a wrong dtype or shape.
    """
    array = _checked_image(image)
    table = _checked_lut(lut)
    return table[array]


def negative(image: np.ndarray) -> np.ndarray:
    """Produce the photographic negative ``s = 255 - r``.

    Parameters
    ----------
    image:
        ``uint8`` image.

    Returns
    -------
    numpy.ndarray
        New ``uint8`` image of the same shape.
    """
    return apply_lut(image, (255 - _LEVELS).astype(np.uint8))


def contrast_stretch(
    image: np.ndarray,
    low_pct: float = 0.0,
    high_pct: float = 100.0,
) -> np.ndarray:
    """Linearly stretch the intensity range of an image.

    With ``r_lo = percentile(image, low_pct)`` and
    ``r_hi = percentile(image, high_pct)`` the mapping is
    ``s = (r - r_lo) / (r_hi - r_lo) * 255`` clipped to ``[0, 255]``.  A
    constant image has ``r_hi == r_lo`` and is returned unchanged instead of
    dividing by zero.

    Parameters
    ----------
    image:
        ``uint8`` image.
    low_pct, high_pct:
        Percentiles, in ``[0, 100]``, defining the black and white points.

    Returns
    -------
    numpy.ndarray
        New ``uint8`` image of the same shape.

    Raises
    ------
    ValueError
        For non-``uint8`` input or an invalid percentile pair.
    """
    array = _checked_image(image)
    if not 0.0 <= float(low_pct) < float(high_pct) <= 100.0:
        raise ValueError(
            "percentiles must satisfy 0 <= low_pct < high_pct <= 100, got "
            f"low_pct={low_pct!r}, high_pct={high_pct!r}"
        )
    r_lo = float(np.percentile(array, low_pct))
    r_hi = float(np.percentile(array, high_pct))
    if r_hi <= r_lo:
        LOGGER.debug("constant image, contrast stretch skipped")
        return array.copy()
    lut = np.clip(
        (_LEVELS - r_lo) / (r_hi - r_lo) * 255.0, 0.0, 255.0
    )
    return apply_lut(array, np.rint(lut).astype(np.uint8))


def piecewise_stretch(
    image: np.ndarray,
    points: Sequence[tuple[float, float]],
) -> np.ndarray:
    """Apply an arbitrary piecewise-linear intensity transformation.

    The control points ``(r_in, s_out)`` are sorted, the endpoints
    ``(0, 0)`` and ``(255, 255)`` are added automatically and the levels in
    between are linearly interpolated.

    Parameters
    ----------
    image:
        ``uint8`` image.
    points:
        Sequence of ``(input level, output level)`` pairs.

    Returns
    -------
    numpy.ndarray
        New ``uint8`` image of the same shape.

    Raises
    ------
    ValueError
        For non-``uint8`` input or control points outside ``[0, 255]``.
    """
    array = _checked_image(image)
    control = [(0.0, 0.0), (255.0, 255.0)]
    for pair in points:
        x_value, y_value = float(pair[0]), float(pair[1])
        if not 0.0 <= x_value <= 255.0 or not 0.0 <= y_value <= 255.0:
            raise ValueError(
                f"control points must lie in [0, 255], got {pair!r}"
            )
        control.append((x_value, y_value))
    control.sort(key=lambda item: item[0])
    xs = np.array([item[0] for item in control], dtype=np.float64)
    ys = np.array([item[1] for item in control], dtype=np.float64)
    lut = np.interp(_LEVELS, xs, ys)
    return apply_lut(array, np.rint(np.clip(lut, 0, 255)).astype(np.uint8))


def threshold_binary(
    gray: np.ndarray,
    t: int,
    invert: bool = False,
) -> np.ndarray:
    """Threshold a gray image into a boolean mask.

    Parameters
    ----------
    gray:
        ``uint8`` array of shape ``(H, W)``.
    t:
        Threshold level in ``[0, 255]``; a pixel is above the threshold
        when ``gray > t``.
    invert:
        When true, return the complement (``gray <= t``).

    Returns
    -------
    numpy.ndarray
        New ``bool`` array of the same shape.

    Raises
    ------
    ValueError
        If the input is not a 2-D ``uint8`` image or ``t`` is out of range.
    """
    array = _checked_image(gray)
    if array.ndim != 2:
        raise ValueError(
            f"threshold_binary expects a 2-D image, got {array.shape}"
        )
    if not 0 <= int(t) <= 255:
        raise ValueError(f"t must lie in [0, 255], got {t!r}")
    level = int(t)
    if invert:
        return array <= level
    return array > level


def intensity_slice(
    gray: np.ndarray,
    low: int,
    high: int,
    mode: str = "highlight",
    value: int = 255,
) -> np.ndarray:
    """Isolate a band of gray levels.

    Parameters
    ----------
    gray:
        ``uint8`` array of shape ``(H, W)``.
    low, high:
        Inclusive band limits in ``[0, 255]``.
    mode:
        ``"highlight"`` sets everything outside the band to 0, ``"preserve"``
        keeps the other pixels unchanged.
    value:
        Gray level written inside the band.

    Returns
    -------
    numpy.ndarray
        New ``uint8`` array of the same shape.

    Raises
    ------
    ValueError
        For a bad mode, an inverted band or a value outside ``[0, 255]``.
    """
    array = _checked_image(gray)
    if array.ndim != 2:
        raise ValueError(
            f"intensity_slice expects a 2-D image, got {array.shape}"
        )
    if mode not in SLICE_MODES:
        raise ValueError(
            f"mode must be one of {SLICE_MODES}, got {mode!r}"
        )
    if not 0 <= int(low) <= int(high) <= 255:
        raise ValueError(
            f"need 0 <= low <= high <= 255, got low={low!r}, high={high!r}"
        )
    if not 0 <= int(value) <= 255:
        raise ValueError(f"value must lie in [0, 255], got {value!r}")
    band = (array >= int(low)) & (array <= int(high))
    if mode == "highlight":
        out = np.zeros_like(array)
    else:
        out = array.copy()
    out[band] = np.uint8(int(value))
    return out


def log_transform(image: np.ndarray, c: float | None = None) -> np.ndarray:
    """Apply the logarithmic transformation ``s = c * log(1 + r)``.

    Parameters
    ----------
    image:
        ``uint8`` image.
    c:
        Constant; defaults to ``255 / log(256)`` so that ``0`` maps to ``0``
        and ``255`` maps to ``255``.

    Returns
    -------
    numpy.ndarray
        New ``uint8`` image of the same shape.

    Raises
    ------
    ValueError
        For non-``uint8`` input or a non-positive ``c``.
    """
    array = _checked_image(image)
    constant = (255.0 / np.log(256.0)) if c is None else float(c)
    if constant <= 0.0:
        raise ValueError(f"c must be > 0, got {c!r}")
    lut = np.clip(np.rint(constant * np.log1p(_LEVELS)), 0, 255)
    return apply_lut(array, lut.astype(np.uint8))


def power_law(
    image: np.ndarray,
    gamma: float,
    c: float = 1.0,
) -> np.ndarray:
    """Apply the power-law (gamma) transformation.

    ``s = 255 * c * (r / 255) ** gamma`` clipped to ``[0, 255]``.  A gamma
    below 1 brightens the image, a gamma above 1 darkens it.

    Parameters
    ----------
    image:
        ``uint8`` image.
    gamma:
        Exponent; must be strictly positive.
    c:
        Scale factor; must be strictly positive.

    Returns
    -------
    numpy.ndarray
        New ``uint8`` image of the same shape.

    Raises
    ------
    ValueError
        For non-``uint8`` input or a non-positive ``gamma`` or ``c``.
    """
    array = _checked_image(image)
    if float(gamma) <= 0.0:
        raise ValueError(f"gamma must be > 0, got {gamma!r}")
    if float(c) <= 0.0:
        raise ValueError(f"c must be > 0, got {c!r}")
    normalized = _LEVELS / 255.0
    lut = np.clip(
        255.0 * float(c) * np.power(normalized, float(gamma)), 0.0, 255.0
    )
    return apply_lut(array, np.rint(lut).astype(np.uint8))
