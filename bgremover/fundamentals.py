"""Digital image fundamentals.

This module covers the first syllabus topic: matrix representation, colour
channels, spatial resolution, gray-level resolution and file formats.  Every
operator is written from scratch with NumPy; ``cv2.resize`` is the only
OpenCV call and is deliberately confined to :func:`resize_max_side` where
area-averaged downscaling is the right tool.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)

# ITU-R BT.601 luma weights, used by the manual gray conversion.
_LUMA_WEIGHTS = np.array([0.299, 0.587, 0.114], dtype=np.float32)
_RESAMPLE_METHODS = ("nearest", "bilinear")


@dataclass(frozen=True)
class ImageInfo:
    """Descriptive statistics of an image array.

    Attributes
    ----------
    height, width:
        Number of rows and columns.
    channels:
        1 for a 2-D array, otherwise the third dimension size.
    dtype:
        String form of the array dtype.
    bit_depth:
        Bits per sample.
    n_pixels:
        ``height * width``.
    nbytes:
        Size of the array in bytes.
    per_channel_min, per_channel_max, per_channel_mean:
        Per-channel statistics as tuples of floats.
    """

    height: int
    width: int
    channels: int
    dtype: str
    bit_depth: int
    n_pixels: int
    nbytes: int
    per_channel_min: tuple[float, ...]
    per_channel_max: tuple[float, ...]
    per_channel_mean: tuple[float, ...]


def _as_array(image: np.ndarray) -> np.ndarray:
    """Validate and return the input as an ndarray of rank 2 or 3."""
    array = np.asarray(image)
    if array.ndim not in (2, 3):
        raise ValueError(
            f"image must have 2 or 3 dimensions, got shape {array.shape}"
        )
    return array


def describe_image(image: np.ndarray) -> ImageInfo:
    """Return a :class:`ImageInfo` summary of an image array.

    Parameters
    ----------
    image:
        Array of shape ``(H, W)`` or ``(H, W, C)``.

    Returns
    -------
    ImageInfo
        Height, width, channel count, dtype, bit depth, memory footprint
        and per-channel minimum, maximum and arithmetic mean.

    Raises
    ------
    ValueError
        If the array rank is not 2 or 3.
    """
    array = _as_array(image)
    if array.ndim == 2:
        flat = array.reshape(-1, 1)
        channels = 1
    else:
        flat = array.reshape(-1, array.shape[2])
        channels = int(array.shape[2])
    return ImageInfo(
        height=int(array.shape[0]),
        width=int(array.shape[1]),
        channels=channels,
        dtype=str(array.dtype),
        bit_depth=int(array.dtype.itemsize * 8),
        n_pixels=int(array.shape[0] * array.shape[1]),
        nbytes=int(array.nbytes),
        per_channel_min=tuple(float(v) for v in flat.min(axis=0)),
        per_channel_max=tuple(float(v) for v in flat.max(axis=0)),
        per_channel_mean=tuple(float(v) for v in flat.mean(axis=0)),
    )


def to_gray(rgb: np.ndarray) -> np.ndarray:
    """Convert an RGB image to gray using the BT.601 luma formula.

    ``Y = 0.299 R + 0.587 G + 0.114 B`` is evaluated in ``float32`` and
    rounded with ``np.rint``; the result is clipped to ``[0, 255]``.  The
    conversion is written out manually so that it does not depend on the
    OpenCV version.

    Parameters
    ----------
    rgb:
        ``uint8`` array of shape ``(H, W, 3)`` in RGB order.

    Returns
    -------
    numpy.ndarray
        ``uint8`` array of shape ``(H, W)``.

    Raises
    ------
    ValueError
        If the input is not an ``uint8`` array of shape ``(H, W, 3)``.
    """
    array = np.asarray(rgb)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(
            f"to_gray expects an (H, W, 3) array, got {array.shape}"
        )
    if array.dtype != np.uint8:
        raise ValueError(f"to_gray expects uint8 input, got {array.dtype}")
    weights = _LUMA_WEIGHTS.reshape(1, 1, 3)
    luma = array.astype(np.float32) * weights
    return np.clip(np.rint(luma.sum(axis=2)), 0, 255).astype(np.uint8)


def split_channels(image: np.ndarray) -> tuple[np.ndarray, ...]:
    """Split an image into its channels.

    Parameters
    ----------
    image:
        ``uint8`` array of shape ``(H, W, C)`` with ``C >= 1``.

    Returns
    -------
    tuple of numpy.ndarray
        One ``(H, W)`` array per channel, in order.

    Raises
    ------
    ValueError
        If the input is not a 3-D array.
    """
    array = _as_array(image)
    if array.ndim != 3:
        raise ValueError(
            f"split_channels expects an (H, W, C) array, got {array.shape}"
        )
    return tuple(np.ascontiguousarray(array[:, :, c]) for c in
                 range(array.shape[2]))


def merge_channels(*channels: np.ndarray) -> np.ndarray:
    """Stack equally shaped ``(H, W)`` channels into a ``(H, W, C)`` array.

    Parameters
    ----------
    *channels:
        One or more 2-D arrays of identical shape.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(H, W, len(channels))``.

    Raises
    ------
    ValueError
        If no channel is given or the shapes differ.
    """
    if not channels:
        raise ValueError("merge_channels needs at least one channel")
    first = np.asarray(channels[0])
    if first.ndim != 2:
        raise ValueError(
            f"channels must be 2-D, got shape {first.shape}"
        )
    for index, channel in enumerate(channels):
        if np.asarray(channel).shape != first.shape:
            raise ValueError(
                f"channel {index} has shape {np.asarray(channel).shape}, "
                f"expected {first.shape}"
            )
    return np.stack(channels, axis=2)


def _target_size(height: int, width: int, factor: float) -> tuple[int, int]:
    """Return the output size for a resampling factor."""
    out_h = max(1, int(round(height * factor)))
    out_w = max(1, int(round(width * factor)))
    return out_h, out_w


def spatial_resample(
    image: np.ndarray,
    factor: float,
    method: str = "nearest",
) -> np.ndarray:
    """Resample an image by a scale factor using hand-written interpolation.

    Destination pixel ``(y_dst, x_dst)`` samples the source at
    ``x_src = (x_dst + 0.5) / factor - 0.5`` (and likewise for ``y``), which
    keeps the image centred under both magnification and minification.
    ``nearest`` rounds to the closest source pixel, ``bilinear`` interpolates
    the four surrounding samples with weights
    ``(1 - fy)(1 - fx)``, ``(1 - fy)fx``, ``fy(1 - fx)`` and ``fy fx``.
    Coordinates are clamped to the image border, so the border is extended
    rather than zero padded.

    Parameters
    ----------
    image:
        Array of shape ``(H, W)`` or ``(H, W, C)``.
    factor:
        Strictly positive scale factor; ``0.5`` halves, ``2.0`` doubles.
    method:
        ``"nearest"`` or ``"bilinear"``.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(round(H * factor), round(W * factor))`` with the
        same number of channels and the same dtype as the input.

    Raises
    ------
    ValueError
        For a non-positive factor, an unknown method or a bad array rank.
    """
    array = _as_array(image)
    if method not in _RESAMPLE_METHODS:
        raise ValueError(
            f"method must be one of {_RESAMPLE_METHODS}, got {method!r}"
        )
    scale = float(factor)
    if scale <= 0.0:
        raise ValueError(f"factor must be > 0, got {factor!r}")
    height, width = array.shape[:2]
    out_h, out_w = _target_size(height, width, scale)
    squeeze = array.ndim == 2
    work = array[:, :, None] if squeeze else array

    dst_rows = np.arange(out_h, dtype=np.float32)
    dst_cols = np.arange(out_w, dtype=np.float32)
    src_y = (dst_rows + 0.5) / scale - 0.5
    src_x = (dst_cols + 0.5) / scale - 0.5
    src_y = np.clip(src_y, 0.0, height - 1)
    src_x = np.clip(src_x, 0.0, width - 1)

    if method == "nearest":
        iy = np.clip(np.rint(src_y).astype(np.intp), 0, height - 1)
        ix = np.clip(np.rint(src_x).astype(np.intp), 0, width - 1)
        out = work[iy][:, ix]
    else:
        y0 = np.floor(src_y).astype(np.intp)
        x0 = np.floor(src_x).astype(np.intp)
        wy = (src_y - y0).astype(np.float32)[:, None, None]
        wx = (src_x - x0).astype(np.float32)[None, :, None]
        y0c = np.clip(y0, 0, height - 1)
        y1c = np.clip(y0 + 1, 0, height - 1)
        x0c = np.clip(x0, 0, width - 1)
        x1c = np.clip(x0 + 1, 0, width - 1)
        data = work.astype(np.float32)
        top = data[y0c][:, x0c] * (1.0 - wx) + data[y0c][:, x1c] * wx
        bottom = data[y1c][:, x0c] * (1.0 - wx) + data[y1c][:, x1c] * wx
        out = top * (1.0 - wy) + bottom * wy

    if squeeze:
        out = out[:, :, 0]
    if np.issubdtype(array.dtype, np.integer):
        info = np.iinfo(array.dtype)
        out = np.clip(np.rint(out), info.min, info.max).astype(array.dtype)
    else:
        out = out.astype(array.dtype, copy=False)
    return np.ascontiguousarray(out)


def requantize(gray: np.ndarray, levels: int) -> np.ndarray:
    """Reduce the gray-level resolution of an image.

    With ``step = 256 / levels`` the mapping is
    ``s = floor(r / step) * step + step / 2``, clipped to ``[0, 255]``, so
    every output level is the mid-point of its quantisation interval.

    Parameters
    ----------
    gray:
        2-D ``uint8`` image.
    levels:
        Number of gray levels to keep, from 2 to 256.

    Returns
    -------
    numpy.ndarray
        ``uint8`` array of the same shape using at most ``levels`` distinct
        values.

    Raises
    ------
    ValueError
        If the input is not 2-D ``uint8`` or ``levels`` is out of range.
    """
    array = np.asarray(gray)
    if array.ndim != 2:
        raise ValueError(
            f"requantize expects a 2-D array, got shape {array.shape}"
        )
    if array.dtype != np.uint8:
        raise ValueError(f"requantize expects uint8 input, got {array.dtype}")
    count = int(levels)
    if not 2 <= count <= 256:
        raise ValueError(f"levels must lie in [2, 256], got {levels!r}")
    step = 256.0 / count
    scaled = array.astype(np.float32) / step
    quantised = np.floor(scaled) * step + step / 2.0
    return np.clip(np.rint(quantised), 0, 255).astype(np.uint8)


def resize_max_side(
    image: np.ndarray,
    max_side: int,
) -> tuple[np.ndarray, float]:
    """Shrink an image so that its longest side is at most ``max_side``.

    ``cv2.resize`` is used here on purpose: its ``INTER_AREA`` mode performs
    proper area averaging when minifying, which no hand-written 4-neighbour
    interpolation can do without aliasing.

    Parameters
    ----------
    image:
        Array of shape ``(H, W)`` or ``(H, W, C)``.
    max_side:
        Longest side allowed, at least 1.

    Returns
    -------
    tuple
        The resized array (a copy of the input when no resize is needed) and
        the scale factor that was applied, ``1.0`` when the image already
        fits.

    Raises
    ------
    ValueError
        If ``max_side`` is smaller than 1 or the array rank is invalid.
    """
    array = _as_array(image)
    limit = int(max_side)
    if limit < 1:
        raise ValueError(f"max_side must be >= 1, got {max_side!r}")
    height, width = array.shape[:2]
    longest = max(height, width)
    if longest <= limit:
        return array.copy(), 1.0
    scale = limit / float(longest)
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    resized = cv2.resize(
        array,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA,
    )
    LOGGER.debug("resized %s to %s", array.shape, resized.shape)
    if resized.ndim == 2 and array.ndim == 3:
        resized = resized[:, :, None]
    return resized, scale


def bit_depth_table(image: np.ndarray) -> str:
    """Return a small formatted report about an image array.

    Parameters
    ----------
    image:
        Array of shape ``(H, W)`` or ``(H, W, C)``.

    Returns
    -------
    str
        Multi-line report with rows, columns, channels, bits per sample,
        bits per pixel, total bytes and the per-channel range.  Intended for
        teaching output; it never prints.
    """
    info = describe_image(image)
    bits_per_pixel = info.bit_depth * info.channels
    lines = [
        "image description",
        f"  rows            : {info.height}",
        f"  columns         : {info.width}",
        f"  channels        : {info.channels}",
        f"  dtype           : {info.dtype}",
        f"  bits per sample : {info.bit_depth}",
        f"  bits per pixel  : {bits_per_pixel}",
        f"  total bytes     : {info.nbytes}",
        f"  pixels          : {info.n_pixels}",
    ]
    for index, (low, high, mean) in enumerate(
        zip(
            info.per_channel_min,
            info.per_channel_max,
            info.per_channel_mean,
        )
    ):
        lines.append(
            f"  channel {index}       : min={low:.3f} max={high:.3f} "
            f"mean={mean:.3f}"
        )
    return "\n".join(lines)


def channels_summary(image: np.ndarray) -> str:
    """Return a one-line summary of the channel layout of an image.

    Parameters
    ----------
    image:
        Array of shape ``(H, W)`` or ``(H, W, C)``.

    Returns
    -------
    str
        Human readable description such as ``"160x120 RGB uint8"``.
    """
    info = describe_image(image)
    names: dict[int, str] = {
        1: "gray",
        2: "gray+alpha",
        3: "RGB",
        4: "RGBA",
    }
    name = names.get(info.channels, "multi-channel")
    return f"{info.width}x{info.height} {name} {info.dtype}"
