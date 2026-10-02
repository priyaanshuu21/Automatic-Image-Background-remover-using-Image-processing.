"""Image input and output built on Pillow.

The whole project works on plain ``numpy`` arrays with the following
conventions:

* colour images -- ``uint8`` with shape ``(H, W, 3)`` in **RGB** order,
* gray images -- ``uint8`` with shape ``(H, W)``,
* binary masks -- ``bool`` with shape ``(H, W)``,
* floating point images -- ``float32`` in ``[0, 1]``.

Only Pillow touches the file system, so JPEG, PNG, TIFF and BMP behave the
same everywhere and EXIF orientation is handled exactly once, at load time.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

LOGGER = logging.getLogger(__name__)

READABLE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
WRITABLE_SUFFIXES = READABLE_SUFFIXES
ALPHA_SUFFIXES = (".png", ".tif", ".tiff")
JPEG_SUFFIXES = (".jpg", ".jpeg")


def _check_suffix(path: Path, allowed: tuple[str, ...]) -> str:
    """Return the lowercase suffix, raising if it is not allowed."""
    suffix = path.suffix.lower()
    if suffix not in allowed:
        raise ValueError(
            f"unsupported image extension {suffix!r} for {path}; "
            f"expected one of {', '.join(allowed)}"
        )
    return suffix


def _flatten_to_rgb(image: Image.Image) -> Image.Image:
    """Convert any Pillow mode to RGB, compositing alpha over white."""
    if image.mode in ("RGBA", "LA") or (
        image.mode == "P" and "transparency" in image.info
    ):
        rgba = image.convert("RGBA")
        canvas = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(canvas, rgba).convert("RGB")
    if image.mode == "CMYK":
        return image.convert("RGB")
    return image.convert("RGB")


def load_image(path: str | Path) -> np.ndarray:
    """Load an image file as an ``uint8`` RGB array.

    EXIF orientation is applied first, then palette, grayscale, CMYK and
    alpha images are converted to RGB.  Fully transparent pixels are
    composited over a white background.

    Parameters
    ----------
    path:
        Path to a ``.png``, ``.jpg``, ``.jpeg``, ``.tif``, ``.tiff`` or
        ``.bmp`` file.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(H, W, 3)`` and dtype ``uint8`` in RGB order.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the extension is not supported or the file cannot be decoded.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"image file not found: {file_path}")
    _check_suffix(file_path, READABLE_SUFFIXES)
    try:
        with Image.open(file_path) as handle:
            handle.load()
            rgb = _flatten_to_rgb(ImageOps.exif_transpose(handle))
            array = np.array(rgb, dtype=np.uint8)
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot decode image {file_path}: {exc}") from exc
    LOGGER.debug("loaded %s with shape %s", file_path, array.shape)
    return np.ascontiguousarray(array)


def _as_pil_image(array: np.ndarray) -> Image.Image:
    """Convert a validated uint8 array to a Pillow image."""
    if array.ndim == 2:
        return Image.fromarray(array)
    if array.ndim == 3 and array.shape[2] == 3:
        return Image.fromarray(array)
    if array.ndim == 3 and array.shape[2] == 4:
        return Image.fromarray(array)
    raise ValueError(
        "array must have shape (H, W), (H, W, 3) or (H, W, 4), got "
        f"{array.shape}"
    )


def save_image(
    path: str | Path,
    array: np.ndarray,
    *,
    jpeg_quality: int = 95,
) -> None:
    """Save a uint8 array to disk, creating parent folders as needed.

    Parameters
    ----------
    path:
        Destination file; the extension selects the encoder.
    array:
        ``uint8`` array of shape ``(H, W)``, ``(H, W, 3)`` or ``(H, W, 4)``.
    jpeg_quality:
        Quality used by the JPEG encoder only.

    Raises
    ------
    ValueError
        If the dtype, the shape or the extension is not supported, or if an
        RGBA array is written to a JPEG file (JPEG has no alpha channel).
    """
    file_path = Path(path)
    suffix = _check_suffix(file_path, WRITABLE_SUFFIXES)
    data = np.asarray(array)
    if data.dtype != np.uint8:
        raise ValueError(f"array dtype must be uint8, got {data.dtype}")
    image = _as_pil_image(data)
    if data.ndim == 3 and data.shape[2] == 4 and suffix in JPEG_SUFFIXES:
        raise ValueError(
            "JPEG cannot store an alpha channel; save RGBA data as PNG or "
            f"TIFF instead (requested path: {file_path})"
        )
    file_path.parent.mkdir(parents=True, exist_ok=True)
    if suffix in JPEG_SUFFIXES:
        image.save(file_path, quality=int(jpeg_quality))
    else:
        image.save(file_path)
    LOGGER.debug("saved %s with shape %s", file_path, data.shape)


def image_file_info(path: str | Path) -> dict[str, Any]:
    """Return metadata about an image file without decoding its pixels.

    Parameters
    ----------
    path:
        Path of an existing image file.

    Returns
    -------
    dict
        Keys ``path``, ``suffix``, ``format``, ``mode``, ``width``,
        ``height`` and ``size_bytes``.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"image file not found: {file_path}")
    with Image.open(file_path) as handle:
        width, height = handle.size
        info: dict[str, Any] = {
            "path": str(file_path),
            "suffix": file_path.suffix.lower(),
            "format": handle.format,
            "mode": handle.mode,
            "width": int(width),
            "height": int(height),
            "size_bytes": int(file_path.stat().st_size),
        }
    return info
