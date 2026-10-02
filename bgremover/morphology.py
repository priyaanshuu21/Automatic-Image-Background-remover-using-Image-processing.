"""Binary and grayscale morphology with mask cleanup.

All operations are implemented classically: structuring elements are generated
explicitly, erosion/dilation follow the min/max definition, and opening/closing
are the standard compositions. No calls to OpenCV morphology are used in the
implementation of these functions.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "structuring_element",
    "dilate",
    "erode",
    "opening",
    "closing",
    "hit_or_miss",
    "thinning",
    "thickening",
    "morphological_gradient",
    "top_hat",
    "black_hat",
    "clean_mask",
    "fill_holes",
    "remove_small_objects",
    "remove_objects_touching_border",
    "morphology_defaults",
]


def _as_2d(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array)
    if arr.ndim != 2:
        raise ValueError(f"expected a 2-D array, got {arr.shape}")
    if arr.size == 0:
        raise ValueError("the array is empty")
    return arr


def structuring_element(
    shape: str = "disk",
    radius: int = 1,
    size: tuple[int, int] | None = None,
) -> np.ndarray:
    if size is not None:
        rows, cols = int(size[0]), int(size[1])
        if rows < 1 or cols < 1:
            raise ValueError(f"size must be positive, got {size}")
        return np.ones((rows, cols), dtype=bool)

    r = int(radius)
    if r < 0:
        raise ValueError(f"radius must be non-negative, got {radius}")
    s = 2 * r + 1
    shape_lower = shape.lower()
    if shape_lower == "square":
        return np.ones((s, s), dtype=bool)
    if shape_lower == "disk":
        y, x = np.ogrid[-r : r + 1, -r : r + 1]
        dist = np.sqrt(x * x + y * y)
        return dist <= r + 1e-10
    if shape_lower == "cross":
        element = np.zeros((s, s), dtype=bool)
        element[r, :] = True
        element[:, r] = True
        return element
    raise ValueError(f"unsupported structuring element shape: {shape!r}")


def _pad_for_kernel(
    image: np.ndarray,
    kernel: np.ndarray,
    border: str = "constant",
) -> tuple[np.ndarray, tuple[slice, slice]]:
    krows, kcols = kernel.shape
    pad_r = krows // 2
    pad_c = kcols // 2
    border = border.lower()
    if border == "constant":
        padded = np.pad(image, ((pad_r, pad_r), (pad_c, pad_c)), mode="constant")
    elif border == "replicate":
        padded = np.pad(image, ((pad_r, pad_r), (pad_c, pad_c)), mode="edge")
    elif border == "reflect":
        padded = np.pad(
            image, ((pad_r, pad_r), (pad_c, pad_c)), mode="reflect"
        )
    elif border == "wrap":
        padded = np.pad(image, ((pad_r, pad_r), (pad_c, pad_c)), mode="wrap")
    else:
        raise ValueError(f"unsupported border mode: {border}")
    crop = slice(pad_r, padded.shape[0] - pad_r), slice(
        pad_c, padded.shape[1] - pad_c
    )
    return padded, crop


def dilate(
    array: np.ndarray,
    kernel: np.ndarray | None = None,
    iterations: int = 1,
    border: str = "constant",
) -> np.ndarray:
    data = np.asarray(array)
    is_bool = data.dtype == bool
    image = _as_2d(data)
    if kernel is None:
        kernel = structuring_element("disk", 1)
    kern = np.asarray(kernel).astype(bool)
    if kern.ndim != 2 or not kern.any():
        raise ValueError("kernel must be a non-empty 2-D boolean array")
    if int(iterations) < 1:
        raise ValueError(f"iterations must be >= 1, got {iterations}")
    result = image
    for _ in range(int(iterations)):
        padded, crop = _pad_for_kernel(result, kern, border=border)
        out = np.zeros_like(result, dtype=result.dtype)
        krows, kcols = kern.shape
        coords = np.argwhere(kern)
        for dr, dc in coords:
            shifted = padded[
                dr : padded.shape[0] - (krows - 1 - dr),
                dc : padded.shape[1] - (kcols - 1 - dc),
            ]
            if shifted.shape == out.shape:
                out = np.maximum(out, shifted)
            else:
                out[crop] = np.maximum(out[crop], shifted[crop])
        result = out
    if is_bool:
        return result.astype(bool)
    return result


def erode(
    array: np.ndarray,
    kernel: np.ndarray | None = None,
    iterations: int = 1,
    border: str = "constant",
) -> np.ndarray:
    data = np.asarray(array)
    is_bool = data.dtype == bool
    image = _as_2d(data)
    if kernel is None:
        kernel = structuring_element("disk", 1)
    kern = np.asarray(kernel).astype(bool)
    if kern.ndim != 2 or not kern.any():
        raise ValueError("kernel must be a non-empty 2-D boolean array")
    if int(iterations) < 1:
        raise ValueError(f"iterations must be >= 1, got {iterations}")
    if is_bool:
        result = image.astype(float)
        fill_val = 1.0
    else:
        result = image.astype(float)
        fill_val = float(np.inf)
    for _ in range(int(iterations)):
        padded, crop = _pad_for_kernel(result, kern, border=border)
        out = np.full_like(result, fill_val, dtype=float)
        krows, kcols = kern.shape
        coords = np.argwhere(kern)
        for dr, dc in coords:
            shifted = padded[
                dr : padded.shape[0] - (krows - 1 - dr),
                dc : padded.shape[1] - (kcols - 1 - dc),
            ]
            if shifted.shape == out.shape:
                out = np.minimum(out, shifted)
            else:
                out[crop] = np.minimum(out[crop], shifted[crop])
        result = out
    if is_bool:
        return result <= 0.5
    return result.astype(data.dtype) if result.dtype == data.dtype else result


def opening(array: np.ndarray, kernel: np.ndarray | None = None, border: str = "constant") -> np.ndarray:
    eroded = erode(array, kernel=kernel, iterations=1, border=border)
    return dilate(eroded, kernel=kernel, iterations=1, border=border)


def closing(array: np.ndarray, kernel: np.ndarray | None = None, border: str = "constant") -> np.ndarray:
    dilated = dilate(array, kernel=kernel, iterations=1, border=border)
    return erode(dilated, kernel=kernel, iterations=1, border=border)


def morphological_gradient(array: np.ndarray, kernel: np.ndarray | None = None) -> np.ndarray:
    dilated = dilate(array, kernel=kernel)
    eroded = erode(array, kernel=kernel)
    return dilated.astype(float) - eroded.astype(float)


def top_hat(array: np.ndarray, kernel: np.ndarray | None = None) -> np.ndarray:
    opened = opening(array, kernel=kernel)
    return array.astype(float) - opened.astype(float)


def black_hat(array: np.ndarray, kernel: np.ndarray | None = None) -> np.ndarray:
    closed = closing(array, kernel=kernel)
    return closed.astype(float) - array.astype(float)
