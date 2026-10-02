"""Binary and grayscale morphology with mask cleanup.

All operations are implemented classically: structuring elements are generated
explicitly, erosion/dilation follow the min/max definition, and opening/closing
are the standard compositions. No calls to OpenCV morphology are used in the
implementation of these functions.
"""

from __future__ import annotations

import numpy as np

from bgremover.config import PipelineConfig
from bgremover.pixels import border_touching_labels, label_components

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
        padded = np.pad(
            image, ((pad_r, pad_r), (pad_c, pad_c)), mode="constant"
        )
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
    if kern.shape[0] % 2 == 0 or kern.shape[1] % 2 == 0:
        raise ValueError(
            f"kernel sides must be odd, got shape {kern.shape}"
        )
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
    if kern.shape[0] % 2 == 0 or kern.shape[1] % 2 == 0:
        raise ValueError(
            f"kernel sides must be odd, got shape {kern.shape}"
        )
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
        return result >= 0.5
    return result.astype(data.dtype)


def opening(
    array: np.ndarray,
    kernel: np.ndarray | None = None,
    border: str = "constant",
) -> np.ndarray:
    eroded = erode(array, kernel=kernel, iterations=1, border=border)
    return dilate(eroded, kernel=kernel, iterations=1, border=border)


def closing(
    array: np.ndarray,
    kernel: np.ndarray | None = None,
    border: str = "constant",
) -> np.ndarray:
    dilated = dilate(array, kernel=kernel, iterations=1, border=border)
    return erode(dilated, kernel=kernel, iterations=1, border=border)


def morphological_gradient(
    array: np.ndarray,
    kernel: np.ndarray | None = None,
) -> np.ndarray:
    dilated = dilate(array, kernel=kernel)
    eroded = erode(array, kernel=kernel)
    return dilated.astype(float) - eroded.astype(float)


def top_hat(array: np.ndarray, kernel: np.ndarray | None = None) -> np.ndarray:
    opened = opening(array, kernel=kernel)
    return array.astype(float) - opened.astype(float)


def black_hat(
    array: np.ndarray, kernel: np.ndarray | None = None
) -> np.ndarray:
    closed = closing(array, kernel=kernel)
    return closed.astype(float) - array.astype(float)


def _as_binary(mask: np.ndarray) -> np.ndarray:
    """Validate a boolean (or 0/1) mask and return it as ``bool``."""
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
        raise ValueError(
            "mask must be boolean or hold only 0 and 1"
        )
    return array.astype(bool)


def _as_hit_miss_kernel(kernel: np.ndarray, name: str) -> np.ndarray:
    """Validate a 2-D non-empty hit-or-miss kernel."""
    array = np.asarray(kernel, dtype=bool)
    if array.ndim != 2 or array.size == 0:
        raise ValueError(
            f"{name} must be a non-empty 2-D array, "
            f"got shape {np.asarray(kernel).shape}"
        )
    return array


def hit_or_miss(
    binary: np.ndarray,
    kernel_hit: np.ndarray,
    kernel_miss: np.ndarray | None = None,
) -> np.ndarray:
    """Apply the hit-or-miss transform with a kernel pair.

    A pixel is reported when ``kernel_hit`` fits the foreground and
    ``kernel_miss`` fits the background at the same position, i.e.
    ``erode(image, hit) & erode(~image, miss)``.  Positions covered by
    neither kernel are "don't care".

    Parameters
    ----------
    binary:
        Boolean (or 0/1) array of shape ``(H, W)``.
    kernel_hit:
        Foreground pattern that must fit the object.
    kernel_miss:
        Background pattern that must fit the background; defaults to
        the complement of ``kernel_hit``.

    Returns
    -------
    numpy.ndarray
        Boolean array of shape ``(H, W)``.

    Raises
    ------
    ValueError
        For a non-binary image, a bad kernel, differently shaped
        kernels or overlapping hit/miss positions.
    """
    image = _as_binary(binary)
    hit = _as_hit_miss_kernel(kernel_hit, "kernel_hit")
    if kernel_miss is None:
        miss = ~hit
    else:
        miss = _as_hit_miss_kernel(kernel_miss, "kernel_miss")
    if hit.shape != miss.shape:
        raise ValueError(
            f"kernel_hit shape {hit.shape} != kernel_miss "
            f"shape {miss.shape}"
        )
    if np.any(hit & miss):
        raise ValueError("kernel_hit and kernel_miss must not overlap")
    if not hit.any() or not miss.any():
        raise ValueError("both kernels must hold at least one True")
    return erode(image, kernel=hit) & erode(~image, kernel=miss)


_THIN_HIT = np.array(
    [[False, False, False], [False, True, False], [True, True, True]]
)
_THIN_MISS = np.array(
    [[True, True, True], [False, False, False], [False, False, False]]
)
_THICK_HIT = _THIN_MISS.copy()
_THICK_MISS = _THIN_HIT.copy()


def _rotated_pairs(
    hit: np.ndarray, miss: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return the four 90-degree rotations of a kernel pair."""
    return [
        (np.rot90(hit, turns).copy(), np.rot90(miss, turns).copy())
        for turns in range(4)
    ]


def thinning(binary: np.ndarray, max_iter: int = 100) -> np.ndarray:
    """Thin a binary object with sequential hit-or-miss peeling.

    One cycle applies the four rotations of the textbook south-peeling
    kernel pair and removes every matched pixel; cycles repeat until
    the image stops changing, so the result is idempotent and always a
    subset of the input.

    Parameters
    ----------
    binary:
        Boolean (or 0/1) array of shape ``(H, W)``.
    max_iter:
        Safety cap on the number of peeling cycles; must be >= 1.

    Returns
    -------
    numpy.ndarray
        Boolean thinned array of shape ``(H, W)``.

    Raises
    ------
    ValueError
        For a non-binary image or ``max_iter < 1``.
    """
    image = _as_binary(binary)
    cycles = int(max_iter)
    if cycles < 1:
        raise ValueError(f"max_iter must be >= 1, got {max_iter!r}")
    pairs = _rotated_pairs(_THIN_HIT, _THIN_MISS)
    current = image
    for _ in range(cycles):
        previous = current
        for hit, miss in pairs:
            current = current & ~hit_or_miss(current, hit, miss)
        if np.array_equal(current, previous):
            break
    return current


def thickening(binary: np.ndarray, max_iter: int = 100) -> np.ndarray:
    """Thicken a binary object with sequential hit-or-miss growing.

    This is the dual of :func:`thinning`: one cycle applies the four
    rotations of the swapped kernel pair and adds every matched
    background pixel, so the result is idempotent and always a
    superset of the input.

    Parameters
    ----------
    binary:
        Boolean (or 0/1) array of shape ``(H, W)``.
    max_iter:
        Safety cap on the number of growing cycles; must be >= 1.

    Returns
    -------
    numpy.ndarray
        Boolean thickened array of shape ``(H, W)``.

    Raises
    ------
    ValueError
        For a non-binary image or ``max_iter < 1``.
    """
    image = _as_binary(binary)
    cycles = int(max_iter)
    if cycles < 1:
        raise ValueError(f"max_iter must be >= 1, got {max_iter!r}")
    pairs = _rotated_pairs(_THICK_HIT, _THICK_MISS)
    current = image
    for _ in range(cycles):
        previous = current
        for hit, miss in pairs:
            current = current | hit_or_miss(current, hit, miss)
        if np.array_equal(current, previous):
            break
    return current


def clean_mask(
    mask: np.ndarray,
    open_radius: int = 2,
    close_radius: int = 4,
) -> np.ndarray:
    """Clean a foreground mask with an opening then a closing.

    The opening (with a disc of ``open_radius``) removes specks and
    thin bridges; the closing (with a disc of ``close_radius``) seals
    narrow gaps.  A radius of 0 skips that step.

    Parameters
    ----------
    mask:
        Boolean (or 0/1) array of shape ``(H, W)``.
    open_radius, close_radius:
        Non-negative disc radii in pixels.

    Returns
    -------
    numpy.ndarray
        Boolean array of shape ``(H, W)``.

    Raises
    ------
    ValueError
        For a non-binary mask or a negative radius.
    """
    image = _as_binary(mask)
    for name, radius in (
        ("open_radius", open_radius),
        ("close_radius", close_radius),
    ):
        if int(radius) < 0:
            raise ValueError(
                f"{name} must be >= 0, got {radius!r}"
            )
    current = image
    if int(open_radius) > 0:
        current = opening(
            current,
            kernel=structuring_element("disk", int(open_radius)),
        ).astype(bool)
    if int(close_radius) > 0:
        current = closing(
            current,
            kernel=structuring_element("disk", int(close_radius)),
        ).astype(bool)
    return current


def fill_holes(mask: np.ndarray, connectivity: int = 8) -> np.ndarray:
    """Fill every hole of a foreground mask, however large.

    The background components of the inverted mask that touch the
    image border stay background; every other background component is
    a hole and is filled.

    Parameters
    ----------
    mask:
        Boolean (or 0/1) array of shape ``(H, W)``.
    connectivity:
        4 or 8, used when labelling the background.

    Returns
    -------
    numpy.ndarray
        Boolean array of shape ``(H, W)``.

    Raises
    ------
    ValueError
        For a non-binary mask or a bad connectivity.
    """
    image = _as_binary(mask)
    if connectivity not in (4, 8):
        raise ValueError(
            f"connectivity must be 4 or 8, got {connectivity!r}"
        )
    labels, _ = label_components(~image, connectivity=connectivity)
    touching = border_touching_labels(labels)
    holes = (labels > 0) & ~np.isin(labels, sorted(touching))
    return image | holes


def remove_small_objects(
    mask: np.ndarray,
    min_size: int = 64,
    connectivity: int = 8,
) -> np.ndarray:
    """Drop every connected component smaller than ``min_size``.

    Parameters
    ----------
    mask:
        Boolean (or 0/1) array of shape ``(H, W)``.
    min_size:
        Minimum component area in pixels; must be >= 1.
    connectivity:
        4 or 8, used when labelling the foreground.

    Returns
    -------
    numpy.ndarray
        Boolean array of shape ``(H, W)``.

    Raises
    ------
    ValueError
        For a non-binary mask, ``min_size < 1`` or a bad
        connectivity.
    """
    image = _as_binary(mask)
    if int(min_size) < 1:
        raise ValueError(f"min_size must be >= 1, got {min_size!r}")
    if connectivity not in (4, 8):
        raise ValueError(
            f"connectivity must be 4 or 8, got {connectivity!r}"
        )
    labels, _ = label_components(image, connectivity=connectivity)
    sizes = np.bincount(labels.ravel())
    keep = set(
        int(label)
        for label, size in enumerate(sizes)
        if label > 0 and size >= int(min_size)
    )
    if not keep:
        return np.zeros(image.shape, dtype=bool)
    return np.isin(labels, sorted(keep))


def remove_objects_touching_border(
    mask: np.ndarray,
    connectivity: int = 8,
) -> np.ndarray:
    """Drop every connected component that touches the image border.

    Parameters
    ----------
    mask:
        Boolean (or 0/1) array of shape ``(H, W)``.
    connectivity:
        4 or 8, used when labelling the foreground.

    Returns
    -------
    numpy.ndarray
        Boolean array of shape ``(H, W)`` holding only interior
        components.

    Raises
    ------
    ValueError
        For a non-binary mask or a bad connectivity.
    """
    image = _as_binary(mask)
    if connectivity not in (4, 8):
        raise ValueError(
            f"connectivity must be 4 or 8, got {connectivity!r}"
        )
    labels, _ = label_components(image, connectivity=connectivity)
    touching = border_touching_labels(labels)
    if not touching:
        return image.copy()
    return image & ~np.isin(labels, sorted(touching))


def morphology_defaults(cfg: PipelineConfig) -> dict[str, int | float]:
    """Return the mask-cleanup parameters of a configuration.

    The helper keeps the refinement stage of Task 08 free of magic
    numbers.

    Parameters
    ----------
    cfg:
        The pipeline configuration.

    Returns
    -------
    dict
        Mapping with the opening/closing radii, the minimum object
        fraction, the hole-filling fraction and the number of kept
        components.
    """
    cfg.validate()
    return {
        "open_radius": int(cfg.morph_open_radius),
        "close_radius": int(cfg.morph_close_radius),
        "min_object_frac": float(cfg.min_object_frac),
        "fill_hole_frac": float(cfg.fill_hole_frac),
        "keep_largest": int(cfg.keep_largest),
    }
