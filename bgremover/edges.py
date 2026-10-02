"""First- and second-derivative edge detection.

Every operator is a direct, inspectable transcription of the textbook
technique on top of :func:`bgremover.filters.correlate2d`: no Python loop
ever touches an individual pixel (non-maximum suppression, zero-crossing
detection and hysteresis tracking are all vectorised NumPy slicing), so
even the multi-stage Canny pipeline runs in well under a second on the
test fixtures.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from bgremover.filters import correlate2d, gaussian_kernel
from bgremover.pixels import label_components

__all__ = [
    "SOBEL_X",
    "SOBEL_Y",
    "PREWITT_X",
    "PREWITT_Y",
    "LAPLACIAN_4",
    "LAPLACIAN_8",
    "log_kernel",
    "gradients",
    "gradient_magnitude",
    "gradient_direction",
    "sobel_edges",
    "prewitt_edges",
    "laplacian_edges",
    "log_edges",
    "canny",
    "canny_stages",
    "edge_barrier",
    "overlay_edges",
]

SOBEL_X = np.array(
    [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
    dtype=np.float32,
)
SOBEL_Y = np.array(
    [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
    dtype=np.float32,
)
PREWITT_X = np.array(
    [[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]],
    dtype=np.float32,
)
PREWITT_Y = np.array(
    [[-1.0, -1.0, -1.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
    dtype=np.float32,
)
LAPLACIAN_4 = np.array(
    [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
    dtype=np.float32,
)
LAPLACIAN_8 = np.array(
    [[1.0, 1.0, 1.0], [1.0, -8.0, 1.0], [1.0, 1.0, 1.0]],
    dtype=np.float32,
)

# Responses at or below this magnitude (in gray levels) are treated as
# floating-point noise: the smallest genuine Sobel response to a
# one-gray-level step is 4.0, while float32 rounding on flat areas stays
# around 1e-5.
_NOISE_EPS = 1e-3


def _as_gray_float(gray: np.ndarray) -> np.ndarray:
    """Validate a 2-D gray image and return it as ``float32``."""
    array = np.asarray(gray)
    if array.ndim != 2:
        raise ValueError(
            f"expected a 2-D gray image, got shape {array.shape}"
        )
    if array.size == 0:
        raise ValueError("image must not be empty")
    return array.astype(np.float32)


def _as_kernel(kernel: np.ndarray) -> np.ndarray:
    """Validate a 2-D derivative kernel with odd side lengths."""
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


def _check_fraction(name: str, value: float) -> float:
    """Validate a Canny-style threshold fraction in ``(0, 1]``."""
    number = float(value)
    if not 0.0 < number <= 1.0:
        raise ValueError(
            f"{name} must lie in (0, 1], got {value!r}"
        )
    return number


def _check_level(threshold: float | None) -> float | None:
    """Validate an absolute edge strength threshold."""
    if threshold is None:
        return None
    level = float(threshold)
    if level < 0.0:
        raise ValueError(
            f"threshold must be >= 0, got {threshold!r}"
        )
    return level


def log_kernel(size: int = 7, sigma: float = 1.0) -> np.ndarray:
    """Return a Laplacian-of-Gaussian (LoG) kernel.

    ``LoG(x, y) = -1 / (pi sigma^4) * (1 - r^2 / (2 sigma^2))
    * exp(-r^2 / (2 sigma^2))`` with ``r^2 = x^2 + y^2``.  The mean is
    subtracted so the kernel sums to exactly zero and a flat image gives
    a flat (zero) response.

    Parameters
    ----------
    size:
        Odd side length, at least 3.
    sigma:
        Standard deviation in pixels; must be positive.

    Returns
    -------
    numpy.ndarray
        ``float32`` array of shape ``(size, size)`` summing to zero.

    Raises
    ------
    ValueError
        For an even (or too small) size or a non-positive sigma.
    """
    side = int(size)
    if side < 3 or side % 2 == 0:
        raise ValueError(
            f"size must be an odd integer >= 3, got {size!r}"
        )
    deviation = float(sigma)
    if deviation <= 0.0:
        raise ValueError(f"sigma must be > 0, got {sigma!r}")
    axis = np.arange(side, dtype=np.float64) - (side - 1) / 2.0
    grid_x, grid_y = np.meshgrid(axis, axis)
    squared = grid_x ** 2 + grid_y ** 2
    kernel = (
        -1.0
        / (np.pi * deviation ** 4)
        * (1.0 - squared / (2.0 * deviation ** 2))
        * np.exp(-squared / (2.0 * deviation ** 2))
    )
    kernel -= float(kernel.mean())
    return kernel.astype(np.float32)


def gradients(
    gray: np.ndarray,
    kernel_x: np.ndarray | None = None,
    kernel_y: np.ndarray | None = None,
    border: str = "reflect",
) -> tuple[np.ndarray, np.ndarray]:
    """Correlate a gray image with a pair of derivative kernels.

    Parameters
    ----------
    gray:
        2-D array of shape ``(H, W)``.
    kernel_x, kernel_y:
        Horizontal and vertical derivative kernels; default to
        :data:`SOBEL_X` and :data:`SOBEL_Y`.
    border:
        Border handling, see :func:`bgremover.filters.pad_image`.

    Returns
    -------
    tuple
        ``(gx, gy)`` as ``float32`` arrays of shape ``(H, W)``.

    Raises
    ------
    ValueError
        For a bad image, kernel or border mode.
    """
    image = _as_gray_float(gray)
    weights_x = _as_kernel(SOBEL_X if kernel_x is None else kernel_x)
    weights_y = _as_kernel(SOBEL_Y if kernel_y is None else kernel_y)
    gx = correlate2d(image, weights_x, border).astype(np.float32)
    gy = correlate2d(image, weights_y, border).astype(np.float32)
    return gx, gy


def gradient_magnitude(gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """Return ``sqrt(gx^2 + gy^2)`` element by element.

    Parameters
    ----------
    gx, gy:
        Gradient components of identical shape.

    Returns
    -------
    numpy.ndarray
        ``float32`` magnitude image.

    Raises
    ------
    ValueError
        If the shapes differ.
    """
    first = np.asarray(gx, dtype=np.float32)
    second = np.asarray(gy, dtype=np.float32)
    if first.shape != second.shape:
        raise ValueError(
            f"gx shape {first.shape} != gy shape {second.shape}"
        )
    return np.sqrt(first ** 2 + second ** 2).astype(np.float32)


def gradient_direction(gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """Return the gradient direction in degrees in ``[0, 360)``.

    Parameters
    ----------
    gx, gy:
        Gradient components of identical shape.

    Returns
    -------
    numpy.ndarray
        ``float32`` direction image; ``arctan2(gy, gx)`` mapped from
        ``(-180, 180]`` to ``[0, 360)``.

    Raises
    ------
    ValueError
        If the shapes differ.
    """
    first = np.asarray(gx, dtype=np.float32)
    second = np.asarray(gy, dtype=np.float32)
    if first.shape != second.shape:
        raise ValueError(
            f"gx shape {first.shape} != gy shape {second.shape}"
        )
    angle = np.degrees(np.arctan2(second, first)).astype(np.float32)
    return np.mod(angle + 360.0, 360.0).astype(np.float32)


def _magnitude_edges(
    gray: np.ndarray,
    kernel_x: np.ndarray,
    kernel_y: np.ndarray,
    threshold: float | None,
    border: str,
    default_ratio: float = 0.1,
) -> np.ndarray:
    """Threshold a gradient magnitude into a boolean edge mask."""
    level = _check_level(threshold)
    magnitude = gradient_magnitude(*gradients(gray, kernel_x, kernel_y,
                                              border))
    peak = float(magnitude.max())
    if peak <= _NOISE_EPS:
        return np.zeros(magnitude.shape, dtype=bool)
    if level is None:
        level = default_ratio * peak
    return magnitude >= level


def sobel_edges(
    gray: np.ndarray,
    threshold: float | None = None,
    border: str = "reflect",
) -> np.ndarray:
    """Detect edges with the Sobel magnitude.

    Pixels whose Sobel magnitude reaches ``threshold`` (or, when
    ``None``, 10% of the maximum magnitude) are edges.  A flat image
    has no edges.

    Parameters
    ----------
    gray:
        2-D array of shape ``(H, W)``.
    threshold:
        Absolute magnitude level; must be ``>= 0`` when given.
    border:
        Border handling, see :func:`bgremover.filters.pad_image`.

    Returns
    -------
    numpy.ndarray
        Boolean edge mask of shape ``(H, W)``.

    Raises
    ------
    ValueError
        For a bad image, threshold or border mode.
    """
    return _magnitude_edges(gray, SOBEL_X, SOBEL_Y, threshold, border)


def prewitt_edges(
    gray: np.ndarray,
    threshold: float | None = None,
    border: str = "reflect",
) -> np.ndarray:
    """Detect edges with the Prewitt magnitude.

    Parameters
    ----------
    gray:
        2-D array of shape ``(H, W)``.
    threshold:
        Absolute magnitude level; must be ``>= 0`` when given.
    border:
        Border handling, see :func:`bgremover.filters.pad_image`.

    Returns
    -------
    numpy.ndarray
        Boolean edge mask of shape ``(H, W)``.

    Raises
    ------
    ValueError
        For a bad image, threshold or border mode.
    """
    return _magnitude_edges(gray, PREWITT_X, PREWITT_Y, threshold, border)


def _zero_crossings(
    response: np.ndarray,
    connectivity: int = 8,
    threshold: float | None = None,
) -> np.ndarray:
    """Find zero crossings of a second-derivative response.

    The neighbour minima/maxima are gathered with vectorised slicing, so
    no Python loop touches a pixel.  A pixel is an edge when its
    neighbourhood straddles zero and the local span (max - min) reaches
    ``threshold`` (defaulting to 5% of the maximum absolute response).

    Parameters
    ----------
    response:
        2-D float response image.
    connectivity:
        4 or 8 neighbours to compare against.
    threshold:
        Minimum local span; must be ``>= 0`` when given.

    Returns
    -------
    numpy.ndarray
        Boolean edge mask of the same shape.

    Raises
    ------
    ValueError
        For a bad response, connectivity or threshold.
    """
    field = np.asarray(response, dtype=np.float32)
    if field.ndim != 2 or field.size == 0:
        raise ValueError(
            f"expected a non-empty 2-D response, got {field.shape}"
        )
    if connectivity not in (4, 8):
        raise ValueError(
            f"connectivity must be 4 or 8, got {connectivity!r}"
        )
    level = _check_level(threshold)
    peak = float(np.abs(field).max())
    if peak <= _NOISE_EPS:
        return np.zeros(field.shape, dtype=bool)
    if level is None:
        level = 0.05 * peak
    padded = np.pad(field, 1, mode="reflect")
    up = padded[:-2, 1:-1]
    down = padded[2:, 1:-1]
    left = padded[1:-1, :-2]
    right = padded[1:-1, 2:]
    neighbours = [up, down, left, right]
    if connectivity == 8:
        neighbours += [
            padded[:-2, :-2],
            padded[:-2, 2:],
            padded[2:, :-2],
            padded[2:, 2:],
        ]
    lowest = np.minimum.reduce(neighbours)
    highest = np.maximum.reduce(neighbours)
    straddles = (
        ((field > 0.0) & (lowest < 0.0))
        | ((field < 0.0) & (highest > 0.0))
        | ((field == 0.0) & (lowest < 0.0) & (highest > 0.0))
    )
    return straddles & ((highest - lowest) >= level)


def laplacian_edges(
    gray: np.ndarray,
    variant: int = 8,
    threshold: float | None = None,
    border: str = "reflect",
) -> np.ndarray:
    """Detect edges as Laplacian zero crossings.

    Parameters
    ----------
    gray:
        2-D array of shape ``(H, W)``.
    variant:
        4- or 8-neighbour Laplacian kernel.
    threshold:
        Minimum local span across the zero crossing; ``None`` uses 5%
        of the maximum absolute response.
    border:
        Border handling, see :func:`bgremover.filters.pad_image`.

    Returns
    -------
    numpy.ndarray
        Boolean edge mask of shape ``(H, W)``.

    Raises
    ------
    ValueError
        For a bad image, variant, threshold or border mode.
    """
    image = _as_gray_float(gray)
    if int(variant) == 4:
        kernel = LAPLACIAN_4
    elif int(variant) == 8:
        kernel = LAPLACIAN_8
    else:
        raise ValueError(f"variant must be 4 or 8, got {variant!r}")
    response = correlate2d(image, kernel, border)
    return _zero_crossings(response, int(variant), threshold)


def log_edges(
    gray: np.ndarray,
    sigma: float = 1.0,
    size: int | None = None,
    threshold: float | None = None,
    border: str = "reflect",
) -> np.ndarray:
    """Detect edges as LoG zero crossings.

    The image is correlated with a Laplacian-of-Gaussian kernel and the
    zero crossings of the response are reported, which combines
    smoothing and second-derivative detection in one step.

    Parameters
    ----------
    gray:
        2-D array of shape ``(H, W)``.
    sigma:
        LoG standard deviation in pixels; must be positive.
    size:
        Odd kernel side length; defaults to ``2 * ceil(3 * sigma) + 1``.
    threshold:
        Minimum local span across the zero crossing; ``None`` uses 5%
        of the maximum absolute response.
    border:
        Border handling, see :func:`bgremover.filters.pad_image`.

    Returns
    -------
    numpy.ndarray
        Boolean edge mask of shape ``(H, W)``.

    Raises
    ------
    ValueError
        For a bad image, sigma, size, threshold or border mode.
    """
    image = _as_gray_float(gray)
    deviation = float(sigma)
    if deviation <= 0.0:
        raise ValueError(f"sigma must be > 0, got {sigma!r}")
    if size is None:
        side = 2 * int(np.ceil(3.0 * deviation)) + 1
        side = max(side, 3)
    else:
        side = int(size)
    kernel = log_kernel(side, deviation)
    response = correlate2d(image, kernel, border)
    return _zero_crossings(response, 8, threshold)


def _non_maximum_suppression(
    magnitude: np.ndarray,
    direction: np.ndarray,
) -> np.ndarray:
    """Thin gradient ridges by keeping only local maxima.

    The direction image (degrees in ``[0, 360)``) is quantised into the
    four principal orientations and every pixel is compared against its
    two neighbours along the gradient with vectorised slicing.

    Parameters
    ----------
    magnitude:
        ``float32`` gradient magnitude of shape ``(H, W)``.
    direction:
        ``float32`` gradient direction in degrees of shape ``(H, W)``.

    Returns
    -------
    numpy.ndarray
        ``float32`` thinned magnitude; non-maxima are zero.
    """
    mag = np.asarray(magnitude, dtype=np.float32)
    angle = np.asarray(direction, dtype=np.float32)
    if mag.shape != angle.shape:
        raise ValueError(
            f"magnitude shape {mag.shape} != direction {angle.shape}"
        )
    padded = np.pad(mag, 1, mode="constant", constant_values=0.0)
    centre = padded[1:-1, 1:-1]
    north = padded[:-2, 1:-1]
    south = padded[2:, 1:-1]
    west = padded[1:-1, :-2]
    east = padded[1:-1, 2:]
    north_west = padded[:-2, :-2]
    north_east = padded[:-2, 2:]
    south_west = padded[2:, :-2]
    south_east = padded[2:, 2:]
    sector = np.mod(
        ((angle + 22.5) // 45).astype(np.int64), 4
    )
    first = np.where(
        sector == 0,
        west,
        np.where(
            sector == 1,
            north_east,
            np.where(sector == 2, north, north_west),
        ),
    )
    second = np.where(
        sector == 0,
        east,
        np.where(
            sector == 1,
            south_west,
            np.where(sector == 2, south, south_east),
        ),
    )
    keep = (centre >= first) & (centre >= second)
    return np.where(keep, centre, 0.0).astype(np.float32)


def canny_stages(
    gray: np.ndarray,
    low: float = 0.08,
    high: float = 0.20,
    sigma: float = 1.2,
    border: str = "reflect",
) -> dict[str, np.ndarray]:
    """Run the Canny pipeline and return every intermediate stage.

    The stages are: Gaussian smoothing, Sobel gradients, gradient
    magnitude, gradient direction, non-maximum suppression, strong and
    weak threshold masks, and the hysteresis-tracked edge mask.

    Parameters
    ----------
    gray:
        2-D array of shape ``(H, W)``.
    low, high:
        Weak/strong thresholds as fractions of the maximum thinned
        magnitude; ``0 < low < high <= 1``.
    sigma:
        Gaussian smoothing sigma in pixels; must be positive.
    border:
        Border handling, see :func:`bgremover.filters.pad_image`.

    Returns
    -------
    dict
        Mapping with keys ``"smoothed"``, ``"gx"``, ``"gy"``,
        ``"magnitude"``, ``"direction"``, ``"nms"``, ``"strong"``,
        ``"weak"`` and ``"edges"``.

    Raises
    ------
    ValueError
        For a bad image, threshold, sigma or border mode.
    """
    image = _as_gray_float(gray)
    low_ratio = _check_fraction("low", low)
    high_ratio = _check_fraction("high", high)
    if not low_ratio < high_ratio:
        raise ValueError(
            f"low ({low!r}) must be smaller than high ({high!r})"
        )
    deviation = float(sigma)
    if deviation <= 0.0:
        raise ValueError(f"sigma must be > 0, got {sigma!r}")
    side = max(2 * int(np.ceil(3.0 * deviation)) + 1, 3)
    smoothed = correlate2d(
        image, gaussian_kernel(side, deviation), border
    ).astype(np.float32)
    gx, gy = gradients(smoothed, SOBEL_X, SOBEL_Y, border)
    magnitude = gradient_magnitude(gx, gy)
    direction = gradient_direction(gx, gy)
    thinned = _non_maximum_suppression(magnitude, direction)
    peak = float(thinned.max())
    if peak <= _NOISE_EPS:
        empty = np.zeros(image.shape, dtype=bool)
        return {
            "smoothed": smoothed,
            "gx": gx,
            "gy": gy,
            "magnitude": magnitude,
            "direction": direction,
            "nms": thinned,
            "strong": empty,
            "weak": empty.copy(),
            "edges": empty.copy(),
        }
    strong = thinned >= high_ratio * peak
    weak = thinned >= low_ratio * peak
    labels, _ = label_components(weak, connectivity=8)
    keep = np.unique(labels[strong])
    keep = keep[keep > 0]
    edges = np.isin(labels, keep)
    return {
        "smoothed": smoothed,
        "gx": gx,
        "gy": gy,
        "magnitude": magnitude,
        "direction": direction,
        "nms": thinned,
        "strong": strong,
        "weak": weak,
        "edges": edges,
    }


def canny(
    gray: np.ndarray,
    low: float = 0.08,
    high: float = 0.20,
    sigma: float = 1.2,
    border: str = "reflect",
) -> np.ndarray:
    """Detect edges with the Canny pipeline.

    Parameters
    ----------
    gray:
        2-D array of shape ``(H, W)``.
    low, high:
        Weak/strong thresholds as fractions of the maximum thinned
        magnitude; ``0 < low < high <= 1`` (matching
        :class:`~bgremover.config.PipelineConfig` ``canny_low`` /
        ``canny_high``).
    sigma:
        Gaussian smoothing sigma in pixels; must be positive.
    border:
        Border handling, see :func:`bgremover.filters.pad_image`.

    Returns
    -------
    numpy.ndarray
        Boolean edge mask of shape ``(H, W)``.

    Raises
    ------
    ValueError
        For a bad image, threshold, sigma or border mode.
    """
    return canny_stages(gray, low, high, sigma, border)["edges"]


def edge_barrier(edges: np.ndarray, radius: int = 1) -> np.ndarray:
    """Dilate an edge mask into a barrier for region growing.

    Every ``True`` pixel within ``radius`` (Chebyshev distance) of an
    edge becomes part of the barrier, so a later region growing pass
    can use ``~barrier`` as its ``allowed`` mask and stop at edges.

    Parameters
    ----------
    edges:
        Boolean array of shape ``(H, W)``.
    radius:
        Non-negative dilation radius in pixels; ``0`` returns a copy.

    Returns
    -------
    numpy.ndarray
        Boolean barrier mask of shape ``(H, W)``.

    Raises
    ------
    ValueError
        For a non-2-D mask or a negative radius.
    """
    mask = np.asarray(edges)
    if mask.ndim != 2:
        raise ValueError(
            f"expected a 2-D edge mask, got shape {mask.shape}"
        )
    if mask.size == 0:
        raise ValueError("edge mask must not be empty")
    pixels = mask.astype(bool)
    side = int(radius)
    if side < 0:
        raise ValueError(f"radius must be >= 0, got {radius!r}")
    if side == 0:
        return pixels.copy()
    width = 2 * side + 1
    padded = np.pad(pixels, side, mode="reflect")
    windows = sliding_window_view(padded, (width, width))
    return np.any(windows, axis=(-2, -1))


def overlay_edges(
    image: np.ndarray,
    edges: np.ndarray,
    color: tuple[int, int, int] = (255, 0, 0),
) -> np.ndarray:
    """Paint an edge mask over an RGB image in a highlight colour.

    Parameters
    ----------
    image:
        ``uint8`` RGB array of shape ``(H, W, 3)``.
    edges:
        Boolean array of shape ``(H, W)``.
    color:
        ``(R, G, B)`` highlight, each component in ``0..255``.

    Returns
    -------
    numpy.ndarray
        New ``uint8`` RGB array; the input is not modified.

    Raises
    ------
    ValueError
        For a non-RGB image, a shape mismatch, a non-boolean mask or a
        bad colour.
    """
    rgb = np.asarray(image)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(
            f"expected an RGB image of shape (H, W, 3), got {rgb.shape}"
        )
    if rgb.dtype != np.uint8:
        raise ValueError(f"image must be uint8, got {rgb.dtype}")
    mask = np.asarray(edges)
    if mask.shape != rgb.shape[:2]:
        raise ValueError(
            f"edges shape {mask.shape} != image shape {rgb.shape[:2]}"
        )
    if mask.dtype != np.bool_:
        raise ValueError(f"edges must be boolean, got {mask.dtype}")
    channels = tuple(int(component) for component in color)
    if len(channels) != 3 or any(
        component < 0 or component > 255 for component in channels
    ):
        raise ValueError(
            f"color must hold three values in 0..255, got {color!r}"
        )
    painted = rgb.copy()
    painted[mask] = np.array(channels, dtype=np.uint8)
    return painted
