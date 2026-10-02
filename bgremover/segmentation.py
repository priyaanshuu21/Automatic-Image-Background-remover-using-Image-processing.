"""Thresholding, region growing, hysteresis and region merging.

Every function in this module is a direct, inspectable transcription of the
corresponding textbook technique.  Nothing is delegated to a library
implementation, and no randomness is involved, so results are reproducible.
"""

from __future__ import annotations

from collections import deque
from typing import Iterable

import numpy as np

from bgremover.config import PipelineConfig
from bgremover.histogram import compute_histogram
from bgremover.pixels import label_components, neighbor_offsets

__all__ = [
    "otsu_threshold",
    "otsu_between_class_variance",
    "otsu_mask",
    "otsu_on_lightness",
    "region_growing",
    "hysteresis_threshold",
    "region_means",
    "region_adjacency",
    "merge_regions",
    "merge_similar_regions",
    "segmentation_defaults",
    "describe_regions",
]


def _as_gray(gray: np.ndarray) -> np.ndarray:
    """Return a 2-D uint8 array or raise."""
    array = np.asarray(gray)
    if array.ndim != 2:
        raise ValueError(f"expected a 2-D gray image, got {array.shape}")
    if array.size == 0:
        raise ValueError("the image is empty")
    return array


def _as_field(image: np.ndarray) -> np.ndarray:
    """Reduce an image to a 2-D float32 field for scalar comparisons.

    Colour images are reduced with the mean of the channels so that a
    tolerance keeps the meaning "difference in intensity".

    Parameters
    ----------
    image:
        Array of shape ``(H, W)`` or ``(H, W, C)`` with ``C >= 1``.

    Returns
    -------
    numpy.ndarray
        Float32 array of shape ``(H, W)``.

    Raises
    ------
    ValueError
        For a bad ndim, an empty image or a zero width.
    """
    array = np.asarray(image)
    if array.ndim not in (2, 3) or array.size == 0:
        raise ValueError(
            f"expected a non-empty (H, W) or (H, W, C) image, got "
            f"{array.shape}"
        )
    if array.ndim == 2:
        return array.astype(np.float32)
    if array.shape[2] == 0:
        raise ValueError("the image has no channels")
    return array.astype(np.float32).mean(axis=2)


def _check_connectivity(connectivity: int) -> int:
    """Validate a connectivity value."""
    if connectivity not in (4, 8):
        raise ValueError(f"connectivity must be 4 or 8, got {connectivity}")
    return connectivity


def _check_ratio(low_ratio: float) -> float:
    """Validate a hysteresis ratio in ``(0, 1]``."""
    value = float(low_ratio)
    if not 0.0 < value <= 1.0:
        raise ValueError(f"low_ratio must lie in (0, 1], got {low_ratio}")
    return value


# ---------------------------------------------------------------- Otsu ------
def otsu_between_class_variance(
    gray: np.ndarray,
    bins: int = 256,
) -> np.ndarray:
    """Between-class variance for every candidate threshold.

    The classical Otsu criterion is the maximisation of
    ``sigma_b^2(t) = w0 * w1 * (mu0 - mu1)^2`` where ``w0`` and ``w1`` are the
    class probabilities below and above the candidate threshold ``t``.

    Parameters
    ----------
    gray:
        2-D uint8 image.
    bins:
        Number of histogram bins.

    Returns
    -------
    numpy.ndarray
        Float64 array of ``bins - 1`` values; index ``i`` corresponds to the
        candidate threshold ``i + 1``.

    Raises
    ------
    ValueError
        For a bad shape or ``bins < 2``.
    """
    image = _as_gray(gray)
    bins = int(bins)
    if bins < 2:
        raise ValueError(f"bins must be at least 2, got {bins}")
    counts = compute_histogram(image, bins=bins).astype(np.float64)
    levels = np.arange(bins, dtype=np.float64)
    total = counts.sum()
    if total <= 0:
        return np.zeros(bins - 1, dtype=np.float64)
    weight_low = np.cumsum(counts)[:-1]
    weight_high = total - weight_low
    sum_low = np.cumsum(counts * levels)[:-1]
    sum_high = np.cumsum(counts * levels)[-1] - sum_low
    valid = (weight_low > 0.0) & (weight_high > 0.0)
    mu_low = np.zeros(bins - 1, dtype=np.float64)
    mu_high = np.zeros(bins - 1, dtype=np.float64)
    np.divide(sum_low, weight_low, out=mu_low, where=valid)
    np.divide(sum_high, weight_high, out=mu_high, where=valid)
    variance = np.zeros(bins - 1, dtype=np.float64)
    difference = mu_low - mu_high
    variance[valid] = (
        weight_low[valid] * weight_high[valid] * difference[valid] ** 2
    )
    return variance


def otsu_threshold(gray: np.ndarray, bins: int = 256) -> int:
    """Return the Otsu threshold of a gray image.

    The threshold is the first intensity that maximises the between-class
    variance; ties therefore resolve to the lowest candidate, which keeps the
    result deterministic.

    Parameters
    ----------
    gray:
        2-D uint8 image.
    bins:
        Number of histogram bins.

    Returns
    -------
    int
        Threshold in ``0..255``.

    Raises
    ------
    ValueError
        For a bad shape or ``bins < 2``.
    """
    variance = otsu_between_class_variance(gray, bins=bins)
    return int(np.argmax(variance))


def otsu_mask(
    gray: np.ndarray,
    invert: bool = False,
    bins: int = 256,
) -> tuple[np.ndarray, int]:
    """Binarise a gray image with the Otsu threshold.

    Parameters
    ----------
    gray:
        2-D uint8 image.
    invert:
        Return ``gray < threshold`` instead of ``gray >= threshold``.
    bins:
        Number of histogram bins.

    Returns
    -------
    tuple
        The boolean mask and the threshold that produced it.

    Raises
    ------
    ValueError
        For a bad shape or ``bins < 2``.
    """
    image = _as_gray(gray)
    threshold = otsu_threshold(image, bins=bins)
    mask = image <= threshold if invert else image > threshold
    return mask, threshold


def otsu_on_lightness(
    lab: np.ndarray,
    invert: bool = False,
) -> tuple[np.ndarray, int]:
    """Otsu on the lightness channel of a CIELAB image.

    Splitting on lightness alone keeps shadows and highlights intact better
    than splitting on a gamma-encoded average of the channels.

    Parameters
    ----------
    lab:
        Array of shape ``(H, W, 3)`` as returned by
        :func:`bgremover.color.rgb_to_lab`.
    invert:
        Return the dark side instead of the bright side.

    Returns
    -------
    tuple
        The boolean mask of shape ``(H, W)`` and the lightness threshold.

    Raises
    ------
    ValueError
        For a bad shape.
    """
    array = np.asarray(lab)
    if array.ndim != 3 or array.shape[2] < 1:
        raise ValueError(
            f"expected an (H, W, C) array, got {array.shape}"
        )
    lightness = np.rint(np.clip(array[..., 0], 0.0, 255.0)).astype(np.uint8)
    return otsu_mask(lightness, invert=invert)


# ------------------------------------------------------- region growing ------
def _seed_positions(
    seeds: object,
    shape: tuple[int, int],
) -> list[tuple[int, int]]:
    """Normalise the accepted seed notations into ``(row, col)`` pairs."""
    height, width = shape

    def one(item: object) -> tuple[int, int]:
        if isinstance(item, (int, np.integer)):
            flat = int(item)
            if not 0 <= flat < height * width:
                raise ValueError(f"seed {flat} is outside the image")
            return divmod(flat, width)
        pair = np.asarray(item)
        if pair.shape != (2,):
            raise ValueError(
                f"a seed must be an int or a (row, col) pair, got {item!r}"
            )
        row, col = int(pair[0]), int(pair[1])
        if not (0 <= row < height and 0 <= col < width):
            raise ValueError(f"seed {(row, col)} is outside the image")
        return row, col

    if isinstance(seeds, (int, np.integer)):
        positions = [one(seeds)]
    elif isinstance(seeds, tuple) and len(seeds) == 2:
        try:
            first = seeds[0]
        except Exception:
            first = None
        if isinstance(first, (int, np.integer)) and not isinstance(
            first, bool
        ):
            positions = [one(seeds)]
        else:
            positions = [one(item) for item in seeds]
    elif isinstance(seeds, np.ndarray) and seeds.shape == (2,):
        positions = [one(seeds)]
    else:
        try:
            positions = [one(item) for item in seeds]
        except TypeError:
            raise ValueError(f"unsupported seeds value: {seeds!r}") from None
    return positions


def region_growing(
    seeds: object,
    image: np.ndarray,
    tolerance: float = 14.0,
    connectivity: int = 8,
    allowed: np.ndarray | None = None,
    reference: str = "seed",
) -> np.ndarray:
    """Grow a region outwards from one or several seeds.

    The classic breadth-first formulation: a candidate neighbour joins the
    region when the absolute difference between its value and the reference
    value is not larger than ``tolerance``.  The traversal order is the
    raster order of the queue, so the result is deterministic.

    Parameters
    ----------
    seeds:
        A flat index, a ``(row, col)`` pair, or an iterable of either.
    image:
        ``(H, W)`` or ``(H, W, C)`` image; colours are compared through the
        mean of the channels.
    tolerance:
        Maximum admissible difference, in intensity units.
    connectivity:
        4 or 8.
    allowed:
        Optional boolean mask; ``False`` pixels can never be added.
    reference:
        ``"seed"`` compares against the mean value of the seed pixels,
        ``"mean"`` against the running mean of the region.

    Returns
    -------
    numpy.ndarray
        Boolean array of shape ``(H, W)`` holding the region.

    Raises
    ------
    ValueError
        For an unsupported seed notation, a negative tolerance, a bad
        connectivity, an unknown ``reference`` or a mismatched ``allowed``.
    """
    field = _as_field(image)
    if float(tolerance) < 0.0:
        raise ValueError(f"tolerance must not be negative, got {tolerance}")
    _check_connectivity(connectivity)
    if reference not in ("seed", "mean"):
        raise ValueError(f"reference must be seed or mean, got {reference!r}")
    mask = np.zeros(field.shape, dtype=bool)
    if allowed is not None:
        gate = np.asarray(allowed)
        if gate.shape != field.shape:
            raise ValueError(
                f"allowed must have shape {field.shape}, got {gate.shape}"
            )
        if gate.dtype != np.bool_:
            gate = gate.astype(bool)
    else:
        gate = np.ones(field.shape, dtype=bool)

    positions = _seed_positions(seeds, field.shape)
    if not positions:
        return mask
    offsets = neighbor_offsets(connectivity)
    tolerance = float(tolerance)

    queue: deque[tuple[int, int]] = deque()
    # track running mean per connected region of seeds? but multiple seeds
    # treat all seeds as part of the same region (union). Initialize mean.
    seed_values = [float(field[r, c]) for r, c in positions]
    if reference == "seed":
        reference_value = float(np.mean(seed_values))
        total = 0.0
        count = 0.0
    else:
        reference_value = 0.0
        total = 0.0
        count = 0.0

    for row, col in positions:
        if not mask[row, col] and gate[row, col]:
            mask[row, col] = True
            total += float(field[row, col])
            count += 1.0
            queue.append((row, col))
    if reference == "mean" and count > 0:
        reference_value = total / count
    elif reference == "seed" and count == 0:
        pass
    height, width = field.shape
    while queue:
        row, col = queue.popleft()
        for delta_row, delta_col in offsets:
            next_row, next_col = row + delta_row, col + delta_col
            if not 0 <= next_row < height or not 0 <= next_col < width:
                continue
            if mask[next_row, next_col] or not gate[next_row, next_col]:
                continue
            if abs(float(field[next_row, next_col]) - reference_value) \
                    > tolerance:
                continue
            mask[next_row, next_col] = True
            total += float(field[next_row, next_col])
            count += 1.0
            if reference == "mean":
                reference_value = total / count
            queue.append((next_row, next_col))
    return mask


# --------------------------------------------------------- hysteresis -------
def hysteresis_threshold(
    gray: np.ndarray,
    low_ratio: float = 0.5,
    high: float | int | None = None,
    connectivity: int = 8,
) -> np.ndarray:
    """Canny style hysteresis thresholding.

    Pixels at or above ``high`` are definite edges.  Pixels between
    ``low_ratio * high`` and ``high`` are kept only when they form a connected
    chain of weak pixels that reaches a definite edge.

    Parameters
    ----------
    gray:
        2-D uint8 image.
    low_ratio:
        Fraction of ``high`` used as the weak level, in ``(0, 1]``.
    high:
        Definite level; the Otsu threshold is used when omitted.
    connectivity:
        4 or 8, used when following the weak chains.

    Returns
    -------
    numpy.ndarray
        Boolean array of shape ``(H, W)``.

    Raises
    ------
    ValueError
        For a bad shape, a ratio outside ``(0, 1]``, a bad connectivity or a
        high level outside ``(0, 255]``.
    """
    image = _as_gray(gray)
    ratio = _check_ratio(low_ratio)
    _check_connectivity(connectivity)
    if high is None:
        level = float(otsu_threshold(image))
    else:
        level = float(high)
        if not 0.0 < level <= 255.0:
            raise ValueError(f"high must lie in (0, 255], got {high}")
    weak_level = level * ratio
    weak = image >= np.ceil(weak_level)
    strong = image >= level
    if not strong.any() or not weak.any():
        return np.zeros(image.shape, dtype=bool)
    labels, count = label_components(weak, connectivity=connectivity)
    if count == 0:
        return np.zeros(image.shape, dtype=bool)
    keep = np.unique(labels[strong])
    keep = keep[keep > 0]
    if keep.size == 0:
        return np.zeros(image.shape, dtype=bool)
    return np.isin(labels, keep)


# ------------------------------------------------------- region merging ------
def region_means(
    labels: np.ndarray,
    image: np.ndarray,
) -> dict[int, np.ndarray]:
    """Mean colour of every labelled region.

    Parameters
    ----------
    labels:
        Integer label image of shape ``(H, W)``; ``0`` is the background.
    image:
        ``(H, W)`` or ``(H, W, C)`` image the labels refer to.

    Returns
    -------
    dict
        Mapping from label to mean colour as a float64 array.

    Raises
    ------
    ValueError
        For a bad shape or non-integer labels.
    """
    label_image = np.asarray(labels)
    if label_image.ndim != 2:
        raise ValueError(
            f"expected a 2-D label image, got {label_image.shape}"
        )
    if not np.issubdtype(label_image.dtype, np.integer):
        raise ValueError(f"labels must be integers, got {label_image.dtype}")
    values = np.asarray(image)
    if values.shape[:2] != label_image.shape:
        raise ValueError(
            f"image shape {values.shape[:2]} does not match labels "
            f"{label_image.shape}"
        )
    flat_labels = label_image.ravel()
    flat_values = values.reshape(flat_labels.size, -1).astype(np.float64)
    highest = int(flat_labels.max()) if flat_labels.size else 0
    counts = np.bincount(flat_labels, minlength=highest + 1)
    totals = np.stack(
        [
            np.bincount(
                flat_labels, weights=flat_values[:, channel],
                minlength=highest + 1,
            )
            for channel in range(flat_values.shape[1])
        ],
        axis=1,
    )
    nonzero = counts > 0
    means_array = np.zeros_like(totals)
    means_array[nonzero] = totals[nonzero] / counts[nonzero, None]
    return {
        label: means_array[label]
        for label in range(1, highest + 1)
        if counts[label] > 0
    }


def region_adjacency(
    labels: np.ndarray,
    connectivity: int = 4,
) -> dict[tuple[int, int], int]:
    """Count the touching pixel pairs of every pair of neighbouring regions.

    Parameters
    ----------
    labels:
        Integer label image of shape ``(H, W)``.
    connectivity:
        4 or 8.

    Returns
    -------
    dict
        Mapping from an ordered pair ``(a, b)`` with ``a < b`` to the number
        of neighbouring pixel pairs.  Insertion order follows the raster scan,
        which makes the dictionary reproducible.

    Raises
    ------
    ValueError
        For a bad shape, non-integer labels or a bad connectivity.
    """
    label_image = np.asarray(labels)
    if label_image.ndim != 2:
        raise ValueError(
            f"expected a 2-D label image, got {label_image.shape}"
        )
    if not np.issubdtype(label_image.dtype, np.integer):
        raise ValueError(f"labels must be integers, got {label_image.dtype}")
    _check_connectivity(connectivity)
    counts: dict[tuple[int, int], int] = {}
    for delta_row, delta_col in neighbor_offsets(connectivity):
        if delta_row < 0 or (delta_row == 0 and delta_col < 0):
            continue
        first = label_image[
            max(0, -delta_row): label_image.shape[0] - max(0, delta_row),
            max(0, -delta_col): label_image.shape[1] - max(0, delta_col),
        ]
        second = label_image[
            max(0, delta_row): label_image.shape[0] - max(0, -delta_row),
            max(0, delta_col): label_image.shape[1] - max(0, -delta_col),
        ]
        touching = (first != second) & (first > 0) & (second > 0)
        if not touching.any():
            continue
        left = first[touching].astype(np.int64)
        right = second[touching].astype(np.int64)
        low = np.minimum(left, right)
        high = np.maximum(left, right)
        pairs, weights = np.unique(
            np.stack([low, high], axis=1), axis=0, return_counts=True
        )
        for pair, weight in zip(pairs.tolist(), weights.tolist()):
            key = (int(pair[0]), int(pair[1]))
            counts[key] = counts.get(key, 0) + int(weight)
    return dict(sorted(counts.items()))


def merge_regions(
    labels: np.ndarray,
    pairs: Iterable[tuple[int, int]],
) -> np.ndarray:
    """Merge the requested label pairs with union-find.

    The merged labels are renumbered in the raster order of their first pixel,
    exactly like :func:`bgremover.pixels.label_components`, so the output is
    deterministic and forms the range ``1..k``.

    Parameters
    ----------
    labels:
        Integer label image of shape ``(H, W)``.
    pairs:
        Iterable of ``(a, b)`` label pairs to merge.  Pairs that mention an
        unknown label are ignored.

    Returns
    -------
    numpy.ndarray
        Integer array of shape ``(H, W)``.

    Raises
    ------
    ValueError
        For a bad shape, non-integer labels or a malformed pair.
    """
    label_image = np.asarray(labels)
    if label_image.ndim != 2:
        raise ValueError(
            f"expected a 2-D label image, got {label_image.shape}"
        )
    if not np.issubdtype(label_image.dtype, np.integer):
        raise ValueError(f"labels must be integers, got {label_image.dtype}")

    unique = np.unique(label_image)
    present = {int(value) for value in unique}
    parent: dict[int, int] = {int(value): int(value) for value in unique}

    def find(item: int) -> int:
        root = item
        while parent[root] != root:
            root = parent[root]
        while parent[item] != root:
            parent[item], item = root, parent[item]
        return root

    for pair in pairs:
        first, second = (int(value) for value in pair)
        if first not in present or second not in present:
            continue
        if first == second:
            continue
        root_a, root_b = find(first), find(second)
        if root_a == root_b:
            continue
        low, high = sorted((root_a, root_b))
        parent[high] = low

    highest = int(unique[-1]) if unique.size else 0
    canonical = np.zeros(highest + 1, dtype=np.int64)
    for value in unique:
        canonical[int(value)] = find(int(value))

    # Renumber the merged labels in the raster order of their first pixel.
    roots_flat = canonical[label_image.ravel()]
    foreground = roots_flat[roots_flat > 0]
    if foreground.size == 0:
        return np.zeros(label_image.shape, dtype=np.int32)
    _, first_index = np.unique(foreground, return_index=True)
    order = np.sort(first_index)
    lookup = np.zeros(highest + 1, dtype=np.int64)
    lookup[foreground[order]] = np.arange(1, order.size + 1, dtype=np.int64)
    return lookup[roots_flat].reshape(label_image.shape).astype(np.int32)


def merge_similar_regions(
    labels: np.ndarray,
    image: np.ndarray,
    tolerance: float = 10.0,
    connectivity: int = 4,
    max_passes: int = 8,
) -> np.ndarray:
    """Merge neighbouring regions of similar mean colour.

    Regions are merged when they touch and their mean colours differ by no
    more than ``tolerance`` in the mean channel.  Because a merge changes the
    mean colours, the criterion is reapplied up to ``max_passes`` times; the
    number of merges performed is finite, so the loop always terminates.

    Parameters
    ----------
    labels:
        Integer label image of shape ``(H, W)``.
    image:
        ``(H, W)`` or ``(H, W, C)`` image the labels refer to.
    tolerance:
        Maximum admissible mean-colour difference.
    connectivity:
        4 or 8.
    max_passes:
        Upper bound on the number of merging passes.

    Returns
    -------
    numpy.ndarray
        Merged integer labels of shape ``(H, W)``.

    Raises
    ------
    ValueError
        For a negative tolerance, ``max_passes < 1`` or a bad connectivity.
    """
    if float(tolerance) < 0.0:
        raise ValueError(f"tolerance must not be negative, got {tolerance}")
    if int(max_passes) < 1:
        raise ValueError(f"max_passes must be at least 1, got {max_passes}")
    _check_connectivity(connectivity)
    current = np.asarray(labels)
    if not np.issubdtype(current.dtype, np.integer):
        raise ValueError(f"labels must be integers, got {current.dtype}")
    for _ in range(int(max_passes)):
        means = region_means(current, image)
        adjacency = region_adjacency(current, connectivity=connectivity)
        pairs = [
            (first, second)
            for (first, second) in adjacency
            if first in means
            and second in means
            and float(
                np.abs(means[first] - means[second]).mean()
            ) <= float(tolerance)
        ]
        if not pairs:
            break
        current = merge_regions(current, pairs)
    return current


def segmentation_defaults(cfg: PipelineConfig) -> dict[str, float | int]:
    """Return the thresholding parameters of a configuration.

    The helper keeps the segmentation stage free of magic numbers and is used
    by the pipeline of Task 08.

    Parameters
    ----------
    cfg:
        The pipeline configuration.

    Returns
    -------
    dict
        Mapping with the growth tolerance, the growth connectivity, the edge
        barrier flag, the hysteresis ratio and the hysteresis levels.
    """
    _check_connectivity(cfg.grow_connectivity)
    _check_ratio(cfg.hysteresis_low_ratio)
    return {
        "grow_tolerance": float(cfg.grow_tolerance),
        "grow_connectivity": int(cfg.grow_connectivity),
        "use_edge_barrier": bool(cfg.use_edge_barrier),
        "hysteresis_low_ratio": float(cfg.hysteresis_low_ratio),
        "canny_low": float(cfg.canny_low),
        "canny_high": float(cfg.canny_high),
    }


def describe_regions(labels: np.ndarray, name: str = "labels") -> str:
    """Return a short ASCII map of a label image for teaching output.

    Parameters
    ----------
    labels:
        Integer label image of shape ``(H, W)``.
    name:
        Caption placed in front of the map.

    Returns
    -------
    str
        One character per pixel, cycling over ``.123456789`` and ``abcdef``.

    Raises
    ------
    ValueError
        For a bad shape or non-integer labels.
    """
    label_image = np.asarray(labels)
    if label_image.ndim != 2:
        raise ValueError(
            f"expected a 2-D label image, got {label_image.shape}"
        )
    if not np.issubdtype(label_image.dtype, np.integer):
        raise ValueError(f"labels must be integers, got {label_image.dtype}")
    alphabet = ".123456789abcdefghijklmnopqrstuvwxyz"
    lines = [name]
    for row in label_image:
        lines.append(
            "".join(
                alphabet[value] if 0 <= value < len(alphabet) else "?"
                for value in row.tolist()
            )
        )
    return "\n".join(lines)
