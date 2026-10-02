"""Pixel relationships: neighbourhoods, connectivity and components.

Everything the rest of the project needs to reason about *which pixels belong
together* lives here: the 4- and 8-neighbour offsets, the classical distance
metrics, deterministic connected-component labelling implemented with the
two-pass union-find algorithm, and the component statistics used by region
growing and by the mask clean-up stage.

The labelling routine deliberately avoids ``scipy``, ``skimage`` and
``cv2.connectedComponents``: the tests use OpenCV only as an oracle.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import NamedTuple

import numpy as np

LOGGER = logging.getLogger(__name__)

NEIGHBORS_4: tuple[tuple[int, int], ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))
NEIGHBORS_8: tuple[tuple[int, int], ...] = NEIGHBORS_4 + (
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1),
)

DISTANCE_METRICS = ("euclidean", "city_block", "chessboard")
_CONNECTIVITY_CHOICES = (4, 8)


def neighbor_offsets(connectivity: int = 8) -> tuple[tuple[int, int], ...]:
    """Return the neighbour offsets for a connectivity.

    Parameters
    ----------
    connectivity:
        4 for the four-sided neighbourhood, 8 to add the diagonals.

    Returns
    -------
    tuple of tuple
        ``(dy, dx)`` offsets.

    Raises
    ------
    ValueError
        If ``connectivity`` is neither 4 nor 8.
    """
    if connectivity == 4:
        return NEIGHBORS_4
    if connectivity == 8:
        return NEIGHBORS_8
    raise ValueError(
        f"connectivity must be 4 or 8, got {connectivity!r}"
    )


def neighbors(
    shape: tuple[int, int],
    y: int,
    x: int,
    connectivity: int = 8,
) -> list[tuple[int, int]]:
    """Return the in-bounds neighbours of the pixel ``(y, x)``.

    Parameters
    ----------
    shape:
        Image shape as ``(height, width)``.
    y, x:
        Row and column of the pixel.
    connectivity:
        4 or 8.

    Returns
    -------
    list of tuple
        ``(row, column)`` pairs inside the image, in a deterministic order.

    Raises
    ------
    ValueError
        For an unsupported connectivity or an out-of-bounds pixel.
    """
    offsets = neighbor_offsets(connectivity)
    height, width = int(shape[0]), int(shape[1])
    if not 0 <= y < height or not 0 <= x < width:
        raise ValueError(
            f"pixel ({y}, {x}) is outside the image of shape {shape}"
        )
    found: list[tuple[int, int]] = []
    for dy, dx in offsets:
        ny, nx = y + dy, x + dx
        if 0 <= ny < height and 0 <= nx < width:
            found.append((ny, nx))
    return found


def distance(
    point_p: Iterable[float],
    point_q: Iterable[float],
    metric: str = "euclidean",
) -> float:
    """Return the distance between two points under a discrete metric.

    Parameters
    ----------
    point_p, point_q:
        Two sequences of coordinates of equal length.
    metric:
        ``"euclidean"`` (``sqrt(sum d^2)``), ``"city_block"`` (D4,
        ``sum |d|``) or ``"chessboard"`` (D8, ``max |d|``).

    Returns
    -------
    float
        The distance; 0.0 for identical points.

    Raises
    ------
    ValueError
        For an unknown metric or mismatched coordinates.
    """
    if metric not in DISTANCE_METRICS:
        raise ValueError(
            f"metric must be one of {DISTANCE_METRICS}, got {metric!r}"
        )
    first = [float(v) for v in point_p]
    second = [float(v) for v in point_q]
    if len(first) != len(second) or not first:
        raise ValueError(
            "points must be non-empty sequences of equal length"
        )
    diffs = [a - b for a, b in zip(first, second)]
    if metric == "euclidean":
        return float(np.sqrt(sum(d * d for d in diffs)))
    if metric == "city_block":
        return float(sum(abs(d) for d in diffs))
    return float(max(abs(d) for d in diffs))


class _UnionFind:
    """Minimal union-find with path compression, used by the labeller."""

    def __init__(self) -> None:
        self.parent: list[int] = [0]

    def make_set(self) -> int:
        """Create and return a new singleton label."""
        label = len(self.parent)
        self.parent.append(label)
        return label

    def find(self, item: int) -> int:
        """Return the representative of ``item``."""
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != root:
            self.parent[item], item = root, self.parent[item]
        return root

    def union(self, first: int, second: int) -> None:
        """Merge two sets, keeping the smaller label as representative."""
        root_a = self.find(first)
        root_b = self.find(second)
        if root_a == root_b:
            return
        if root_b < root_a:
            root_a, root_b = root_b, root_a
        self.parent[root_b] = root_a


def label_components(
    mask: np.ndarray,
    connectivity: int = 8,
) -> tuple[np.ndarray, int]:
    """Label the connected components of a binary mask.

    The classical two-pass algorithm is used: a provisional label is created
    for every foreground pixel that has no labelled neighbour, otherwise the
    smallest provisional neighbour label is reused and the candidates are
    merged with union-find.  A final pass resolves every label and renumbers
    the classes in the raster order of their first pixel, so the result is
    deterministic and labels always form the range ``1..n`` with ``0`` for the
    background.

    Parameters
    ----------
    mask:
        Boolean or 0/1 array of shape ``(H, W)``.
    connectivity:
        4 or 8.

    Returns
    -------
    tuple
        ``(labels, count)`` where ``labels`` is an ``int32`` array of the
        same shape as ``mask``.

    Raises
    ------
    ValueError
        If the mask is not 2-D or the connectivity is invalid.

    Notes
    -----
    Only foreground pixels are visited, so the cost is
    ``O(H * W + n_foreground)``.
    """
    offsets = neighbor_offsets(connectivity)
    array = np.asarray(mask, dtype=bool)
    if array.ndim != 2:
        raise ValueError(
            f"label_components expects a 2-D mask, got shape {array.shape}"
        )
    height, width = array.shape
    labels = np.zeros((height, width), dtype=np.int32)
    groups = _UnionFind()
    rows, cols = np.nonzero(array)
    for y, x in zip(rows.tolist(), cols.tolist()):
        candidates: list[int] = []
        for dy, dx in offsets:
            ny, nx = y + dy, x + dx
            if ny < 0 or ny >= height or nx < 0 or nx >= width:
                continue
            existing = int(labels[ny, nx])
            if existing > 0:
                candidates.append(existing)
        if not candidates:
            labels[y, x] = groups.make_set()
            continue
        smallest = min(candidates)
        labels[y, x] = smallest
        for candidate in candidates:
            groups.union(smallest, candidate)

    if len(groups.parent) == 1:
        return labels, 0

    remap: dict[int, int] = {}
    flat = labels.ravel()
    for index in range(flat.size):
        label = int(flat[index])
        if label == 0:
            continue
        root = groups.find(label)
        new_label = remap.get(root)
        if new_label is None:
            new_label = len(remap) + 1
            remap[root] = new_label
        flat[index] = new_label
    count = len(remap)
    LOGGER.debug("labelled %d components", count)
    return flat.reshape(height, width), count


class ComponentStats(NamedTuple):
    """Statistics of one connected component."""

    label: int
    area: int
    bbox: tuple[int, int, int, int]
    centroid: tuple[float, float]


def component_stats(labels: np.ndarray) -> list[dict[str, object]]:
    """Return per-component area, bounding box and centroid.

    Parameters
    ----------
    labels:
        Label image as produced by :func:`label_components`.

    Returns
    -------
    list of dict
        One dictionary per label with keys ``label``, ``area``,
        ``bbox`` (``y0, x0, y1, x1``, inclusive) and ``centroid``
        (``row, column`` as floats), ordered by ascending label.

    Raises
    ------
    ValueError
        If ``labels`` is not a 2-D integer array.
    """
    array = np.asarray(labels)
    if array.ndim != 2:
        raise ValueError(
            f"component_stats expects a 2-D array, got {array.shape}"
        )
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError(
            f"labels must be an integer array, got {array.dtype}"
        )
    height, width = array.shape
    flat = array.astype(np.int64, copy=False).ravel()
    stats: list[dict[str, object]] = []
    positives = np.flatnonzero(flat > 0)
    if positives.size == 0:
        return stats
    values = flat[positives]
    order = np.argsort(values, kind="stable")
    ordered_idx = positives[order]
    ordered_val = values[order]
    splits = np.flatnonzero(np.diff(ordered_val)) + 1
    split_points = [0] + splits.tolist()
    for chunk_index, chunk_val in enumerate(
        np.split(ordered_idx, splits)
    ):
        label = int(ordered_val[split_points[chunk_index]])
        rows = chunk_val // width
        cols = chunk_val % width
        stats.append(
            {
                "label": label,
                "area": int(chunk_val.size),
                "bbox": (
                    int(rows.min()),
                    int(cols.min()),
                    int(rows.max()),
                    int(cols.max()),
                ),
                "centroid": (
                    float(rows.mean()),
                    float(cols.mean()),
                ),
            }
        )
    return stats


def border_touching_labels(labels: np.ndarray) -> set[int]:
    """Return the labels that have at least one pixel on the image border.

    Parameters
    ----------
    labels:
        Label image as produced by :func:`label_components`.

    Returns
    -------
    set of int
        Non-zero labels touching the first or last row or column.  The
        background label 0 is never reported.
    """
    array = np.asarray(labels)
    if array.ndim != 2:
        raise ValueError(
            f"expected a 2-D label image, got shape {array.shape}"
        )
    if array.size == 0:
        return set()
    border = np.concatenate(
        (
            array[0, :].ravel(),
            array[-1, :].ravel(),
            array[:, 0].ravel(),
            array[:, -1].ravel(),
        )
    )
    return {int(v) for v in np.unique(border) if int(v) > 0}


def adjacency_demo(
    shape: tuple[int, int],
    y: int,
    x: int,
) -> str:
    """Return an ASCII rendering of the 4- and 8-neighbourhoods.

    Parameters
    ----------
    shape:
        Image shape as ``(height, width)``.
    y, x:
        Row and column of the pixel of interest.

    Returns
    -------
    str
        A multi-line teaching diagram; the centre pixel is ``@``, the
        neighbourhood members are ``o`` and everything else is ``.``.
        The function only returns the string, it never prints.

    Raises
    ------
    ValueError
        If ``(y, x)`` lies outside the image.
    """
    height, width = int(shape[0]), int(shape[1])
    if not 0 <= y < height or not 0 <= x < width:
        raise ValueError(
            f"pixel ({y}, {x}) is outside the image of shape {shape}"
        )
    set_4 = set(neighbors(shape, y, x, 4))
    set_8 = set(neighbors(shape, y, x, 8))
    blocks: list[list[str]] = []
    for membership in (set_4, set_8):
        block: list[str] = []
        for dy in range(-2, 3):
            row: list[str] = []
            for dx in range(-2, 3):
                if dy == 0 and dx == 0:
                    row.append("@")
                elif (y + dy, x + dx) in membership:
                    row.append("o")
                else:
                    row.append(".")
            block.append("".join(row))
        blocks.append(block)
    lines = [
        f"neighbourhood of (y={y}, x={x}) in an image of shape {shape}",
        "4-connectivity (NEIGHBORS_4)     8-connectivity (NEIGHBORS_8)",
    ]
    for row_4, row_8 in zip(blocks[0], blocks[1]):
        lines.append(f"{row_4}          {row_8}")
    lines.append("legend: '@' = pixel, 'o' = neighbour, '.' = outside set")
    return "\n".join(lines)
