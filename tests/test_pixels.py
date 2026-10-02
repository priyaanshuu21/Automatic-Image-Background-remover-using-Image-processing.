"""Tests for pixel relationships: neighbourhoods, distances, components."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from bgremover.pixels import (
    NEIGHBORS_4,
    NEIGHBORS_8,
    adjacency_demo,
    border_touching_labels,
    component_stats,
    distance,
    label_components,
    neighbor_offsets,
    neighbors,
)


def canonical_partition(labels: np.ndarray) -> np.ndarray:
    """Renumber a label image in raster order of first appearance."""
    flat = np.asarray(labels).ravel().tolist()
    mapping: dict[int, int] = {}
    out = [mapping.setdefault(value, len(mapping)) for value in flat]
    return np.array(out, dtype=np.int64).reshape(np.asarray(labels).shape)


def test_neighbor_offset_tables() -> None:
    """The 8-neighbour set is the 4-neighbour set plus the diagonals."""
    assert len(NEIGHBORS_4) == 4
    assert len(NEIGHBORS_8) == 8
    assert set(NEIGHBORS_4).issubset(set(NEIGHBORS_8))
    assert neighbor_offsets(4) == NEIGHBORS_4
    assert neighbor_offsets(8) == NEIGHBORS_8


def test_neighbors_at_corner() -> None:
    """A corner pixel has two 4-neighbours and three 8-neighbours."""
    assert len(neighbors((5, 5), 0, 0, 4)) == 2
    assert len(neighbors((5, 5), 0, 0, 8)) == 3


def test_neighbors_interior() -> None:
    """An interior pixel has a full neighbourhood."""
    assert len(neighbors((5, 5), 2, 2, 4)) == 4
    assert len(neighbors((5, 5), 2, 2, 8)) == 8


def test_neighbors_on_edges() -> None:
    """Edge pixels lose the neighbours outside the image."""
    assert len(neighbors((5, 7), 0, 3, 8)) == 5
    assert len(neighbors((5, 7), 2, 6, 8)) == 5
    assert sorted(neighbors((5, 7), 4, 0, 4)) == [(3, 0), (4, 1)]


def test_neighbors_rejects_bad_input() -> None:
    """Out-of-bounds pixels and bad connectivity are rejected."""
    with pytest.raises(ValueError):
        neighbors((5, 5), 5, 0)
    with pytest.raises(ValueError):
        neighbors((5, 5), 0, 0, 6)
    with pytest.raises(ValueError):
        neighbor_offsets(2)


@pytest.mark.parametrize(
    ("metric", "expected"),
    [("euclidean", 5.0), ("city_block", 7.0), ("chessboard", 4.0)],
)
def test_distance_metrics(metric: str, expected: float) -> None:
    """The 3-4-5 triangle gives the textbook answers."""
    assert distance((0, 0), (3, 4), metric) == pytest.approx(expected)


def test_distance_identity_and_rejection() -> None:
    """Identical points are zero apart; bad metrics are rejected."""
    assert distance((7, 7), (7, 7), "euclidean") == 0.0
    with pytest.raises(ValueError):
        distance((0, 0), (1, 1), "manhattan")
    with pytest.raises(ValueError):
        distance((0, 0), (1, 1, 1), "euclidean")
    with pytest.raises(ValueError):
        distance((), (), "euclidean")


def _diagonal_line() -> np.ndarray:
    mask = np.zeros((7, 7), dtype=bool)
    for index in range(7):
        mask[index, index] = True
    return mask


def test_diagonal_line_connectivity() -> None:
    """A diagonal chain is one 8-connected and seven 4-connected parts."""
    mask = _diagonal_line()
    _, count_8 = label_components(mask, connectivity=8)
    _, count_4 = label_components(mask, connectivity=4)
    assert count_8 == 1
    assert count_4 == 7


def test_label_components_empty_and_full() -> None:
    """An empty mask has no component; a full mask has exactly one."""
    labels, count = label_components(np.zeros((4, 5), dtype=bool))
    assert count == 0
    assert labels.dtype == np.int32
    assert not labels.any()
    labels, count = label_components(np.ones((4, 5), dtype=bool))
    assert count == 1
    assert labels.min() == 1
    assert labels.max() == 1


def test_label_components_labels_are_sequential() -> None:
    """Labels form the range ``1..n`` in raster order of first pixel."""
    mask = np.zeros((6, 6), dtype=bool)
    mask[4, 4] = True
    mask[1, 1] = True
    mask[5, 0] = True
    labels, count = label_components(mask, connectivity=8)
    assert count == 3
    assert labels[1, 1] == 1
    assert labels[4, 4] == 2
    assert labels[5, 0] == 3


@pytest.mark.parametrize("connectivity", [4, 8])
def test_label_components_matches_opencv(connectivity: int) -> None:
    """The two-pass labeller reproduces the OpenCV partition exactly."""
    rng = np.random.default_rng(11)
    for _ in range(10):
        mask = rng.random((40, 55)) < 0.18
        mine, count = label_components(mask, connectivity=connectivity)
        count_ref, reference = cv2.connectedComponents(
            mask.astype(np.uint8), connectivity=connectivity
        )
        # OpenCV counts the background label in its return value.
        assert count == count_ref - 1
        assert np.array_equal(
            canonical_partition(mine), canonical_partition(reference)
        )


def test_label_components_rejects_bad_input() -> None:
    """Non 2-D masks and bad connectivity are rejected."""
    with pytest.raises(ValueError):
        label_components(np.zeros((2, 2, 2), dtype=bool))
    with pytest.raises(ValueError):
        label_components(np.zeros((4, 4), dtype=bool), connectivity=5)


def test_component_stats_area_and_bbox() -> None:
    """Area sums to the foreground size and the bbox is inclusive."""
    mask = np.zeros((10, 12), dtype=bool)
    mask[2:5, 3:7] = True
    mask[8:10, 0:2] = True
    labels, count = label_components(mask)
    stats = component_stats(labels)
    assert count == 2
    assert len(stats) == 2
    total = sum(int(entry["area"]) for entry in stats)
    assert total == int(mask.sum())
    big = stats[0]
    assert big["bbox"] == (2, 3, 4, 6)
    assert big["centroid"] == pytest.approx((3.0, 4.5))
    small = stats[1]
    assert small["bbox"] == (8, 0, 9, 1)
    assert small["area"] == 4


def test_component_stats_empty() -> None:
    """An empty label image yields an empty statistics list."""
    assert component_stats(np.zeros((5, 5), dtype=np.int32)) == []


def test_component_stats_rejects_bad_input() -> None:
    """Wrong rank or dtype is rejected."""
    with pytest.raises(ValueError):
        component_stats(np.zeros((2, 2, 2), dtype=np.int32))
    with pytest.raises(ValueError):
        component_stats(np.zeros((4, 4), dtype=np.float32))


def test_border_touching_labels() -> None:
    """Only components reaching the border are reported."""
    mask = np.zeros((9, 9), dtype=bool)
    mask[0, 4] = True          # touches the top border
    mask[8, 8] = True          # touches the bottom-right corner
    mask[4, 4] = True          # fully enclosed
    mask[4, 4:6] = True
    mask[4, 4] = True
    labels, _ = label_components(mask)
    touching = border_touching_labels(labels)
    assert len(touching) == 2
    assert int(labels[0, 4]) in touching
    assert int(labels[8, 8]) in touching
    assert int(labels[4, 5]) not in touching


def test_border_touching_labels_empty() -> None:
    """An empty image has no border labels."""
    assert border_touching_labels(np.zeros((4, 4), dtype=np.int32)) == set()


def test_adjacency_demo_is_text() -> None:
    """The teaching helper returns a diagram, never printing it."""
    text = adjacency_demo((7, 9), 3, 4)
    assert isinstance(text, str)
    assert "o" in text
    # Exactly one '@' marks the pixel of interest inside the diagram.
    assert any(line.count("@") == 1 for line in text.splitlines())
    assert "legend" in text
    with pytest.raises(ValueError):
        adjacency_demo((7, 9), 7, 0)
