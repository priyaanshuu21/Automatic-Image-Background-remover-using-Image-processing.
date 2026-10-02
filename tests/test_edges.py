"""Tests for first- and second-derivative edge detection."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from bgremover.edges import (
    LAPLACIAN_4,
    LAPLACIAN_8,
    PREWITT_X,
    PREWITT_Y,
    SOBEL_X,
    SOBEL_Y,
    canny,
    canny_stages,
    edge_barrier,
    gradient_direction,
    gradient_magnitude,
    gradients,
    laplacian_edges,
    log_edges,
    log_kernel,
    overlay_edges,
    prewitt_edges,
    sobel_edges,
)
from bgremover.filters import correlate2d


def _step(height: int = 48, width: int = 48) -> np.ndarray:
    """Return a vertical step edge, dark left and bright right."""
    image = np.zeros((height, width), dtype=np.uint8)
    image[:, width // 2 :] = 200
    return image


def test_kernels_have_expected_values() -> None:
    """The textbook kernels are stored exactly."""
    assert SOBEL_X.tolist() == [
        [-1.0, 0.0, 1.0],
        [-2.0, 0.0, 2.0],
        [-1.0, 0.0, 1.0],
    ]
    assert SOBEL_Y.tolist() == [
        [-1.0, -2.0, -1.0],
        [0.0, 0.0, 0.0],
        [1.0, 2.0, 1.0],
    ]
    assert PREWITT_X.tolist() == [
        [-1.0, 0.0, 1.0],
        [-1.0, 0.0, 1.0],
        [-1.0, 0.0, 1.0],
    ]
    assert PREWITT_Y.tolist() == [
        [-1.0, -1.0, -1.0],
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
    ]
    assert LAPLACIAN_4.tolist() == [
        [0.0, 1.0, 0.0],
        [1.0, -4.0, 1.0],
        [0.0, 1.0, 0.0],
    ]
    assert LAPLACIAN_8.tolist() == [
        [1.0, 1.0, 1.0],
        [1.0, -8.0, 1.0],
        [1.0, 1.0, 1.0],
    ]
    for kernel in (
        SOBEL_X,
        SOBEL_Y,
        PREWITT_X,
        PREWITT_Y,
        LAPLACIAN_4,
        LAPLACIAN_8,
    ):
        assert kernel.dtype == np.float32
        assert kernel.shape == (3, 3)


def test_log_kernel_properties() -> None:
    """The LoG kernel is symmetric, zero-sum and sigma sensitive."""
    kernel = log_kernel(7, 1.0)
    assert kernel.shape == (7, 7)
    assert kernel.dtype == np.float32
    assert abs(float(kernel.sum())) < 1e-5
    assert np.allclose(kernel, kernel[::-1, ::-1])
    assert np.allclose(kernel, kernel.T)
    wider = log_kernel(9, 2.0)
    assert wider.shape == (9, 9)
    assert abs(float(wider.sum())) < 1e-5
    with pytest.raises(ValueError):
        log_kernel(4, 1.0)
    with pytest.raises(ValueError):
        log_kernel(7, 0.0)


def test_gradients_match_opencv_sobel() -> None:
    """Sobel responses agree exactly with OpenCV."""
    gray = _step()
    gx, gy = gradients(gray)
    assert gx.shape == gray.shape
    assert gy.shape == gray.shape
    assert gx.dtype == np.float32
    expected_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    expected_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    assert np.allclose(gx, expected_x, atol=1e-3)
    assert np.allclose(gy, expected_y, atol=1e-3)


def test_gradients_reject_bad_input() -> None:
    """Non-2-D images and bad kernels are rejected."""
    with pytest.raises(ValueError):
        gradients(np.zeros((4, 4, 3), dtype=np.uint8))
    with pytest.raises(ValueError):
        gradients(np.zeros((0, 0), dtype=np.uint8))
    with pytest.raises(ValueError):
        gradients(_step(), kernel_x=np.ones((2, 2)))
    with pytest.raises(ValueError):
        gradients(_step(), border="nope")


def test_gradient_magnitude_and_direction() -> None:
    """Magnitude follows Pythagoras and direction spans [0, 360)."""
    gx = np.array([[3.0, 0.0]], dtype=np.float32)
    gy = np.array([[4.0, -1.0]], dtype=np.float32)
    magnitude = gradient_magnitude(gx, gy)
    assert np.allclose(
        magnitude, np.array([[5.0, 1.0]], dtype=np.float32)
    )
    direction = gradient_direction(gx, gy)
    assert direction.shape == (1, 2)
    assert float(direction.min()) >= 0.0
    assert float(direction.max()) < 360.0
    assert direction[0, 0] == pytest.approx(53.130, abs=1e-2)
    assert direction[0, 1] == pytest.approx(270.0, abs=1e-6)
    with pytest.raises(ValueError):
        gradient_magnitude(gx, np.zeros((3, 3), dtype=np.float32))
    with pytest.raises(ValueError):
        gradient_direction(gx, np.zeros((3, 3), dtype=np.float32))


def test_sobel_edges_find_the_step() -> None:
    """The step edge is reported next to the gray-level jump."""
    gray = _step()
    edges = sobel_edges(gray)
    assert edges.dtype == np.bool_
    assert edges.shape == gray.shape
    assert int(edges.sum()) > 0
    assert int(edges.sum()) < gray.size // 4
    band = np.zeros_like(edges)
    band[:, 22:26] = True
    assert np.count_nonzero(edges & band) > 0
    far = np.zeros_like(edges)
    far[:, :10] = True
    far[:, 38:] = True
    assert np.count_nonzero(edges & far) == 0


def test_sobel_edges_threshold_and_flat() -> None:
    """Thresholds behave monotonically; flat images have no edges."""
    gray = _step()
    flat = np.full((16, 16), 90, dtype=np.uint8)
    assert not sobel_edges(flat).any()
    assert not prewitt_edges(flat).any()
    loose = sobel_edges(gray, threshold=1.0)
    strict = sobel_edges(gray, threshold=1000.0)
    assert int(loose.sum()) >= int(sobel_edges(gray).sum())
    assert not strict.any()
    with pytest.raises(ValueError):
        sobel_edges(gray, threshold=-1.0)


def test_prewitt_edges_find_the_step() -> None:
    """Prewitt also fires at the step and stays quiet far away."""
    gray = _step()
    edges = prewitt_edges(gray)
    assert edges.dtype == np.bool_
    band = np.zeros_like(edges)
    band[:, 22:26] = True
    assert np.count_nonzero(edges & band) > 0
    far = np.zeros_like(edges)
    far[:, :10] = True
    assert np.count_nonzero(edges & far) == 0


def test_laplacian_response_matches_opencv() -> None:
    """The 4-neighbour Laplacian agrees with OpenCV."""
    gray = _step()
    mine = correlate2d(gray.astype(np.float32), LAPLACIAN_4)
    expected = cv2.Laplacian(gray, cv2.CV_32F, ksize=1)
    assert np.allclose(mine, expected, atol=1e-3)


def test_laplacian_edges_straddle_the_step() -> None:
    """Zero crossings appear on both sides of the jump."""
    gray = _step()
    for variant in (4, 8):
        edges = laplacian_edges(gray, variant=variant)
        assert edges.dtype == np.bool_
        assert edges.shape == gray.shape
        left = np.count_nonzero(edges[:, :24])
        right = np.count_nonzero(edges[:, 24:])
        assert left > 0 and right > 0
    assert not laplacian_edges(
        np.full((16, 16), 90, dtype=np.uint8)
    ).any()
    with pytest.raises(ValueError):
        laplacian_edges(gray, variant=6)


def test_log_edges_find_a_disc() -> None:
    """LoG zero crossings ring a bright disc."""
    height, width = 64, 64
    grid_y = np.arange(height)[:, None]
    grid_x = np.arange(width)[None, :]
    disc = (grid_y - 31.5) ** 2 + (grid_x - 31.5) ** 2 <= 12.0 ** 2
    gray = np.where(disc, 200, 40).astype(np.uint8)
    edges = log_edges(gray, sigma=1.0)
    assert edges.dtype == np.bool_
    assert 20 < int(edges.sum()) < 600
    ring = np.zeros_like(edges)
    distance = np.sqrt(
        (grid_y - 31.5) ** 2 + (grid_x - 31.5) ** 2
    )
    ring[np.abs(distance - 12.0) <= 2.5] = True
    assert np.count_nonzero(edges & ring) / max(
        int(edges.sum()), 1
    ) > 0.5
    with pytest.raises(ValueError):
        log_edges(gray, sigma=0.0)
    with pytest.raises(ValueError):
        log_edges(gray, size=4)


def test_canny_finds_a_clean_rectangle() -> None:
    """Canny traces most of a rectangle perimeter exactly once."""
    gray = np.zeros((48, 48), dtype=np.uint8)
    gray[12:36, 12:36] = 200
    edges = canny(gray)
    assert edges.dtype == np.bool_
    assert edges.shape == gray.shape
    assert 60 < int(edges.sum()) < 200
    assert not canny(np.full((24, 24), 90, dtype=np.uint8)).any()


def test_canny_is_deterministic_and_monotone() -> None:
    """Repeated runs agree; a higher bar keeps fewer edges."""
    gray = _step()
    first = canny(gray)
    assert np.array_equal(first, canny(gray))
    strict = canny(gray, low=0.3, high=0.6)
    assert int(strict.sum()) <= int(first.sum())


def test_canny_rejects_bad_thresholds() -> None:
    """Fractions outside (0, 1] or inverted pairs are rejected."""
    gray = _step()
    with pytest.raises(ValueError):
        canny(gray, low=0.0, high=0.2)
    with pytest.raises(ValueError):
        canny(gray, low=0.5, high=0.5)
    with pytest.raises(ValueError):
        canny(gray, low=0.6, high=0.2)
    with pytest.raises(ValueError):
        canny(gray, low=0.08, high=1.5)
    with pytest.raises(ValueError):
        canny(gray, sigma=0.0)


def test_canny_stages_exposes_intermediates() -> None:
    """Every documented stage is present with a consistent shape."""
    stages = canny_stages(_step())
    assert set(stages) == {
        "smoothed",
        "gx",
        "gy",
        "magnitude",
        "direction",
        "nms",
        "strong",
        "weak",
        "edges",
    }
    shape = (48, 48)
    for name in (
        "smoothed",
        "gx",
        "gy",
        "magnitude",
        "direction",
        "nms",
    ):
        assert stages[name].shape == shape
        assert stages[name].dtype == np.float32
    for name in ("strong", "weak", "edges"):
        assert stages[name].shape == shape
        assert stages[name].dtype == np.bool_
    assert np.array_equal(stages["edges"], canny(_step()))
    assert int(stages["weak"].sum()) >= int(stages["strong"].sum())
    assert int(stages["strong"].sum()) >= int(stages["edges"].sum())


def test_canny_marks_the_circle_boundary(circle_image) -> None:
    """Most Canny responses sit on the ground-truth disc boundary."""
    gray = circle_image.mean(axis=2).astype(np.uint8)
    edges = canny(gray)
    assert 100 < int(edges.sum()) < 5000
    rows = np.arange(gray.shape[0])[:, None]
    cols = np.arange(gray.shape[1])[None, :]
    distance = np.sqrt(
        (rows - 59.5) ** 2 + (cols - 79.5) ** 2
    )
    ring = np.abs(distance - 35.0) <= 3.0
    assert np.count_nonzero(edges & ring) / max(
        int(edges.sum()), 1
    ) > 0.5


def test_edge_barrier_dilates_edges() -> None:
    """The barrier covers the edges plus a Chebyshev neighbourhood."""
    edges = np.zeros((16, 16), dtype=bool)
    edges[8, 8] = True
    assert np.array_equal(edge_barrier(edges, radius=0), edges)
    barrier = edge_barrier(edges, radius=1)
    assert barrier.shape == edges.shape
    assert int(barrier.sum()) == 9
    wider = edge_barrier(edges, radius=2)
    assert int(wider.sum()) == 25
    assert np.all(wider[barrier])
    with pytest.raises(ValueError):
        edge_barrier(edges, radius=-1)
    with pytest.raises(ValueError):
        edge_barrier(np.zeros((4, 4, 3), dtype=bool))


def test_edge_barrier_blocks_region_growing(circle_mask) -> None:
    """A dilated boundary stops growth from leaking into the disc."""
    from bgremover.segmentation import region_growing

    gray = np.where(circle_mask, 200, 40).astype(np.uint8)
    boundary = laplacian_edges(gray)
    barrier = edge_barrier(boundary, radius=1)
    grown = region_growing(
        (0, 0), gray, tolerance=200.0, allowed=~barrier
    )
    assert grown[0, 0]
    assert int(np.count_nonzero(grown & circle_mask)) == 0


def test_overlay_edges_paints_only_edges(rect_image) -> None:
    """Edge pixels take the highlight colour; the rest is untouched."""
    gray = rect_image.mean(axis=2).astype(np.uint8)
    edges = canny(gray)
    painted = overlay_edges(rect_image, edges, color=(255, 0, 0))
    assert painted.shape == rect_image.shape
    assert painted.dtype == np.uint8
    assert np.array_equal(painted[~edges], rect_image[~edges])
    assert np.all(painted[edges] == np.array([255, 0, 0]))
    with pytest.raises(ValueError):
        overlay_edges(rect_image[:, :, 0], edges)
    with pytest.raises(ValueError):
        overlay_edges(rect_image, edges[:-1])
    with pytest.raises(ValueError):
        overlay_edges(rect_image, edges.astype(np.uint8))
    with pytest.raises(ValueError):
        overlay_edges(rect_image, edges, color=(256, 0, 0))
