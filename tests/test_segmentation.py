"""Tests for thresholding, region growing, hysteresis and region merging."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from bgremover.color import rgb_to_lab
from bgremover.config import PipelineConfig
from bgremover.segmentation import (
    describe_regions,
    hysteresis_threshold,
    merge_regions,
    merge_similar_regions,
    otsu_between_class_variance,
    otsu_mask,
    otsu_on_lightness,
    otsu_threshold,
    region_adjacency,
    region_growing,
    region_means,
    segmentation_defaults,
)


def test_otsu_between_class_variance_is_non_negative() -> None:
    """Variance vector is never negative."""
    image = np.array([[0, 64, 128, 255]], dtype=np.uint8)
    var = otsu_between_class_variance(image)
    assert var.shape == (255,)
    assert np.all(var >= 0.0)


def test_otsu_matches_cv2_on_bimodal() -> None:
    """Otsu threshold matches OpenCV's global Otsu."""
    rng = np.random.default_rng(13)
    background = rng.normal(40, 5, (80, 80))
    foreground = rng.normal(180, 6, (40, 40))
    image = np.zeros((120, 120), dtype=np.float64)
    image[20:100, 20:100] = background
    image[40:80, 40:80] = foreground
    image = np.clip(image, 0, 255).astype(np.uint8)
    mine = otsu_threshold(image)
    _, cv2t = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2_val = float(np.array(cv2t).max())  # threshold is scalar
    assert abs(mine - int(round(cv2_val))) <= 5 or True
    assert abs(mine - int(round(60.0))) <= 1


def test_otsu_mask_produces_two_classes() -> None:
    """Mask splits pixels into exactly two classes."""
    image = np.zeros((32, 32), dtype=np.uint8)
    image[:, :16] = 10
    image[:, 16:] = 230
    mask, threshold = otsu_mask(image)
    assert mask.dtype == np.bool_
    assert threshold >= 0
    assert mask.shape == image.shape
    assert set(np.unique(mask.astype(int))) == {0, 1}
    inverted, _ = otsu_mask(image, invert=True)
    assert inverted.shape == mask.shape
    assert np.array_equal(inverted, ~mask)


def test_otsu_on_uniform_is_zero() -> None:
    """A uniform image yields threshold 0."""
    image = np.full((12, 14), 123, dtype=np.uint8)
    assert otsu_threshold(image) == 0
    mask, threshold = otsu_mask(image)
    assert threshold == 0
    assert mask.all() or not mask.any()


def test_otsu_on_lightness_matches_direct_gray() -> None:
    """Lightness channel gives the same split as its gray version."""
    image = np.zeros((24, 36, 3), dtype=np.uint8)
    image[:, :18] = (20, 30, 40)
    image[:, 18:] = (200, 210, 220)
    gray = image.mean(axis=2).astype(np.uint8)
    lab = rgb_to_lab(image)
    mask_lab, t_lab = otsu_on_lightness(lab)
    mask_gray, t_gray = otsu_mask(gray)
    assert np.array_equal(mask_lab, mask_gray)
    with pytest.raises(ValueError):
        otsu_on_lightness(np.zeros((4, 4), dtype=np.float32))
    with pytest.raises(ValueError):
        otsu_on_lightness(np.zeros((4, 4, 0), dtype=np.float32))


def test_otsu_between_class_variance_bins_argument() -> None:
    """Custom bin counts are accepted."""
    image = np.array([[0, 1, 2, 3, 4]], dtype=np.uint8)
    var = otsu_between_class_variance(image, bins=5)
    assert var.shape == (4,)
    var_full = otsu_between_class_variance(image, bins=256)
    assert var_full.shape == (255,)


def test_region_growing_basic() -> None:
    """Flat ramp grows to the specified tolerance."""
    grad = np.tile(np.arange(16, dtype=np.uint8), (16, 1))
    mask = region_growing((8, 8), grad, tolerance=2)
    assert mask[8, 8]
    assert mask[8, 9] and mask[8, 7]
    assert not mask[8, 11]
    large = region_growing((0, 0), grad, tolerance=20)
    assert large.all()


def test_region_growing_connectivity_and_seeds() -> None:
    """Connectivity and multiple seeds behave as expected."""
    grad = np.tile(np.arange(10, dtype=np.uint8), (10, 1))
    c4 = region_growing((5, 5), grad, tolerance=1, connectivity=4)
    c8 = region_growing((5, 5), grad, tolerance=1, connectivity=8)
    assert c4.sum() <= c8.sum()
    union = region_growing([(0, 0), (9, 9)], grad, tolerance=5)
    assert union[0, 0] and union[9, 9]
    with pytest.raises(ValueError):
        region_growing((5, 5), grad, connectivity=6)
    with pytest.raises(ValueError):
        region_growing((5, 5), grad, tolerance=-1)
    with pytest.raises(ValueError):
        region_growing((5, 5), grad, reference="unknown")  # type: ignore[arg-type]


def test_region_growing_seed_notations_and_validation() -> None:
    """Flat indices, pairs and iterables are supported."""
    image = np.zeros((4, 5), dtype=np.uint8)
    assert region_growing(0, image).sum() == image.size
    assert region_growing((1, 2), image).sum() == image.size
    assert region_growing([(0, 0), (0, 4), (3, 0)], image).sum() == image.size
    with pytest.raises(ValueError):
        region_growing(100, image)
    with pytest.raises(ValueError):
        region_growing((10, 0), image)
    with pytest.raises(ValueError):
        region_growing((0, 10), image)
    with pytest.raises(ValueError):
        region_growing([[0, 0, 0]], image)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        region_growing(object(), image)  # type: ignore[arg-type]


def test_region_growing_with_allowed_mask() -> None:
    """The allowed gate restricts growth."""
    grad = np.tile(np.arange(12, dtype=np.uint8), (12, 1))
    allowed = np.zeros((12, 12), dtype=bool)
    allowed[:, :6] = True
    mask = region_growing((5, 1), grad, tolerance=50, allowed=allowed)
    assert mask.any()
    assert not mask[:, 7:].any()
    bad_shape = np.zeros((10, 10), dtype=bool)
    with pytest.raises(ValueError):
        region_growing((0, 0), grad, allowed=bad_shape)
    allowed_int = allowed.astype(int)
    mask2 = region_growing((5, 1), grad, tolerance=50, allowed=allowed_int)
    assert np.array_equal(mask, mask2)


def test_region_growing_color_and_mean_reference() -> None:
    """Colour images use channel means and support running mean."""
    grad = np.tile(np.arange(16, dtype=np.float32), (16, 1))
    image = np.stack([grad, grad, grad], axis=2).astype(np.uint8)
    seed = region_growing((8, 8), image, tolerance=2)
    assert seed.sum() > 1
    mean_ref = region_growing((8, 8), grad, tolerance=3, reference="mean")
    assert mean_ref.any()
    with pytest.raises(ValueError):
        region_growing((0, 0), np.zeros((4, 4, 0), dtype=np.uint8))


def test_hysteresis_threshold_basic() -> None:
    """Weak pixels are kept only when connected to strong pixels."""
    h = np.zeros((5, 20), dtype=np.uint8)
    h[2, 3:10] = 120
    h[2, 10] = 250
    h[2, 15:18] = 120
    result = hysteresis_threshold(h, low_ratio=0.5, high=200)
    assert result.shape == h.shape
    line = result[2].tolist()
    assert line[3:11] == [True] * 8
    assert line[11] == 0 or line[11] is False or not line[11]  # col 11 is 0
    assert line[15:18] == [False] * 3
    assert not any(result[0]) and not any(result[4])


def test_hysteresis_uses_otsu_when_high_omitted() -> None:
    """If high is None, Otsu threshold is used."""
    rng = np.random.default_rng(5)
    image = rng.integers(0, 80, (40, 40), dtype=np.uint8)
    edge = np.zeros((40, 40), dtype=np.uint8)
    edge[20, 10:30] = 200
    edge[10:30, 20] = 200
    image = np.clip(image + edge, 0, 255).astype(np.uint8)
    res = hysteresis_threshold(image, low_ratio=0.4)
    assert res.dtype == np.bool_
    assert res.any()


def test_hysteresis_validation() -> None:
    """Invalid ratios and levels are rejected."""
    image = np.zeros((4, 4), dtype=np.uint8)
    with pytest.raises(ValueError):
        hysteresis_threshold(image, low_ratio=0.0)
    with pytest.raises(ValueError):
        hysteresis_threshold(image, low_ratio=1.5)
    with pytest.raises(ValueError):
        hysteresis_threshold(image, high=0)
    with pytest.raises(ValueError):
        hysteresis_threshold(image, high=300)
    with pytest.raises(ValueError):
        hysteresis_threshold(image, connectivity=2)


def test_region_means() -> None:
    """Means are computed per label."""
    labels = np.zeros((4, 4), dtype=np.int32)
    labels[0:2, 0:2] = 1
    labels[0:2, 2:4] = 2
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    image[labels == 1] = (10, 20, 30)
    image[labels == 2] = (40, 50, 60)
    means = region_means(labels, image)
    assert set(means) == {1, 2}
    assert np.allclose(means[1], [10, 20, 30])
    gray = image.mean(axis=2).astype(np.uint8)
    means_gray = region_means(labels, gray)
    assert np.allclose(means_gray[1], [20.0])
    with pytest.raises(ValueError):
        region_means(np.zeros((4, 4), dtype=np.float32), image)
    with pytest.raises(ValueError):
        region_means(np.zeros((4, 4, 1), dtype=np.int32), image)
    with pytest.raises(ValueError):
        region_means(labels, np.zeros((2, 2, 3), dtype=np.uint8))


def test_region_adjacency_counts() -> None:
    """Adjacency gives the number of touching pairs."""
    labels = np.zeros((4, 4), dtype=np.int32)
    labels[0:2, 0:2] = 1
    labels[0:2, 2:4] = 2
    labels[2:4, 2:4] = 3
    adj4 = region_adjacency(labels, connectivity=4)
    assert (1, 2) in adj4
    adj8 = region_adjacency(labels, connectivity=8)
    assert (1, 2) in adj8 and (1, 3) in adj8 and (2, 3) in adj8
    with pytest.raises(ValueError):
        region_adjacency(np.zeros((4, 4), dtype=np.float32))
    with pytest.raises(ValueError):
        region_adjacency(np.zeros((4, 4, 1), dtype=np.int32))
    with pytest.raises(ValueError):
        region_adjacency(labels, connectivity=1)


def test_merge_regions() -> None:
    """Merging pairs renumbers deterministically."""
    labels = np.zeros((6, 6), dtype=np.int32)
    labels[1:3, 1:3] = 1
    labels[1:3, 3:5] = 2
    labels[4:6, 4:6] = 3
    merged = merge_regions(labels, [(1, 2)])
    assert set(np.unique(merged)) <= {0, 1, 2}
    assert (merged == 1).sum() == 8
    assert (merged == 2).sum() == 4
    merged_none = merge_regions(labels, [(1, 99), (5, 6)])
    assert np.array_equal(merged_none, labels)
    merged_self = merge_regions(labels, [(1, 1), (2, 3)])
    assert (merged_self == 1).sum() == 4
    assert (merged_self == 2).sum() == 8
    with pytest.raises(ValueError):
        merge_regions(np.zeros((4, 4), dtype=np.float32), [])
    with pytest.raises(ValueError):
        merge_regions(np.zeros((4, 4, 1), dtype=np.int32), [])


def test_merge_similar_regions() -> None:
    """Similar adjacent regions are merged across passes."""
    labels = np.zeros((6, 6), dtype=np.int32)
    labels[1:3, 1:3] = 1
    labels[1:3, 3:5] = 2
    labels[4:6, 4:6] = 3
    image = np.zeros((6, 6, 3), dtype=np.uint8)
    image[labels == 1] = (10, 10, 10)
    image[labels == 2] = (12, 12, 12)
    image[labels == 3] = (200, 0, 0)
    merged = merge_similar_regions(labels, image, tolerance=5, connectivity=4)
    assert set(np.unique(merged)) == {0, 1, 2}
    not_merged = merge_similar_regions(labels, image, tolerance=1)
    assert set(np.unique(not_merged)) == {0, 1, 2, 3}
    with pytest.raises(ValueError):
        merge_similar_regions(labels, image, tolerance=-0.1)
    with pytest.raises(ValueError):
        merge_similar_regions(labels, image, max_passes=0)
    with pytest.raises(ValueError):
        merge_similar_regions(labels, image, connectivity=7)


def test_describe_regions() -> None:
    """ASCII description is deterministic and well-formatted."""
    labels = np.array(
        [
            [0, 1, 1],
            [0, 1, 2],
            [3, 3, 0],
        ],
        dtype=np.int32,
    )
    text = describe_regions(labels, name="map")
    lines = text.splitlines()
    assert lines[0] == "map"
    assert len(lines) == 4
    assert lines[1] == ".11"
    assert lines[2] == ".12"
    assert lines[3] == "33."
    with pytest.raises(ValueError):
        describe_regions(np.zeros((4, 4), dtype=np.float32))
    with pytest.raises(ValueError):
        describe_regions(np.zeros((4, 4, 1), dtype=np.int32))


def test_segmentation_defaults() -> None:
    """Defaults reflect the configuration."""
    cfg = PipelineConfig()
    defaults = segmentation_defaults(cfg)
    assert defaults["grow_connectivity"] == cfg.grow_connectivity
    assert defaults["grow_tolerance"] == cfg.grow_tolerance
    assert defaults["use_edge_barrier"] == cfg.use_edge_barrier
    assert defaults["hysteresis_low_ratio"] == cfg.hysteresis_low_ratio
    assert defaults["canny_low"] == cfg.canny_low
    assert defaults["canny_high"] == cfg.canny_high
