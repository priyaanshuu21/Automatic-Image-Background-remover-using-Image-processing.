"""Tests for binary and grayscale morphology with mask cleanup."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from bgremover.config import PipelineConfig
from bgremover.morphology import (
    black_hat,
    clean_mask,
    closing,
    dilate,
    erode,
    fill_holes,
    hit_or_miss,
    morphological_gradient,
    morphology_defaults,
    opening,
    remove_objects_touching_border,
    remove_small_objects,
    structuring_element,
    thickening,
    thinning,
    top_hat,
)


def test_structuring_element_shapes() -> None:
    """Disc, square and cross elements have the expected geometry."""
    assert structuring_element("square", 1).shape == (3, 3)
    assert bool(structuring_element("square", 1).all())
    disk = structuring_element("disk", 2)
    assert disk.shape == (5, 5)
    assert disk[2, 2] and not disk[0, 0]
    cross = structuring_element("cross", 1)
    assert int(cross.sum()) == 5
    assert structuring_element(size=(2, 4)).shape == (2, 4)
    with pytest.raises(ValueError):
        structuring_element("triangle", 1)
    with pytest.raises(ValueError):
        structuring_element("disk", -1)


def test_dilate_matches_opencv() -> None:
    """Dilation agrees exactly with OpenCV."""
    gray = np.zeros((24, 24), dtype=np.uint8)
    gray[8:16, 8:16] = 200
    kernel = np.ones((3, 3), dtype=np.uint8)
    assert np.array_equal(dilate(gray, kernel), cv2.dilate(gray, kernel))
    with pytest.raises(ValueError):
        dilate(gray, np.zeros((3, 3), dtype=bool))
    with pytest.raises(ValueError):
        dilate(gray, np.ones((2, 2), dtype=bool))
    with pytest.raises(ValueError):
        dilate(gray, kernel, iterations=0)


def test_erode_matches_opencv() -> None:
    """Erosion agrees exactly with OpenCV under zero padding."""
    gray = np.zeros((24, 24), dtype=np.uint8)
    gray[8:16, 8:16] = 200
    kernel = np.ones((3, 3), dtype=np.uint8)
    expected = cv2.erode(
        gray, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0
    )
    result = erode(gray, kernel)
    assert result.dtype == np.uint8
    assert np.array_equal(result, expected)


def test_opening_closing_gradient_hats() -> None:
    """Composed operators behave classically on a speckled block."""
    gray = np.zeros((24, 24), dtype=np.uint8)
    gray[6:18, 6:18] = 200
    gray[0, 0] = 200
    kernel = np.ones((3, 3), dtype=np.uint8)
    opened = opening(gray, kernel)
    assert opened[0, 0] == 0
    assert opened[12, 12] == 200
    assert np.array_equal(
        opened,
        cv2.morphologyEx(
            gray,
            cv2.MORPH_OPEN,
            kernel,
            borderType=cv2.BORDER_CONSTANT,
            borderValue=0,
        ),
    )
    closed = closing(gray, kernel)
    assert np.array_equal(
        closed,
        cv2.morphologyEx(
            gray,
            cv2.MORPH_CLOSE,
            kernel,
            borderType=cv2.BORDER_CONSTANT,
            borderValue=0,
        ),
    )
    gradient = morphological_gradient(gray, kernel)
    assert gradient.shape == gray.shape
    assert float(gradient.max()) > 0.0
    assert np.array_equal(
        top_hat(gray, kernel),
        gray.astype(float) - opened.astype(float),
    )
    assert np.array_equal(
        black_hat(gray, kernel),
        closed.astype(float) - gray.astype(float),
    )


def test_hit_or_miss_finds_corners() -> None:
    """The transform fires exactly where the kernel pair fits."""
    image = np.zeros((9, 9), dtype=bool)
    image[2:7, 2:7] = True
    image[2, 2] = False
    hit = np.array(
        [[False, False, False], [False, False, True], [False, True, True]]
    )
    miss = np.array(
        [[True, True, True], [True, True, False], [True, False, False]]
    )
    result = hit_or_miss(image, hit, miss)
    assert result.dtype == np.bool_
    assert result[2, 2]
    assert int(result.sum()) == 1
    with pytest.raises(ValueError):
        hit_or_miss(image, hit, hit)
    with pytest.raises(ValueError):
        hit_or_miss(image, hit, np.ones((5, 5), dtype=bool))
    with pytest.raises(ValueError):
        hit_or_miss(image, np.zeros((3, 3), dtype=bool), miss)
    with pytest.raises(ValueError):
        hit_or_miss(np.full((4, 4), 7, dtype=np.uint8), hit, miss)


def test_thinning_reduces_and_is_idempotent() -> None:
    """Thinning shrinks a block, stays inside it and converges."""
    block = np.zeros((24, 24), dtype=bool)
    block[6:18, 6:18] = True
    thin = thinning(block)
    assert thin.dtype == np.bool_
    assert int(thin.sum()) < int(block.sum())
    assert not np.any(thin & ~block)
    assert np.array_equal(thin, thinning(thin))
    with pytest.raises(ValueError):
        thinning(block, max_iter=0)


def test_thickening_grows_and_is_idempotent() -> None:
    """Thickening fills a notch, covers the input and converges."""
    shape = np.zeros((20, 20), dtype=bool)
    shape[4:16, 4:16] = True
    shape[9:11, 8:16] = False
    thick = thickening(shape)
    assert thick.dtype == np.bool_
    assert int(thick.sum()) > int(shape.sum())
    assert not np.any(shape & ~thick)
    assert np.array_equal(thick, thickening(thick))
    with pytest.raises(ValueError):
        thickening(shape, max_iter=0)


def test_clean_mask_removes_specks() -> None:
    """Opening drops an isolated speck and keeps the block intact."""
    mask = np.zeros((24, 24), dtype=bool)
    mask[6:18, 6:18] = True
    mask[0, 0] = True
    cleaned = clean_mask(mask, open_radius=1, close_radius=0)
    assert cleaned.dtype == np.bool_
    assert not cleaned[0, 0]
    assert cleaned[7:17, 7:17].all()
    # The plus-shaped element does not fit the sharp convex corners,
    # so exactly the four corner pixels go missing: textbook behaviour.
    assert int(cleaned.sum()) == 12 * 12 - 4
    assert np.array_equal(clean_mask(mask, 0, 0), mask)
    with pytest.raises(ValueError):
        clean_mask(mask, open_radius=-1)


def test_clean_mask_seals_a_short_gap() -> None:
    """Closing seals a one-pixel slit fully inside the block."""
    mask = np.zeros((24, 24), dtype=bool)
    mask[6:18, 6:18] = True
    mask[12, 8:16] = False
    mask[12, 11] = True
    cleaned = clean_mask(mask, open_radius=0, close_radius=1)
    assert cleaned.dtype == np.bool_
    assert cleaned[12, 8:16].all()


def test_fill_holes() -> None:
    """Interior background components are filled; exterior stays."""
    mask = np.zeros((16, 16), dtype=bool)
    mask[3:13, 3:13] = True
    mask[6:10, 6:10] = False
    filled = fill_holes(mask)
    assert filled[6:10, 6:10].all()
    assert not filled[0, 0]
    assert not filled[0, :].any()
    with pytest.raises(ValueError):
        fill_holes(mask, connectivity=6)


def test_remove_small_objects() -> None:
    """Components below the area cutoff disappear."""
    mask = np.zeros((16, 16), dtype=bool)
    mask[2:4, 2:4] = True
    mask[8:14, 8:14] = True
    kept = remove_small_objects(mask, min_size=9)
    assert int(kept.sum()) == 36
    assert not kept[2:4, 2:4].any()
    assert not remove_small_objects(mask, min_size=1000).any()
    with pytest.raises(ValueError):
        remove_small_objects(mask, min_size=0)


def test_remove_objects_touching_border() -> None:
    """Border-touching components disappear; interior ones survive."""
    mask = np.zeros((16, 16), dtype=bool)
    mask[0:3, 0:3] = True
    mask[8:12, 8:12] = True
    kept = remove_objects_touching_border(mask)
    assert int(kept.sum()) == 16
    assert not kept[0:3, 0:3].any()
    interior = np.zeros((16, 16), dtype=bool)
    interior[8:12, 8:12] = True
    assert np.array_equal(
        remove_objects_touching_border(interior), interior
    )


def test_morphology_defaults() -> None:
    """Defaults mirror the pipeline configuration."""
    cfg = PipelineConfig()
    defaults = morphology_defaults(cfg)
    assert defaults["open_radius"] == cfg.morph_open_radius
    assert defaults["close_radius"] == cfg.morph_close_radius
    assert defaults["min_object_frac"] == cfg.min_object_frac
    assert defaults["fill_hole_frac"] == cfg.fill_hole_frac
    assert defaults["keep_largest"] == cfg.keep_largest
