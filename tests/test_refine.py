"""Tests for alpha mattes, decontamination and RGBA export."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from bgremover.refine import (
    alpha_from_mask,
    compose_rgba,
    decontaminate_edges,
    upscale_mask,
)


def test_alpha_from_mask_hard_when_not_feathered() -> None:
    """Zero sigma or zero band returns the mask as float."""
    mask = np.zeros((12, 12), dtype=bool)
    mask[3:9, 3:9] = True
    for kwargs in ({"feather_sigma": 0.0}, {"band_width": 0}):
        alpha = alpha_from_mask(mask, **kwargs)  # type: ignore[arg-type]
        assert alpha.dtype == np.float32
        assert np.array_equal(alpha, mask.astype(np.float32))
    with pytest.raises(ValueError):
        alpha_from_mask(mask, feather_sigma=-1.0)
    with pytest.raises(ValueError):
        alpha_from_mask(mask, band_width=-2)
    with pytest.raises(ValueError):
        alpha_from_mask(np.zeros((4, 4, 3), dtype=bool))
    with pytest.raises(ValueError):
        alpha_from_mask(np.full((4, 4), 7, dtype=np.uint8))


def test_alpha_from_mask_is_band_limited() -> None:
    """Only the boundary band turns soft; solid areas stay 0/1."""
    mask = np.zeros((40, 40), dtype=bool)
    mask[10:30, 10:30] = True
    alpha = alpha_from_mask(mask, feather_sigma=1.5, band_width=4)
    assert alpha.shape == mask.shape
    assert alpha.dtype == np.float32
    assert float(alpha.min()) >= 0.0 and float(alpha.max()) <= 1.0
    assert float(alpha[20, 20]) == 1.0
    assert float(alpha[0, 0]) == 0.0
    assert float(alpha[35, 35]) == 0.0
    assert bool(((alpha > 0.0) & (alpha < 1.0)).any())
    assert not alpha_from_mask(
        np.zeros((8, 8), dtype=bool), 1.5, 4
    ).any()
    assert alpha_from_mask(np.ones((8, 8), dtype=bool), 1.5, 4).all()


def test_decontaminate_edges_propagates_opaque_colours() -> None:
    """Semi pixels take the nearest opaque colour; the rest is kept."""
    rgb = np.full((5, 5, 3), (10, 20, 200), dtype=np.uint8)
    rgb[2, 2] = (220, 30, 30)
    alpha = np.zeros((5, 5), dtype=np.float32)
    alpha[2, 2] = 1.0
    alpha[1, 2] = alpha[3, 2] = alpha[2, 1] = alpha[2, 3] = 0.5
    fixed = decontaminate_edges(rgb, alpha)
    assert fixed.shape == rgb.shape
    assert fixed.dtype == np.uint8
    assert (fixed[2, 2] == (220, 30, 30)).all()
    for row, col in ((1, 2), (3, 2), (2, 1), (2, 3)):
        assert (fixed[row, col] == (220, 30, 30)).all()
    assert (fixed[0, 0] == (10, 20, 200)).all()
    assert (fixed[4, 4] == (10, 20, 200)).all()


def test_decontaminate_edges_degenerate_cases() -> None:
    """No semi pixels (or no opaque pixel) returns an unchanged copy."""
    rgb = np.full((4, 4, 3), (1, 2, 3), dtype=np.uint8)
    hard = np.zeros((4, 4), dtype=np.float32)
    hard[:2] = 1.0
    assert np.array_equal(decontaminate_edges(rgb, hard), rgb)
    assert np.array_equal(
        decontaminate_edges(rgb, np.full((4, 4), 0.5, np.float32)), rgb
    )
    with pytest.raises(ValueError):
        decontaminate_edges(rgb, np.zeros((3, 3), dtype=np.float32))
    with pytest.raises(ValueError):
        decontaminate_edges(rgb[:, :, 0], hard)


def test_compose_rgba() -> None:
    """Alpha 0/0.5/1 maps to 0/128/255 in the fourth channel."""
    rgb = np.full((3, 4, 3), (10, 20, 30), dtype=np.uint8)
    alpha = np.array(
        [[0.0, 0.5, 1.0, 0.25], [1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    rgba = compose_rgba(rgb, alpha)
    assert rgba.shape == (3, 4, 4)
    assert rgba.dtype == np.uint8
    assert np.array_equal(rgba[..., :3], rgb)
    assert rgba[0, 0, 3] == 0
    assert rgba[0, 1, 3] == 128
    assert rgba[0, 2, 3] == 255
    assert rgba[0, 3, 3] == 64
    assert np.array_equal(
        compose_rgba(rgb, np.ones((3, 4), dtype=bool))[..., 3],
        np.full((3, 4), 255, dtype=np.uint8),
    )
    with pytest.raises(ValueError):
        compose_rgba(rgb, np.zeros((2, 2), dtype=np.float32))
    with pytest.raises(ValueError):
        compose_rgba(rgb.astype(np.float32), alpha)


def test_upscale_mask() -> None:
    """Boolean masks stay boolean, float mattes stay in [0, 1]."""
    mask = np.zeros((4, 4), dtype=bool)
    mask[:2, :2] = True
    big = upscale_mask(mask, (8, 8))
    assert big.dtype == np.bool_
    assert big.shape == (8, 8)
    assert big[:4, :4].all() and not big[4:, 4:].any()
    assert np.array_equal(upscale_mask(mask, (4, 4)), mask)
    matte = np.linspace(0.0, 1.0, 16, dtype=np.float32).reshape(4, 4)
    grown = upscale_mask(matte, (9, 7))
    assert grown.dtype == np.float32
    assert grown.shape == (9, 7)
    assert float(grown.min()) >= 0.0 and float(grown.max()) <= 1.0
    expected = cv2.resize(matte, (7, 9), interpolation=cv2.INTER_LINEAR)
    assert np.allclose(grown, np.clip(expected, 0.0, 1.0), atol=1e-5)
    with pytest.raises(ValueError):
        upscale_mask(mask, (0, 8))
    with pytest.raises(ValueError):
        upscale_mask(np.zeros((4, 4), dtype=np.uint8), (8, 8))
    with pytest.raises(ValueError):
        upscale_mask(np.zeros((4, 4, 3), dtype=bool), (8, 8))
