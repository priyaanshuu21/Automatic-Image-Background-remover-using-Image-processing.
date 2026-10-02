"""Robustness against natural-photo-like synthetic scenes.

Studio objects with soft shadows, portraits whose clothing touches the
image border and multi-tone backgrounds exercise the zoned seeding,
the energy-gated growing, the topological solidifier and the saliency
prior far beyond the flat fixtures of ``conftest.py``.
"""

from __future__ import annotations

import numpy as np

from bgremover.color import (
    estimate_zoned_background_model,
    rgb_to_lab,
)
from bgremover.config import PipelineConfig
from bgremover.edges import canny
from bgremover.filters import denoise
from bgremover.fundamentals import to_gray
from bgremover.morphology import dilate, structuring_element
from bgremover.pipeline import STAGE_KEYS, BackgroundRemovalPipeline


def studio_mug() -> tuple[np.ndarray, np.ndarray]:
    """White mug with a ring handle over a gray studio sweep + shadow."""
    height, width = 140, 140
    rows = np.linspace(0.0, 1.0, height)[:, None]
    background = (
        np.array([120.0, 125.0, 130.0])
        + np.array([30.0, 25.0, 20.0]) * rows
    ).astype(np.uint8)
    background = np.repeat(background[:, None, :], width, axis=1)
    grid_y, grid_x = np.mgrid[0:height, 0:width]
    body = (grid_x - 70) ** 2 / 28.0 ** 2 + (grid_y - 75) ** 2 / 38.0 ** 2
    body = body <= 1.0
    ring = (grid_x - 102) ** 2 + (grid_y - 75) ** 2 <= 12.0 ** 2
    hole = (grid_x - 102) ** 2 + (grid_y - 75) ** 2 <= 6.0 ** 2
    mug = body | (ring & ~hole)
    shadow = (
        (grid_x - 70) ** 2 / 40.0 ** 2 + (grid_y - 118) ** 2 / 8.0 ** 2
    ) <= 1.0
    image = background.copy()
    image[mug] = (238, 238, 240)
    darkened = (background.astype(float) * 0.72).astype(np.uint8)
    image[shadow & ~mug] = darkened[shadow & ~mug]
    return image, mug


def portrait_touching() -> tuple[np.ndarray, np.ndarray]:
    """Head disc plus a torso widening onto the bottom image edge."""
    height, width = 140, 140
    blend = np.linspace(0.0, 1.0, height)[:, None, None]
    start = np.array([40.0, 105.0, 200.0])
    end = np.array([50.0, 140.0, 195.0])
    background = (start * (1.0 - blend) + end * blend).astype(np.uint8)
    background = np.repeat(background, width, axis=1)
    grid_y, grid_x = np.mgrid[0:height, 0:width]
    head = (grid_x - 70) ** 2 + (grid_y - 52) ** 2 <= 20.0 ** 2
    torso = (
        (grid_y >= 66)
        & (np.abs(grid_x - 70) <= 14 + (grid_y - 66) * 0.55)
        & (grid_y < height)
    )
    person = head | torso
    image = background.copy()
    image[head] = (205, 165, 145)
    image[torso] = (65, 85, 150)
    return image, person


def twotone_disc() -> tuple[np.ndarray, np.ndarray]:
    """Red disc straddling a blue/green two-tone background."""
    height, width = 120, 120
    background = np.zeros((height, width, 3), dtype=np.uint8)
    background[:, :60] = (40, 40, 200)
    background[:, 60:] = (40, 200, 60)
    grid_y, grid_x = np.mgrid[0:height, 0:width]
    disc = (grid_x - 60) ** 2 + (grid_y - 60) ** 2 <= 25.0 ** 2
    image = background.copy()
    image[disc] = (220, 35, 35)
    return image, disc


def _iou(mask: np.ndarray, truth: np.ndarray) -> float:
    """Intersection over union of two boolean masks."""
    union = np.count_nonzero(mask | truth)
    if union == 0:
        return 1.0
    return np.count_nonzero(mask & truth) / union


def test_studio_mug_with_shadow() -> None:
    """A white mug on a swept gradient segments almost perfectly."""
    image, truth = studio_mug()
    mask = BackgroundRemovalPipeline().run(image).mask
    assert _iou(mask, truth) >= 0.90


def test_studio_handle_hole_is_filled() -> None:
    """The enclosed handle opening must be forcefully foreground."""
    image, truth = studio_mug()
    mask = BackgroundRemovalPipeline().run(image).mask
    grid_y, grid_x = np.mgrid[0:140, 0:140]
    opening = (grid_x - 102) ** 2 + (grid_y - 75) ** 2 <= 5.0 ** 2
    # Only the handle-body attachment pixels are truth here; the rest
    # is the enclosed opening showing background through it.
    assert int(truth[opening].sum()) <= 4
    assert mask[opening].all()


def test_portrait_with_border_touching_clothing() -> None:
    """Torso on the bottom edge must not corrupt the result."""
    image, truth = portrait_touching()
    mask = BackgroundRemovalPipeline().run(image).mask
    assert _iou(mask, truth) >= 0.80


def test_portrait_model_ignores_skin_and_shirt() -> None:
    """Neither skin nor shirt colour may enter the background model."""
    image, _ = portrait_touching()
    cfg = PipelineConfig()
    denoised = denoise(image, cfg)
    edges = canny(to_gray(denoised), cfg.canny_low, cfg.canny_high)
    model = estimate_zoned_background_model(denoised, cfg, edges)
    skin = rgb_to_lab(np.array([[[205, 165, 145]]], dtype=np.uint8))
    skin = skin.ravel()
    shirt = rgb_to_lab(np.array([[[65, 85, 150]]], dtype=np.uint8))
    shirt = shirt.ravel()
    # Skin is far from any background; the shirt sits 14.3 units away,
    # which is exactly why uniform border-touching regions need the
    # model-verified seed zones (a dragged model would sit under 5).
    for mode in model.modes:
        assert float(np.linalg.norm(mode - skin)) > 20.0
        assert float(np.linalg.norm(mode - shirt)) > 10.0


def test_twotone_background_with_two_modes() -> None:
    """A two-tone background needs both colour modes to segment."""
    image, truth = twotone_disc()
    pipeline = BackgroundRemovalPipeline(
        PipelineConfig(n_background_modes=2)
    )
    assert _iou(pipeline.run(image).mask, truth) >= 0.85


def test_stage_keys_keep_gui_contract() -> None:
    """The ten stage keys arrive in the order the GUI listbox shows."""
    image, _ = studio_mug()
    stages = BackgroundRemovalPipeline().run(image).stages
    assert list(stages) == list(STAGE_KEYS)


def test_guided_alpha_stays_in_a_narrow_band() -> None:
    """Semi-transparent pixels hug the mask boundary (3-5 px)."""
    image, _ = portrait_touching()
    result = BackgroundRemovalPipeline().run(image)
    alpha = result.alpha
    semi = (alpha > 0.0) & (alpha < 1.0)
    assert semi.any()
    near = dilate(
        result.mask, kernel=structuring_element("disk", 6)
    ) | dilate(
        ~result.mask, kernel=structuring_element("disk", 6)
    )
    assert bool((semi & ~near).sum() == 0)
